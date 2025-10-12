import mediapipe as mp
import numpy as np
import time, math
from statistics import mean, pstdev

EXERCISE_CONFIG = {
    'chest_pull': {
        'name': '胸部拉伸',
        'landmarks_to_use': ['right_shoulder', 'right_wrist'],
        'logic_function': '_analyze_chest_pull_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': -0.02,
                'end_threshold_y': 0.02,
                'over_extension_threshold_y': -0.2,
                'min_distance': 0.02
            },
            'beginner': {
                'start_threshold_y': -0.015,
                'end_threshold_y': 0.03,
                'over_extension_threshold_y': -0.18,
                'min_distance': 0.015
            }
        }
    },
    'lateral_raise': {
        'name': '侧平举',
        'landmarks_to_use': ['right_shoulder', 'right_elbow', 'right_wrist'],
        'logic_function': '_analyze_lateral_raise_logic',
        'params': {
            'intermediate': {
                'start_threshold_x': 0.15,
                'end_threshold_x': 0.20,
                'over_extension_threshold_y': -0.2,
                'min_distance': 0.02
            },
            'beginner': {
                'start_threshold_x': 0.15,
                'end_threshold_x': 0.18,
                'over_extension_threshold_y': -0.18,
                'min_distance': 0.02
            }
        }
    },
    'front_raise': {
        'name': '前平举',
        'landmarks_to_use': ['right_shoulder', 'right_wrist', 'right_elbow'],
        'logic_function': '_analyze_front_raise_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': 0.18, 
                'end_threshold_y': 0.15,   
                'over_extension_threshold_y': -0.25,
                'min_distance': 0.015
            },
            'beginner': {
                'start_threshold_y': 0.25,
                'end_threshold_y': 0.20,
                'over_extension_threshold_y': -0.25,
                'min_distance': 0.010
            }
        }
    },
    'overhead_press': {
        'name': '过顶举',
        'landmarks_to_use': ['right_shoulder', 'right_wrist', 'right_elbow'],
        'logic_function': '_analyze_overhead_press_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': 0.05, 
                'end_threshold_y': -0.2,   
                'min_distance': 0.02
            },
            'beginner': {
                'start_threshold_y': 0.08,
                'end_threshold_y': -0.18,
                'min_distance': 0.01
            }
        }
    },
    'squat': {
        'name': '深蹲',
        'landmarks_to_use': ['right_hip', 'right_knee', 'right_ankle'],
        'logic_function': '_analyze_squat_logic',
        'params': {
            'intermediate': {
                'up_threshold_angle': 165.0,
                'down_threshold_angle': 95.0
            },
            'beginner': {
                'up_threshold_angle': 160.0,
                'down_threshold_angle': 110.0
            }
        }
    }

}


class WorkoutState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.stage = None
        self.feedback = "请准备开始"
        self.is_paused = False
        self._start_position = None
        self._end_position = None
        self._overextension_detected = False
        self._overextension_type = None
        self._action_active = False
        self.total_distance = 0.0
        self.total_energy = 0.0
        self.BAND_RESISTANCE_N = 25 * 9.81
        self._last_completion_category = 'standard'
        self._completed_this_frame = False
        self._completed_hold_frames = 0
        self._action_overextended = False

class ExerciseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.state = WorkoutState()
        self.config = None
        self.style = 'intermediate'
        self.exercise_id = None
        #new
        self.last_rep_start = None
        self.repetition_durations = []  
        self.smoothness_score = 100

    def _now(self):
        return time.time()

    # def on_phase_change(self, new_phase: str):
    #     t = self._now()
    #     # 记录开始时间
    #     if self.phase is None:
    #         # 第一次识别到相位，认为是 rep 的起点
    #         self.last_rep_start = t
    #     self.phase = new_phase

    def _on_rep_completed(self):
        """
        在任意动作完成一次有效计数后调用。
        负责：
        - 计算本次 repetition 的时长并入库（以 last_rep_start 为基准）
        - 更新 smoothness_score
        - 重新设定 last_rep_start 作为下一次的起点
        """
        now_t = self._now()
        # 第一次完成时若 last_rep_start 未设，直接将当前作为起点
        if self.last_rep_start is None:
            self.last_rep_start = now_t
            return

        # 计算本次时长
        duration = max(0.0, now_t - self.last_rep_start)
        # 合理区间过滤，避免误触发
        if 0.2 <= duration <= 10.0:
            self.repetition_durations.append(duration)
        # 更新下一次计时起点
        self.last_rep_start = now_t

        # 更新平滑度分数
        self.smoothness_score = self._compute_smoothness()

    def _compute_smoothness(self) -> int:
        """
        使用 RSD = std/mean 将节奏一致性映射到 10–100。
        - 少于 3 次时，返回 100（或你可以改成 None）
        - rsd_min = 0.02（几乎完美），rsd_max = 0.5（非常不稳定）
        """
        n = len(self.repetition_durations)
        if n < 3:
            return 100

        m = mean(self.repetition_durations)
        if m <= 1e-6 or not math.isfinite(m):
            return 100

        sigma = pstdev(self.repetition_durations)
        # 变异系数（相对标准差），越小越稳定
        rsd = sigma / m if m > 0 else float('inf')

        # 标定区间与线性映射到 [10, 100]
        rsd_min, rsd_max = 0.02, 0.5
        rsd = max(rsd_min, min(rsd, rsd_max))
        ratio = (rsd - rsd_min) / (rsd_max - rsd_min)  # 0（好）→1（差）
        score = 10 + (100 - 10) * (1.0 - ratio)        # 10（差）→100（好）

        # 四舍五入为整数
        return int(round(score))

    def get_metrics(self):
        # 供路由返回
        return {
            "rep_count": self.rep_count,
            "smoothness": self.smoothness_score,
            "rep_durations": self.repetition_durations[-20:],  # 可限长返回
        }

    def setup(self, exercise_type: str, style: str):
        if exercise_type not in EXERCISE_CONFIG:
            raise ValueError(f"不支持的运动类型: {exercise_type}")
        
        self.config = EXERCISE_CONFIG[exercise_type]
        self.style = style if style in ['intermediate', 'beginner'] else 'intermediate'
        self.exercise_id = exercise_type
        self.reset()
        print(f"分析器已设置为: 运动='{self.config['name']}', 模式='{self.style}' (使用专属评价标准)")

    def reset(self):
        self.state.reset()

    def _get_landmarks(self, results):
        landmarks = {}
        if results.pose_landmarks:
            for landmark_name in self.config['landmarks_to_use']:
                lm = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark[landmark_name.upper()]]
                if lm.visibility > 0.5:
                    landmarks[landmark_name] = np.array([lm.x, lm.y, lm.z])
                else:
                    return None
        return landmarks

    def process(self, image):
        if not self.config:
            raise RuntimeError("分析器未设置。请先调用 setup() 方法。")

        image_rgb = image
        results = self.pose.process(image_rgb)
        landmarks = self._get_landmarks(results)
        smoothness = self.smoothness_score
        # 重置当帧的过伸标记
        self.state._overextension_detected = False
        self.state._overextension_type = None

        if self.state._completed_hold_frames > 0:
            self.state._completed_hold_frames -= 1
            self.state._completed_this_frame = True
        else:
            self.state._completed_this_frame = False
            self.state._last_completion_category = 'standard'

        if self.state.is_paused:
            self.state.feedback = "已暂停，做点赞手势继续"
            return {
                'count': self.state.count,
                'stage': self.state.stage,
                'feedback': self.state.feedback,
                'paused': True,
                'energy': self.state.total_energy,
                'overextended': self.state._overextension_detected,
                'completed': self.state._completed_this_frame,
                'category': self.state._last_completion_category if self.state._completed_this_frame else ('non_standard' if self.state._overextension_detected else 'standard'),
                'smoothness': self.smoothness_score,
                'rep_durations': self.repetition_durations[-20:]
            }

        if landmarks:
            logic_function = getattr(self, self.config['logic_function'])
            logic_function(landmarks)
        else:
            self.state.feedback = "请确保身体关键部位在镜头内"

        self._generate_feedback()

        return {
            'count': self.state.count,
            'stage': self.state.stage,
            'feedback': self.state.feedback,
            'paused': self.state.is_paused,
            'energy': self.state.total_energy,
            'overextended': self.state._overextension_detected,
            'completed': self.state._completed_this_frame,
            'category': self.state._last_completion_category if self.state._completed_this_frame else ('non_standard' if self.state._overextension_detected else 'standard'),
            'smoothness': self.smoothness_score,
            'rep_durations': self.repetition_durations[-20:]
        }
    
    
    ## new function
    def _calculate_angle(self, a, b, c):
            a = np.array(a)  # 第一个点
            b = np.array(b)  # 中间点 (角度所在顶点)
            c = np.array(c)  # 第三个点
            
            # 使用arctan2计算两个向量的角度，然后相减得到夹角
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            # 确保角度在0到180度之间
            if angle > 180.0:
                angle = 360 - angle
                
            return angle

    def _analyze_chest_pull_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder, wrist = landmarks['right_shoulder'], landmarks['right_wrist']
        
        y_diff = wrist[1] - shoulder[1]

        self.state._overextension_detected = False
        self.state._overextension_type = None

        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'
            if self.state._action_active:
                self.state._action_overextended = True

        is_pulled_down = y_diff > params['end_threshold_y']
        self.state.stage = 'pulled' if is_pulled_down else 'start'

        if not self.state._action_active:
            if y_diff < params['start_threshold_y']:
                self.state._action_active = True
                self.state._start_position = wrist
                self.state._action_overextended = False
        elif is_pulled_down:
            distance = np.linalg.norm(wrist - self.state._start_position)
            min_dist = params.get('min_distance', 0.01)
            if distance > min_dist:
                self.state._action_active = False
                self.state._end_position = wrist
                self.state.count += 1
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

                self._on_rep_completed()

                is_non_standard = bool(getattr(self.state, '_action_overextended', False))
                self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'

                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2

                self.state._action_overextended = False

    
    def _analyze_lateral_raise_logic(self, landmarks):
        """
        修正后的侧平举分析逻辑。
        使用“起始 -> 顶点 -> 起始”的完整周期检测，并采用更合理的阈值。
        """
        params = self.config['params'][self.style]
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']

        # 使用水平距离的绝对值，这样左右手都能适用
        x_abs_diff = abs(wrist[0] - shoulder[0])
        # 垂直差用于判断过伸（y坐标越小，位置越高）
        y_diff = wrist[1] - shoulder[1]

        # 1. 过伸检测 (Over-extension)
        # 这个逻辑保持不变，因为它工作正常
        self.state._overextension_detected = False
        self.state._overextension_type = None
        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'
            # 如果在一次有效动作期间发生了过伸，记录下来
            if self.state._action_active:
                self.state._action_overextended = True

        # 2. 阶段判断 (Stage determination)
        # 根据手臂是否打开到顶点阈值来判断当前阶段
        is_up = x_abs_diff > params['end_threshold_x']
        is_down = x_abs_diff < params['start_threshold_x']
        
        if is_up:
            self.state.stage = 'up'
        elif is_down:
            self.state.stage = 'down'
        # 在中间过程，stage保持不变

        # 3. 计数逻辑 (Counting Logic) - 完整的状态机
        # 状态: down -> (moving up) -> up -> (moving down) -> down (计数!)

        # 如果当前在 'down' 阶段，并且一个动作周期没有被激活
        # 这意味着我们准备好开始新的一次动作
        if is_down and not self.state._action_active:
            self.state._action_active = True
            self.state._in_up_phase = False  # 重置“到达顶点”标记
            self.state._action_overextended = False # 重置本次动作的过伸标记
            self.state._start_position = wrist

        # 如果动作已激活，并且我们首次达到了 'up' 阶段
        if self.state._action_active and not self.state._in_up_phase and is_up:
            self.state._in_up_phase = True
            # 可以在这里记录到达顶点的时间等信息

        # 如果动作已激活，并且已经到达过顶点(up)，现在又回到了起始(down)区域
        if self.state._action_active and self.state._in_up_phase and is_down:
            self.state._end_position = wrist
            distance = float(np.linalg.norm(self.state._end_position - self.state._start_position))
            
            # 只有当位移足够大时才计数，防止微小抖动被误判
            if distance > params.get('min_distance', 0.02):
                self.state.count += 1
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

                self._on_rep_completed()

                # 根据本次动作中是否发生过过伸来分类
                is_non_standard = bool(getattr(self.state, '_action_overextended', False))
                self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'

                # 触发完成动画
                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2

            # 一次完整的动作结束，重置状态以便下一次计数
            self.state._action_active = False
            self.state._in_up_phase = False
            self.state._action_overextended = False

    def _analyze_front_raise_logic(self, landmarks):
        """
        重构后的前平举分析逻辑。
        使用“放下 -> 举起 -> 放下” (down -> up -> down) 的完整周期检测。
        """
        params = self.config['params'][self.style]
        shoulder, wrist = landmarks['right_shoulder'], landmarks['right_wrist']
        
        # y_diff: 手腕相对于肩膀的垂直距离。y坐标向下为正，所以手臂越低，y_diff越大。
        y_diff = wrist[1] - shoulder[1]

        # 1. 过伸检测 (Over-extension)
        # 如果手臂举得过高（y_diff 小于过伸阈值），则标记。
        self.state._overextension_detected = False
        self.state._overextension_type = None
        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'
            # 如果在一次有效动作期间发生了过伸，记录下来用于最终的动作质量评估。
            if self.state._action_active:
                self.state._action_overextended = True

        # 2. 阶段判断 (Stage determination)
        # is_up: 手臂是否处于高位（顶点区域）。
        # is_down: 手臂是否处于低位（起始/结束区域）。
        is_up = y_diff < params['end_threshold_y']
        is_down = y_diff > params['start_threshold_y']
        
        if is_up:
            self.state.stage = 'up'
        elif is_down:
            self.state.stage = 'down'
        # 在中间过程（从down到up或从up到down），stage保持不变，给用户一个稳定的状态反馈。

        # 3. 计数逻辑 (Counting Logic) - 完整的 down -> up -> down 状态机
        
        # 初始化内部状态变量（如果它们不存在）
        if not hasattr(self.state, '_in_up_phase'):
            self.state._in_up_phase = False

        # 如果当前手臂在 'down' 状态，并且没有激活的动作，则准备开始新的一次动作。
        if is_down and not self.state._action_active:
            self.state._action_active = True
            self.state._in_up_phase = False  # 重置“到达顶点”的标记
            self.state._action_overextended = False # 重置本次动作的过伸标记
            self.state._start_position = wrist # 记录起始位置用于计算位移和能量

        # 如果动作已激活，并且我们首次达到了 'up' 阶段（顶点）。
        if self.state._action_active and not self.state._in_up_phase and is_up:
            self.state._in_up_phase = True
            # 这里可以添加逻辑，例如记录到达顶点的时间，用于分析“顶峰收缩”。

        # 如果动作已激活，并且已经到达过顶点(up)，现在又回到了起始(down)区域。
        # 这是完成一次完整动作的信号！
        if self.state._action_active and self.state._in_up_phase and is_down:
            self.state._end_position = wrist
            distance = float(np.linalg.norm(self.state._end_position - self.state._start_position))
            
            # 只有当位移足够大时才计数，防止微小的手臂抖动被误判为一次动作。
            if distance > params.get('min_distance', 0.015):
                self.state.count += 1
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

                self._on_rep_completed()

                # 根据本次动作中是否发生过过伸来分类。
                is_non_standard = bool(getattr(self.state, '_action_overextended', False))
                self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'

                # 触发完成动画效果（在UI上显示几帧）。
                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2

            # 一次完整的动作结束，重置所有状态，为下一次动作做准备。
            self.state._action_active = False
            self.state._in_up_phase = False
            self.state._action_overextended = False

    def _analyze_overhead_press_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder, wrist, elbow = landmarks['right_shoulder'], landmarks['right_wrist'], landmarks['right_elbow']
        
        y_diff = wrist[1] - shoulder[1]
        
        is_up = y_diff < params['end_threshold_y']
        # 起始位置判断：手腕在肩膀附近的一个小范围内
        is_at_start_pos = abs(y_diff) < params['start_threshold_y']
        self.state.stage = 'up' if is_up else 'down'

        if not self.state._action_active:
            if is_at_start_pos:
                self.state._action_active = True
                self.state._start_position = wrist
        elif is_up:
            # 因为此函数不检测 overextension，所以 self.state._overextension_detected 永远是 False
            # 但保留这个判断以维持代码结构统一性
            if not self.state._overextension_detected:
                self.state._action_active = False
                self.state._end_position = wrist
                self.state.count += 1
                distance = np.linalg.norm(self.state._end_position - self.state._start_position)
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

                self._on_rep_completed()

    def _analyze_squat_logic(self, landmarks):
        params = self.config['params'][self.style]
        
        hip = landmarks['right_hip']
        knee = landmarks['right_knee']
        ankle = landmarks['right_ankle']

        # 计算膝盖角度
        knee_angle = self._calculate_angle(hip, knee, ankle)

        # 根据膝盖角度判断是处于站立 ('up') 还是深蹲 ('down') 状态
        is_down = knee_angle < params['down_threshold_angle']
        is_up = knee_angle > params['up_threshold_angle']
        
        if is_down:
            self.state.stage = 'down'
        elif is_up:
            self.state.stage = 'up'

        # 核心计数逻辑：从 'down' 状态回到 'up' 状态完成一次计数
        if self.state.stage == 'down':
            # 如果当前处于深蹲状态，则激活下一次计数的准备状态
            self.state._action_active = True
            self.state._start_position = hip # 使用臀部位置来估算能量消耗
        
        if self.state._action_active and self.state.stage == 'up':
            # 如果之前已经激活，并且现在回到了站立状态，则完成一次动作
            self.state._action_active = False
            self.state.count += 1
            self.state._end_position = hip

            self._on_rep_completed()
            # 深蹲的能量消耗主要来自克服自身体重，这里用臀部垂直位移估算
            # 注意：这里的能量计算是一个非常简化的模型
            if self.state._start_position is not None:
                distance = abs(self.state._end_position[1] - self.state._start_position[1])
                # 假设一个70kg的人，其做功的体重部分约为60%
                body_weight_force = 70 * 0.6 * 9.81
                self.state.total_energy += body_weight_force * distance

    def _generate_feedback(self):
        if self.state._overextension_detected:
            if self.state._overextension_type == 'height':
                self.state.feedback = "手臂舉得太高了，請放低一些！"
            elif self.state._overextension_type == 'depth':
                self.state.feedback = "手臂太靠後了，請往前一些！"
            else:
                self.state.feedback = "動作幅度過大，請小心一點！"
            return
        
        # New exercise specific feedback
        if self.exercise_id == 'squat':
            if self.state.stage == 'down':
                self.state.feedback = "很好，保持核心收緊！" if self.style == 'intermediate' else "蹲下去！你能行！"
            elif self.state.stage == 'up':
                self.state.feedback = "準備下蹲" if self.style == 'intermediate' else "再來一個！"
            return
        
        if self.style == 'intermediate':
            if self.state.stage == 'up':
                self.state.feedback = "保持頂峰收縮，然後緩慢下放"
            elif self.state.stage == 'down':
                self.state.feedback = "很好，準備下次"
            else:
                self.state.feedback = "請開始動作"
        
        elif self.style == 'beginner':
            if self.state.stage == 'up':
                self.state.feedback = "漂亮！你太棒了！"
            else:
                self.state.feedback = "加油！再來一個！"