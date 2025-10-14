import mediapipe as mp
import numpy as np
import time, math
from statistics import mean, pstdev

EXERCISE_CONFIG = {
    'bicep_curl': {
        'name': '胸部拉伸',
        'landmarks_to_use': ['right_shoulder', 'right_wrist'],
        'logic_function': '_analyze_bicep_curl_logic',
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
    'chest_pull': {
        'name': '胸前拉開',
        'landmarks_to_use': [
            'left_shoulder','right_shoulder',
            'left_wrist','right_wrist'
        ],
        'logic_function': '_analyze_chest_pull_logic',
        'params': {
            'intermediate': {
                'start_threshold_wx': 0.26,
                'end_threshold_wx':   0.40,
                'min_distance_wx':    0.025,
                'min_wrist_rel_y':   -0.28,
                'max_wrist_rel_y':    0.38
            },
            'beginner': {
                'start_threshold_wx': 0.28,
                'end_threshold_wx':   0.35,
                'min_distance_wx':    0.010,
                'min_wrist_rel_y':   -0.28,
                'max_wrist_rel_y':    0.38
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
    },

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

    def _analyze_bicep_curl_logic(self, landmarks):
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

    def _analyze_chest_pull_logic(self, landmarks):

    # Chest Pull（胸前拉開）
    # - 主判據：wx = |wr.x - wl.x|（左右手腕水平距離）; up 時變大、down 時變小
    # - 輔助：手腕垂直穩定、手腕高度範圍、腕-肩距離合計增加、肘角增大（至少一側）
    # - 狀態機：down -> up(達成且hold) -> 回到down 計數
    # 假設: 坐標為[0..1]正規化，y向下為正。
        P = self.config['params'][self.style]

        ls = np.array(landmarks['left_shoulder'], dtype=float)
        rs = np.array(landmarks['right_shoulder'], dtype=float)
        lw = np.array(landmarks['left_wrist'], dtype=float)
        rw = np.array(landmarks['right_wrist'], dtype=float)

        # 主指標：左右手腕水平距離
        wx = abs(float(rw[0] - lw[0]))

        # 手腕相對同側肩的高度（y向下為正）
        rel_y_l = float(lw[1] - ls[1])
        rel_y_r = float(rw[1] - rs[1])

        # 滯回分區
        is_up = wx >= P['end_threshold_wx']
        is_down = wx <= P['start_threshold_wx']

        if is_up:
            self.state.stage = 'up'
        elif is_down:
            self.state.stage = 'down'

        # 初始化狀態
        if not hasattr(self.state, '_cpull_active'):
            self.state._cpull_active = False
        if not hasattr(self.state, '_cpull_reached_up'):
            self.state._cpull_reached_up = False

        # 在 down 區且未激活 → 設置起點
        if is_down and not self.state._cpull_active:
            self.state._cpull_active = True
            self.state._cpull_reached_up = False
            self.state._cpull_start_wx = wx
            self.state._cpull_start_lw = lw[:2].copy()
            self.state._cpull_start_rw = rw[:2].copy()

        # 首次進入 up 區
        if self.state._cpull_active and not self.state._cpull_reached_up and is_up:
            self.state._cpull_reached_up = True

        # 回到 down 區，嘗試完成一次
        if self.state._cpull_active and self.state._cpull_reached_up and is_down:
            # 幅度要求
            wx_range = abs(wx - self.state._cpull_start_wx)
            pass_range = wx_range >= P['min_distance_wx']

            # 輕量姿態（末端檢查，避免明顯上舉/下壓）
            pass_rel_y = (P['min_wrist_rel_y'] <= rel_y_l <= P['max_wrist_rel_y']) and \
                        (P['min_wrist_rel_y'] <= rel_y_r <= P['max_wrist_rel_y'])

            if pass_range and pass_rel_y:
                move_dist = float(
                    np.linalg.norm(lw[:2] - self.state._cpull_start_lw) +
                    np.linalg.norm(rw[:2] - self.state._cpull_start_rw)
                )
                self.state.count += 1
                self.state.total_distance += move_dist
                self.state.total_energy += self.state.BAND_RESISTANCE_N * move_dist

                self._on_rep_completed()
                self.state._last_completion_category = 'standard'
                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2

            # 復位（結束本輪）
            for k in ['_cpull_active','_cpull_reached_up',
                    '_cpull_start_wx','_cpull_start_lw','_cpull_start_rw']:
                if hasattr(self.state, k):
                    delattr(self.state, k)
            self.state._cpull_active = False
            self.state._cpull_reached_up = False           
    
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

    def _analyze_chest_press_logic(self, landmarks):
    
    # 胸推（Chest Press）
    # 流程：down（屈肘、手在胸前） -> up（前推、肘伸直） -> 回到 down 才计数。
    # 主判据：肘角 angle = ∠(shoulder, elbow, wrist)，角度越大越接近“伸直”。
        P = self.config['params'][self.style]

        # 取点
        sh = np.array(landmarks['right_shoulder'], dtype=float)
        el = np.array(landmarks['right_elbow'], dtype=float)
        wr = np.array(landmarks['right_wrist'], dtype=float)

        # 标量与几何量
        dx = abs(wr[0] - el[0])  # 主指标
        d_ws = float(np.linalg.norm(wr[:2] - sh[:2]))  # 腕-肩平面距离
        y_diff = wr[1] - sh[1]  # 过伸判定用（y小为更高）

        # 帧率与时间
        fps = max(1.0, float(getattr(self, 'fps', 30.0)))
        dt = 1.0 / fps

        # 过伸检测
        self.state._overextension_detected = False
        self.state._overextension_type = None
        if y_diff < P['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'
            if self.state._action_active:
                self.state._action_overextended = True

        # 初始化缓存
        if not hasattr(self.state, '_cp_in_up'):
            self.state._cp_in_up = False
        if not hasattr(self.state, '_cp_active'):
            self.state._cp_active = False
        if not hasattr(self.state, '_cp_hist'):
            self.state._cp_hist = []  # 每帧缓存：wr, el, sh, dx, d_ws

        # 更新历史（限制长度以覆盖最长时间窗）
        max_hist_len = int(fps * max(P['max_up_time_s'] + P['min_down_time_s'] + 0.8, 4.0))
        self.state._cp_hist.append({
            'wr': wr.copy(), 'el': el.copy(), 'sh': sh.copy(),
            'dx': dx, 'dws': d_ws
        })
        if len(self.state._cp_hist) > max_hist_len:
            self.state._cp_hist.pop(0)

        # 滞回区判定
        is_up_zone = dx <= P['end_threshold_dx']
        is_down_zone = dx >= P['start_threshold_dx']
        if is_up_zone:
            self.state.stage = 'up'
        elif is_down_zone:
            self.state.stage = 'down'

        # 启动一轮（进入down区且未激活）
        if is_down_zone and not self.state._cp_active:
            self.state._cp_active = True
            self.state._cp_in_up = False
            self.state._action_overextended = False
            self.state._cp_start_idx = len(self.state._cp_hist) - 1
            self.state._cp_start_dx = dx
            self.state._cp_start_dws = d_ws
            self.state._cp_start_wr = wr.copy()
            self.state._cp_start_el = el.copy()
            self.state._cp_start_sh = sh.copy()
            self.state._cp_up_idx = None
            self.state._cp_hold_frames = 0

        # 达到up顶点（首次进入up区）
        if self.state._cp_active and not self.state._cp_in_up and is_up_zone:
            self.state._cp_in_up = True
            self.state._cp_up_idx = len(self.state._cp_hist) - 1
            self.state._cp_hold_frames = 0

        # 在up区内累计hold时间
        if self.state._cp_active and self.state._cp_in_up and is_up_zone:
            self.state._cp_hold_frames += 1

        # 从 up 回到 down，尝试完成一次
        if self.state._cp_active and self.state._cp_in_up and is_down_zone:
            start_idx = self.state._cp_start_idx
            up_idx = self.state._cp_up_idx if self.state._cp_up_idx is not None else len(self.state._cp_hist) - 1
            end_idx = len(self.state._cp_hist) - 1

            # 若无有效up阶段，直接复位
            if up_idx is None or up_idx <= start_idx:
                # 复位
                self.state._cp_active = False
                self.state._cp_in_up = False
                return

            # 时间窗评估
            up_frames = max(1, up_idx - start_idx)
            down_frames = max(1, end_idx - up_idx)
            up_time = up_frames * dt
            down_time = down_frames * dt
            hold_time = self.state._cp_hold_frames * dt

            if not (P['min_up_time_s'] <= up_time <= P['max_up_time_s']):
                # 失败：推出过快或过慢
                pass_up_time = False
            else:
                pass_up_time = True

            pass_down_time = down_time >= P['min_down_time_s']
            pass_hold = hold_time >= P['hold_up_time_s']

            # 轨迹一致性与速度
            seq = self.state._cp_hist[start_idx:up_idx + 1]
            wr_seq = np.array([it['wr'] for it in seq])
            el_seq = np.array([it['el'] for it in seq])
            sh_seq = np.array([it['sh'] for it in seq])

            # 位移分解（以x为前向）
            disp = wr_seq[-1, :2] - wr_seq[0, :2]
            forward_disp = abs(disp[0])
            total_disp = float(np.linalg.norm(disp))
            forward_rate = (forward_disp / (total_disp + 1e-8))

            # 平均前向速度
            avg_forward_speed = forward_disp / (up_time + 1e-8)

            # 垂直偏移限制
            vertical_exc = abs(wr_seq[-1, 1] - wr_seq[0, 1])

            # 腕-肩距离增量
            d_inc = float(seq[-1]['dws'] - seq[0]['dws'])

            # 肘角变化（增强真实性）
            def angle(a, b, c):
                v1 = a - b; v2 = c - b
                n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8: return 0.0
                cosv = float(np.dot(v1, v2) / (n1 * n2))
                cosv = max(-1.0, min(1.0, cosv))
                return math.degrees(math.acos(cosv))
            elbow_angle_gain = angle(sh_seq[-1, :2], el_seq[-1, :2], wr_seq[-1, :2]) - \
                            angle(sh_seq[0, :2],  el_seq[0, :2],  wr_seq[0, :2])

            pass_forward_rate = forward_rate >= P['min_forward_dx_rate']
            pass_forward_speed = avg_forward_speed >= P['min_forward_speed']
            pass_vertical = vertical_exc <= P['max_vertical_excursion']
            pass_d_inc = d_inc >= P['min_wrist_shoulder_d_inc']
            pass_elbow_gain = elbow_angle_gain >= P['min_elbow_angle_gain']

            # 姿态稳定（肩不应剧烈移动）
            sh_disp = sh_seq[-1, :2] - sh_seq[0, :2]
            pass_shoulder = (abs(sh_disp[1]) <= P['max_shoulder_y_change']) and \
                            (abs(sh_disp[0]) <= P['max_shoulder_x_drift'])

            # Δx 幅度
            dx_start = float(self.state._cp_start_dx)
            dx_min = min([it['dx'] for it in seq]) if len(seq) else dx
            dx_range = dx_start - dx_min
            pass_dx_range = dx_range >= P['min_distance_dx']

            # 多条件合一：必须满足主链 + 至少若干辅条件
            core_ok = pass_dx_range and pass_up_time and pass_down_time and pass_hold
            aux_checks = [pass_forward_rate, pass_forward_speed, pass_vertical, pass_d_inc, pass_elbow_gain, pass_shoulder]
            aux_pass_count = sum(1 for x in aux_checks if x)

            # 要求至少通过 4 项辅助，基本杜绝“抬手/耸肩/摆动”作弊
            if core_ok and aux_pass_count >= 4:
                # 计数
                move_dist = float(np.linalg.norm(self.state._cp_hist[end_idx]['wr'][:2] - self.state._cp_hist[start_idx]['wr'][:2]))
                self.state.count += 1
                self.state.total_distance += move_dist
                self.state.total_energy += self.state.BAND_RESISTANCE_N * move_dist

                self._on_rep_completed()

                is_non_standard = bool(getattr(self.state, '_action_overextended', False))
                self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'
                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2

            # 复位
            self.state._cp_active = False
            self.state._cp_in_up = False
            self.state._action_overextended = False
            for k in ['_cp_start_idx','_cp_up_idx','_cp_hold_frames','_cp_start_dx','_cp_start_dws',
                    '_cp_start_wr','_cp_start_el','_cp_start_sh']:
                if hasattr(self.state, k):
                    delattr(self.state, k)

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