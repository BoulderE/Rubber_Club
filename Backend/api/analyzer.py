import mediapipe as mp
import numpy as np

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
                'start_threshold_x': 0.05,
                'end_threshold_x': 0.09,
                'over_extension_threshold_y': -0.2,
                'min_distance': 0.02
            },
            'beginner': {
                'start_threshold_x': 0.07,
                'end_threshold_x': 0.05,
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
                'start_threshold_y': 0.20,
                'end_threshold_y': 0.25,
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
        # New
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

        # 重置当帧的过伸标记
        self.state._overextension_detected = False
        self.state._overextension_type = None

        # completed 使用“延迟清理”机制
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
                'category': self.state._last_completion_category if self.state._completed_this_frame else ('non_standard' if self.state._overextension_detected else 'standard')
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
            'category': self.state._last_completion_category if self.state._completed_this_frame else ('non_standard' if self.state._overextension_detected else 'standard')
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

    # def _analyze_chest_pull_logic(self, landmarks):
    #     params = self.config['params'][self.style]
    #     shoulder, wrist = landmarks['right_shoulder'], landmarks['right_wrist']
        
    #     y_diff = wrist[1] - shoulder[1]
        
    #     if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
    #         self.state._overextension_detected = True
    #         self.state._overextension_type = 'height'

    #     # 阶段判断: 手臂在下方为 'pulled'，在上方为 'start'
    #     is_pulled_down = y_diff > params['end_threshold_y']
    #     self.state.stage = 'pulled' if is_pulled_down else 'start'

    #     if not self.state._action_active:
    #         if y_diff < params['start_threshold_y']:
    #             self.state._action_active = True
    #             self.state._start_position = wrist
    #     elif is_pulled_down:
    #         # 允许无论是否过伸都进入完成分支，用 category 标注标准/非标准
    #         distance = np.linalg.norm(wrist - self.state._start_position)
    #         min_dist = params.get('min_distance', 0.01)
    #         if distance > min_dist:
    #             self.state._action_active = False
    #             self.state._end_position = wrist
    #             self.state.count += 1
    #             self.state.total_distance += distance
    #             self.state.total_energy += self.state.BAND_RESISTANCE_N * distance
    #             self.state._last_completion_category = 'non_standard' if self.state._overextension_detected else 'standard'
    #             self.state._completed_this_frame = True
    #             self.state._completed_hold_frames = 2

    def _analyze_chest_pull_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder, wrist = landmarks['right_shoulder'], landmarks['right_wrist']
        
        y_diff = wrist[1] - shoulder[1]

        # 每帧初始化本帧的过伸显示（仅用于即时反馈）
        self.state._overextension_detected = False
        self.state._overextension_type = None

        # 仅保留上相位过伸：手腕明显高于肩（举得过高）
        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'
            # 在动作进行期间，一旦出现过伸，就把本次动作标记为非标准
            if self.state._action_active:
                self.state._action_overextended = True

        # 阶段判断: 手臂在下方为 'pulled'，在上方为 'start'
        is_pulled_down = y_diff > params['end_threshold_y']
        self.state.stage = 'pulled' if is_pulled_down else 'start'

        if not self.state._action_active:
            # 开启新动作前，清除本次动作的过伸累积标记
            if y_diff < params['start_threshold_y']:
                self.state._action_active = True
                self.state._start_position = wrist
                self.state._action_overextended = False
        elif is_pulled_down:
            # 完成：无论是否过伸都计数，category 由“本次动作期间是否出现过伸”决定
            distance = np.linalg.norm(wrist - self.state._start_position)
            min_dist = params.get('min_distance', 0.01)
            if distance > min_dist:
                self.state._action_active = False
                self.state._end_position = wrist
                self.state.count += 1
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

                # 用动作级标志决定这一“完成”的类别
                is_non_standard = bool(getattr(self.state, '_action_overextended', False))
                self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'

                # 触发完成帧，并保持若干帧
                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2

                # 重置动作级过伸标记（为下一次动作做准备）
                self.state._action_overextended = False

    def _analyze_front_raise_logic(self, landmarks):
        """前平举的特定分析逻辑"""
        params = self.config['params'][self.style]
        
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']

        y_diff = wrist[1] - shoulder[1]

        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height'

        # 当手臂举到与肩同高或更高时为 'up'
        is_up = y_diff < params['end_threshold_y']
        self.state.stage = 'up' if is_up else 'down'

        if not self.state._action_active:
            # 当手臂处于较低位置时，准备开始一个动作
            if y_diff > params['start_threshold_y']:
                self.state._action_active = True
                self.state._start_position = wrist
        
        elif is_up:
            # 允许无论是否过伸都进入完成分支，用 category 标注标准/非标准
            self.state._action_active = False
            self.state._end_position = wrist
            self.state.count += 1
            distance = np.linalg.norm(self.state._end_position - self.state._start_position)
            self.state.total_distance += distance
            self.state.total_energy += self.state.BAND_RESISTANCE_N * distance
            self.state._last_completion_category = 'non_standard' if self.state._overextension_detected else 'standard'
            self.state._completed_this_frame = True
            self.state._completed_hold_frames = 2

    def _analyze_overhead_press_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder, wrist, elbow = landmarks['right_shoulder'], landmarks['right_wrist'], landmarks['right_elbow']
        
        y_diff = wrist[1] - shoulder[1]
       
        # 【已移除】不再检测 z_diff 和 over_extension
        
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