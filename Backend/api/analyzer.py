import mediapipe as mp
import numpy as np

EXERCISE_CONFIG = {
    'chest_pull': {
        'name': '胸部拉伸',
        'landmarks_to_use': ['right_shoulder', 'right_wrist'],
        'logic_function': '_analyze_chest_pull_logic',
        'params': {
            'guide': {  # 严格模式
                'start_threshold_y': -0.02, # 要求起始位置更低
                'end_threshold_y': 0.02,   # 要求结束位置更标准
                'over_extension_threshold_y': -0.2,
                'min_distance': 0.02
            },
            'motivator': {  # 鼓励模式
                'start_threshold_y': -0.015, # 起始位置要求较宽松
                'end_threshold_y': 0.03,   # 结束位置要求较宽松
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
            'guide': {  # 严格模式
                'start_threshold_x': 0.05, # 要求手臂初始更贴近身体
                'end_threshold_x': 0.20, # 要求手臂举得更高、更开
                'over_extension_threshold_y': -0.2,
                'min_distance': 0.02
            },
            'motivator': {  # 鼓励模式
                'start_threshold_x': 0.07,  # 初始位置要求较宽松
                'end_threshold_x': 0.15, # 举到大概位置就算完成
                'over_extension_threshold_y': -0.18,
                'min_distance': 0.015
            }
        }
    },
    'front_raise': {
        'name': '前平举',
        'landmarks_to_use': ['right_shoulder', 'right_wrist', 'right_elbow'],
        'logic_function': '_analyze_front_raise_logic',
        'params': {
            'guide': {
                'start_threshold_y': 0.15,  # 手臂放下时，手腕低于肩膀的y坐标差
                'end_threshold_y': 0.05,   # 手臂举到与肩同高时，y坐标差阈值
                'over_extension_threshold_y': -0.05, # 手臂举过高
                'min_distance': 0.02
            },
            'motivator': {
                'start_threshold_y': 0.18,
                'end_threshold_y': 0.08,
                'over_extension_threshold_y': -0.03,
                'min_distance': 0.015
            }
        }
    },
    'overhead_press': {
        'name': '过顶举',
        'landmarks_to_use': ['right_shoulder', 'right_wrist', 'right_elbow'],
        'logic_function': '_analyze_overhead_press_logic',
        'params': {
            'guide': {
                'start_threshold_y': 0.05, # 起始时，手腕在肩膀附近的高度差
                'end_threshold_y': -0.2,   # 举到最高点时，手腕高于肩膀的y坐标差
                'min_distance': 0.02
            },
            'motivator': {
                'start_threshold_y': 0.08,
                'end_threshold_y': -0.18,
                'min_distance': 0.015
            }
        }
    },
    'squat': {
        'name': '深蹲',
        'landmarks_to_use': ['right_hip', 'right_knee', 'right_ankle'],
        'logic_function': '_analyze_squat_logic',
        'params': {
            'guide': {
                'up_threshold_angle': 165.0, # 站立时膝盖角度
                'down_threshold_angle': 95.0 # 深蹲时膝盖角度
            },
            'motivator': {
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
        self._action_active = False
        self.total_distance = 0.0
        self.total_energy = 0.0
        self.BAND_RESISTANCE_N = 25 * 9.81

class ExerciseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.state = WorkoutState()
        self.config = None
        self.style = 'guide'
        self.exercise_id = None

    def setup(self, exercise_type: str, style: str):
        if exercise_type not in EXERCISE_CONFIG:
            raise ValueError(f"不支持的运动类型: {exercise_type}")
        
        self.config = EXERCISE_CONFIG[exercise_type]
        self.style = style if style in ['guide', 'motivator'] else 'guide'
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

        if self.state.is_paused:
            self.state.stage = self.state.stage 
            self.state.feedback = "已暂停，做点赞手势继续"
            return {
                'count': self.state.count,
                'stage': self.state.stage,
                'feedback': self.state.feedback,
                'paused': True,
                'energy': self.state.total_energy
            }

        if landmarks:
            logic_function = getattr(self, self.config['logic_function'])
            logic_function(landmarks)
        else:
            self.state.feedback = "请确保身体关键部位在镜头内"

        self._generate_feedback()

        # self.config['name'] 用于前端显示
        return {
            'count': self.state.count,
            'stage': self.state.stage,
            'feedback': self.state.feedback,
            'paused': self.state.is_paused,
            'energy': self.state.total_energy
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
        
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']
        
        y_diff = wrist[1] - shoulder[1]
        self.state.stage = 'up' if y_diff < params['start_threshold_y'] else 'down'

        if not self.state._action_active:
            if y_diff < params['start_threshold_y']:
                self.state._action_active = True
                self.state._start_position = wrist
        else:
            if y_diff > params['end_threshold_y']:
                self.state._action_active = False
                self.state._end_position = wrist
                self.state.count += 1
                distance = np.linalg.norm(self.state._end_position - self.state._start_position)
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

    def _analyze_lateral_raise_logic(self, landmarks):
        params = self.config['params'][self.style]

        shoulder = landmarks['right_shoulder']
        elbow = landmarks['right_elbow']
        wrist = landmarks['right_wrist']

        self.state._overextension_detected = (elbow[1] < shoulder[1] + params['over_extension_threshold_y'] or 
                                              wrist[1] < shoulder[1] + params['over_extension_threshold_y'])

        x_diff = abs(wrist[0] - shoulder[0])
        self.state.stage = 'up' if x_diff > params['end_threshold_x'] else 'down'

        if not self.state._action_active:
            # 使用从 params 加载的动态阈值
            if x_diff < params['start_threshold_x']:
                self.state._action_active = True
                self.state._start_position = wrist
        else:
            # 使用从 params 加载的动态阈值
            if x_diff > params['end_threshold_x']:
                if not self.state._overextension_detected:
                    self.state._action_active = False
                    self.state._end_position = wrist
                    self.state.count += 1
                    distance = np.linalg.norm(self.state._end_position - self.state._start_position)
                    self.state.total_distance += distance
                    self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

    # New exercises
    def _analyze_front_raise_logic(self, landmarks):
        """前平举的特定分析逻辑"""
        params = self.config['params'][self.style]
        
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']

        y_diff = wrist[1] - shoulder[1]

        # 检测是否举得过高
        self.state._overextension_detected = (y_diff < params['over_extension_threshold_y'])

        # 当手臂举到与肩同高或更高时为 'up'
        is_up = y_diff < params['end_threshold_y']
        self.state.stage = 'up' if is_up else 'down'

        if not self.state._action_active:
            # 当手臂处于较低位置时，准备开始一个动作
            if y_diff > params['start_threshold_y']:
                self.state._action_active = True
                self.state._start_position = wrist
        else:
            # 当手臂从准备状态举到目标高度时，计数一次
            if is_up:
                if not self.state._overextension_detected:
                    self.state._action_active = False
                    self.state._end_position = wrist
                    self.state.count += 1
                    distance = np.linalg.norm(self.state._end_position - self.state._start_position)
                    self.state.total_distance += distance
                    self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

    def _analyze_overhead_press_logic(self, landmarks):
        """过顶举的特定分析逻辑"""
        params = self.config['params'][self.style]
        
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']
        
        y_diff = wrist[1] - shoulder[1]

        # 当手臂举过头顶时为 'up'
        is_up = y_diff < params['end_threshold_y']
        # 当手臂在肩膀高度时为 'down' (起始位置)
        is_at_start_pos = abs(y_diff) < params['start_threshold_y']
        
        self.state.stage = 'up' if is_up else 'down'

        if not self.state._action_active:
            # 当手臂在肩膀高度的起始位置时，准备开始一个动作
            if is_at_start_pos:
                self.state._action_active = True
                self.state._start_position = wrist
        else:
            # 当手臂从起始位置推举到最高点时，计数一次
            if is_up:
                self.state._action_active = False
                self.state._end_position = wrist
                self.state.count += 1
                distance = np.linalg.norm(self.state._end_position - self.state._start_position)
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

    def _analyze_squat_logic(self, landmarks):
        """深蹲的特定分析逻辑"""
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
        """根据 style 和当前状态生成反馈信息"""
        if self.state._overextension_detected:
            self.state.feedback = "动作过高，请放低一些！"
            return
        
        # New exercise specific feedback
        if self.exercise_id == 'squat':
            if self.state.stage == 'down':
                self.state.feedback = "很好，保持核心收紧！" if self.style == 'guide' else "蹲下去！你能行！"
            elif self.state.stage == 'up':
                self.state.feedback = "准备下蹲" if self.style == 'guide' else "漂亮！再来一个！"
            return
        
        if self.style == 'guide':
            if self.state.stage == 'up':
                self.state.feedback = "保持顶峰收缩，然后缓慢下放"
            elif self.state.stage == 'down':
                self.state.feedback = "很好，准备下一次"
            else:
                self.state.feedback = "请开始动作"
        
        elif self.style == 'motivator':
            if self.state.stage == 'up':
                self.state.feedback = "漂亮！你太棒了！"
            else:
                self.state.feedback = "加油！再来一个！"