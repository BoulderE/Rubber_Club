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
            self.state.stage = self.state.stage  # 保持上一次阶段或根据姿态轻量更新
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

        return {
            'count': self.state.count,
            'stage': self.state.stage,
            'feedback': self.state.feedback,
            'paused': self.state.is_paused,
            'energy': self.state.total_energy
        }

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
        """侧平举的特定分析逻辑，现在使用分层参数"""
        # 【关键】根据 self.style 动态加载对应的参数集
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

    def _generate_feedback(self):
        """根据 style 和当前状态生成反馈信息"""
        if self.state._overextension_detected:
            self.state.feedback = "动作过高，请放低一些！"
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