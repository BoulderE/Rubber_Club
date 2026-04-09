import mediapipe as mp
import numpy as np
import time, math
from statistics import mean, pstdev
from .quality_checker import QualityChecker
import os
from .lstm_scorer import LSTMScorer

def load_exercise_config():
    try:
        from models.db_models import get_session, ExerciseRule
        
        session = get_session()
        rules = session.query(ExerciseRule).all()
        session.close()
        
        config = {}
        for rule in rules:
            config[rule.exercise_key] = {
                'name': rule.name,
                'landmarks_to_use': rule.landmarks_to_use,
                'logic_function': rule.logic_function,
                'params': rule.params
            }
        
        print(f"從數據庫載入了 {len(config)} 個運動規則")
        return config
    
    except Exception as e:
        print(f"無法從數據庫載入，使用預設配置: {e}")
        return get_default_config()

def get_default_config():
    return {
    'bicep_curl': {
        'name': '胸部拉伸',
        'landmarks_to_use': ['right_shoulder', 'right_wrist'],
        'logic_function': '_analyze_bicep_curl_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': -0.015,
                'end_threshold_y': 0.03,
                'over_extension_threshold_y': -0.18,
                'min_distance': 0.015
            },
            'beginner': {
                'start_threshold_y': -0.010,
                'end_threshold_y': 0.04,
                'over_extension_threshold_y': -0.16,
                'min_distance': 0.010
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
                'start_threshold_wx': 0.28,
                'end_threshold_wx':   0.30,
                'min_distance_wx':    0.01,
                'min_wrist_rel_y':   -0.28,
                'max_wrist_rel_y':    0.38
            },
            'beginner': {
                'start_threshold_wx': 0.26,
                'end_threshold_wx':   0.275,
                'min_distance_wx':    0.01,
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
                'end_threshold_x': 0.18,
                'over_extension_threshold_y': -0.18,
                'min_distance': 0.02
            },
            'beginner': {
                'start_threshold_x': 0.12,
                'end_threshold_x': 0.16,
                'over_extension_threshold_y': -0.16,
                'min_distance': 0.015
            }
        }
    },
    'front_raise': {
        'name': '前平举',
        'landmarks_to_use': ['right_shoulder', 'right_wrist', 'right_elbow', 'right_hip'],
        'logic_function': '_analyze_front_raise_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': 0.70,
                'end_threshold_y': 0.20,
                'min_distance': 0.015
            },
            'beginner': {
                'start_threshold_y': 0.75,
                'end_threshold_y': 0.30,
                'min_distance': 0.012
            }
        }
    },
    'overhead_press': {
        'name': '过顶举',
        'landmarks_to_use': ['right_shoulder', 'right_wrist', 'right_elbow'],
        'logic_function': '_analyze_overhead_press_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': 0.08,
                'end_threshold_y': -0.18,
                'min_distance': 0.01
            },
            'beginner': {
                'start_threshold_y': 0.10,
                'end_threshold_y': -0.15,
                'min_distance': 0.008
            }
        }
    },
    'diagonal_lift': {
        'name': '對角線動作',
        'landmarks_to_use': [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip'
        ],
        'logic_function': '_analyze_diagonal_lift_logic',
        'params': {
            'intermediate': {
                'start_threshold_y': 0.20,
                'end_threshold_y': 0.00,
                'min_horizontal_disp': 0.08,
                'min_vertical_disp':   0.10,
                'min_distance':        0.18,
                'min_diagonal_angle':  8,
                'max_diagonal_angle':  85,
            },
            'beginner': {
                'start_threshold_y': 0.22,
                'end_threshold_y': 0.05,
                'min_horizontal_disp': 0.08,
                'min_vertical_disp':   0.10,
                'min_distance':        0.18,
                'min_diagonal_angle':  8,
                'max_diagonal_angle':  85,
            }
        }
    },
}

EXERCISE_CONFIG = load_exercise_config()

class WorkoutState:
    def __init__(self):
        self.lstm_scorer = LSTMScorer(
            models_dir=os.path.join(os.path.dirname(__file__), '..', 'lstm_models'),
            score_every_n=5
        )
        self.reset()
        self.reset()

    def reset(self):
        self.count = 0
        self.stage = None
        self.feedback = "請做 👍 手勢開始運動"
        self.is_paused = True
        self._overextension_detected = False
        self._overextension_type = None
        self.total_distance = 0.0
        self.total_energy = 0.0
        self.BAND_RESISTANCE_N = 25 * 9.81
        self._last_completion_category = 'standard'
        self._completed_this_frame = False
        self._completed_hold_frames = 0
        
        self._exercise_states = {
            'bicep_curl':  {'movement_state': 'down', 'start_pos': None, 'last_count_time': 0},
            'chest_pull':  {'movement_state': 'down', 'start_pos': None, 'last_count_time': 0},
            'lateral':     {'movement_state': 'down', 'start_pos': None, 'last_count_time': 0},
            'front':       {'movement_state': 'down', 'start_pos': None, 'last_count_time': 0},
            'overhead':    {'movement_state': 'down', 'start_pos': None, 'last_count_time': 0},
            'squat':       {'movement_state': 'up',   'start_pos': None, 'last_count_time': 0},
        }

        self._diag_left_state = {
            'movement_state': 'down',
            'start_wrist_y': None,
            'start_wrist_x': None,
            'last_count_time': 0,
        }
        self._diag_right_state = {
            'movement_state': 'down',
            'start_wrist_y': None,
            'start_wrist_x': None,
            'last_count_time': 0,
        }
        self._diag_active_side = None

        self.up_durations = []      
        self.down_durations = []    
        self.current_phase = None   
        self.phase_start_time = None  
        self._is_first_phase_switch = True 

class ExerciseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0)
        self.state = WorkoutState()
        self.config = None
        self.style = 'intermediate'
        self.exercise_id = None
        
        self.last_rep_start = None
        self.repetition_durations = []  
        self.smoothness_score = 100

        self.phase_timings = []
        self._phase_name = None
        self._phase_start_time = None

        self.target_count = 15
        self.current_task_id = None
        
        # new
        self.quality_checker = QualityChecker()
        self._last_angles = None
        self._last_quality = {'is_bad': False, 'error_type': None, 'message': None, 'violations': []}

    def _now(self):
        return time.time()
    
    def _finalize_open_phase(self):
        if self.state.current_phase is not None and self.state.phase_start_time is not None:
            duration = self._now() - self.state.phase_start_time
            if 0.1 <= duration <= 10.0:
                if self.state.current_phase == 'up':
                    self.state.up_durations.append(duration)
                    if len(self.state.up_durations) > 50:
                        self.state.up_durations.pop(0)
                elif self.state.current_phase == 'down':
                    self.state.down_durations.append(duration)
                    if len(self.state.down_durations) > 50:
                        self.state.down_durations.pop(0)

        self.state.current_phase = None
        self.state.phase_start_time = None

    def _update_phase(self, is_up: bool, is_down: bool):
        now_t = self._now()
        target_phase = None
        if is_up:
            target_phase = 'up'
        elif is_down:
            target_phase = 'down'
        else:
            return
        
        if self.state.current_phase != target_phase:
            if self.state.current_phase is not None and self.state.phase_start_time is not None:
                duration = now_t - self.state.phase_start_time
                if self.state._is_first_phase_switch:
                    self.state._is_first_phase_switch = False 
                elif 0.1 <= duration <= 10.0:
                    if self.state.current_phase == 'up':
                        self.state.up_durations.append(duration)
                        if len(self.state.up_durations) > 50:
                            self.state.up_durations.pop(0)
                    elif self.state.current_phase == 'down':
                        self.state.down_durations.append(duration)
                        if len(self.state.down_durations) > 50:
                            self.state.down_durations.pop(0)
            
            self.state.current_phase = target_phase
            self.state.phase_start_time = now_t
    
    def _on_rep_completed(self):
        now_t = self._now()
        if self.last_rep_start is not None:
            duration = max(0.0, now_t - self.last_rep_start)
            if 0.2 <= duration <= 10.0:
                self.repetition_durations.append(duration)
        self.last_rep_start = now_t
        smoothness_cutoff = max(3, self.target_count - 1)
        if self.state.count <= smoothness_cutoff:
            self.smoothness_score = self._compute_smoothness()

    def _compute_smoothness(self) -> int:
        import statistics
        max_reps = max(3, min(self.target_count - 1, 30))
        
        raw_up = self.state.up_durations[-20:]
        raw_down = self.state.down_durations[-20:]

        def calculate_phase_score(durations):
            if len(durations) < 3: return None
            med = statistics.median(durations)
            if med < 0.1: return None
            valid_durations = [d for d in durations if d <= med * 2.5]
            if len(valid_durations) < 2: return None
            avg = statistics.mean(valid_durations)
            std = statistics.pstdev(valid_durations)
            cv = std / avg  
            score = 100 - (cv * 140)
            return max(10, min(100, int(score)))

        up_score = calculate_phase_score(raw_up)
        down_score = calculate_phase_score(raw_down)
        final_score = 100
        if up_score is not None and down_score is not None:
            final_score = (up_score + down_score) / 2
        elif up_score is not None:
            final_score = up_score
        elif down_score is not None:
            final_score = down_score
        return int(final_score)

    def get_metrics(self):
        return {
            "rep_count": self.state.count,
            "smoothness": self.smoothness_score,
            "rep_durations": self.repetition_durations[-20:], 
        }

    def setup(self, exercise_type: str, style: str, target_count: int = None, task_id: int = None):
        if exercise_type not in EXERCISE_CONFIG:
            raise ValueError(f"不支援的運動類型:{exercise_type}")
        
        self.config = EXERCISE_CONFIG[exercise_type]
        self.style = style if style in ['intermediate', 'beginner'] else 'intermediate'
        self.exercise_id = exercise_type
        self.current_task_id = task_id

        # new
        if target_count is not None:
            self.target_count = target_count
        else:
            self.target_count = 10 if self.style == 'beginner' else 15

        self.reset()
        print(f"分析器已設定為: 運動='{self.config['name']}', 模式='{self.style}'")

    def reset(self):
        self.state.reset()
        self.last_rep_start = None
        self.repetition_durations = []
        self.smoothness_score = 100
        self.phase_timings = []
        self._phase_name = None
        self._phase_start_time = None

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
            raise RuntimeError("分析器未設定。請先呼叫 setup() 方法。")

        image_rgb = image
        results = self.pose.process(image_rgb)
        landmarks = self._get_landmarks(results)
        
        self.state._overextension_detected = False
        self.state._overextension_type = None

        if self.state._completed_hold_frames > 0:
            self.state._completed_hold_frames -= 1
            self.state._completed_this_frame = True
        else:
            self.state._completed_this_frame = False
            self.state._last_completion_category = 'standard'

        if self.state.is_paused:
            self._finalize_open_phase()
            self.state.feedback = "已暫停，做按讚手勢繼續"
            return self._build_result(paused=True)

        if landmarks:
            # new
            if results.pose_landmarks and self.exercise_id:
                self._last_angles = self.quality_checker.compute_angles(results.pose_landmarks)
                self._last_quality = self.quality_checker.check(self.exercise_id, self._last_angles)
                if self._last_quality['is_bad']:
                    self.state._overextension_detected = True
                    self.state._overextension_type = self._last_quality['error_type']

            logic_function = getattr(self, self.config['logic_function'])
            logic_function(landmarks)
        else:
            self._finalize_open_phase()
            self.state.feedback = "請確保身體關鍵部位在鏡頭內"

        self._generate_feedback()
        return self._build_result(paused=False)

    def _build_result(self, paused):
        return {
            'count': self.state.count,
            'stage': self.state.stage,
            'feedback': self.state.feedback,
            'paused': paused,
            'energy': self.state.total_energy,
            'overextended': self.state._overextension_detected,
            'completed': self.state._completed_this_frame,
            'category': self.state._last_completion_category if self.state._completed_this_frame else ('non_standard' if self.state._overextension_detected else 'standard'),
            'smoothness': self.smoothness_score,
            'rep_durations': self.repetition_durations[-20:],
            'phase_timings': self.phase_timings[-20:],
            'up_durations': self.state.up_durations[-10:],
            'down_durations': self.state.down_durations[-10:],

            #new
            'quality_angles': self._last_angles,
            'quality_violations': self._last_quality.get('violations', []),
        }
    
    def _calculate_angle(self, a, b, c):
        a = np.array(a)  
        b = np.array(b)  
        c = np.array(c)  
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def _process_state_machine(self, state_dict, is_up, is_down, current_pos, min_dist=0.01, debounce_time=0.5, count_on_up=True):
        current_time = time.time()
        
        self._update_phase(is_up, is_down)
        
        if is_up:
            self.state.stage = 'up'
        elif is_down:
            self.state.stage = 'down'

        if state_dict['movement_state'] == 'down' and is_up:
            if current_time - state_dict['last_count_time'] > debounce_time:
                dist = 0.0
                if state_dict['start_pos'] is not None:
                    dist = np.linalg.norm(current_pos - state_dict['start_pos'])
                
                if dist >= min_dist or state_dict['start_pos'] is None:
                    if state_dict['start_pos'] is not None:
                        self.state.count += 1
                        self.state.total_distance += float(dist)
                        self.state.total_energy += self.state.BAND_RESISTANCE_N * float(dist)
                        
                        self._on_rep_completed()
                        
                        self.state._completed_this_frame = True
                        self.state._completed_hold_frames = 2
                        
                        is_non_standard = self.state._overextension_detected
                        self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'

                    state_dict['movement_state'] = 'up'
                    state_dict['last_count_time'] = current_time

        elif state_dict['movement_state'] == 'up' and is_down:
            state_dict['movement_state'] = 'down'
            state_dict['start_pos'] = current_pos.copy()

        if state_dict['movement_state'] == 'down' and state_dict['start_pos'] is None and is_down:
            state_dict['start_pos'] = current_pos.copy()

    def _analyze_bicep_curl_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']
        
        y_diff = wrist[1] - shoulder[1]
        
        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'

        is_up = y_diff < params['start_threshold_y']
        is_down = y_diff > params['end_threshold_y']

        self._process_state_machine(
            self.state._exercise_states['bicep_curl'],
            is_up, is_down, 
            current_pos=wrist,
            min_dist=params.get('min_distance', 0.015),
            debounce_time=0.3 
        )

    def _analyze_chest_pull_logic(self, landmarks):
        P = self.config['params'][self.style]
        lw = landmarks['left_wrist']
        rw = landmarks['right_wrist']
        
        wx = abs(float(rw[0] - lw[0]))

        is_up = wx >= P['end_threshold_wx']
        is_down = wx <= P['start_threshold_wx']

        current_pos_combined = np.concatenate([lw, rw])

        ls = landmarks['left_shoulder']
        rs = landmarks['right_shoulder']
        rel_y_l = lw[1] - ls[1]
        rel_y_r = rw[1] - rs[1]
        height_ok = (P['min_wrist_rel_y'] <= rel_y_l <= P['max_wrist_rel_y']) and \
                    (P['min_wrist_rel_y'] <= rel_y_r <= P['max_wrist_rel_y'])
        
        if not height_ok:
            is_up = False

        self._process_state_machine(
            self.state._exercise_states['chest_pull'],
            is_up, is_down,
            current_pos=current_pos_combined,
            min_dist=0.02,
            debounce_time=0.3
        )

    def _analyze_lateral_raise_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']

        x_abs_diff = abs(wrist[0] - shoulder[0])
        y_diff = wrist[1] - shoulder[1]

        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'

        is_up = x_abs_diff > params['end_threshold_x']
        is_down = x_abs_diff < params['start_threshold_x']
        
        is_safely_down = is_down and y_diff > 0.0 

        self._process_state_machine(
            self.state._exercise_states['lateral'],
            is_up, is_safely_down,
            current_pos=wrist,
            min_dist=params.get('min_distance', 0.02),
            debounce_time=0.3
        )

    def _analyze_front_raise_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']
        hip = landmarks['right_hip']

        torso_length = abs(shoulder[1] - hip[1])
        if torso_length < 0.05: torso_length = 0.3
        y_diff = (wrist[1] - shoulder[1]) / torso_length
        
        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'
        
        is_up = y_diff < params['end_threshold_y']
        is_down = y_diff > params['start_threshold_y']
        
        self._process_state_machine(
            self.state._exercise_states['front'],
            is_up, is_down,
            current_pos=wrist,
            min_dist=params.get('min_distance', 0.015),
            debounce_time=0.3
        )

    def _analyze_overhead_press_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']
        
        y_diff = wrist[1] - shoulder[1]
        
        is_up = y_diff < params['end_threshold_y']
        is_down = abs(y_diff) < params['start_threshold_y']

        self._process_state_machine(
            self.state._exercise_states['overhead'],
            is_up, is_down,
            current_pos=wrist,
            min_dist=params.get('min_distance', 0.01),
            debounce_time=0.3
        )

    def _analyze_diagonal_lift_logic(self, landmarks):
        P = self.config['params'][self.style]
        
        ls = np.array(landmarks['left_shoulder'], dtype=float)
        rs = np.array(landmarks['right_shoulder'], dtype=float)
        le = np.array(landmarks['left_elbow'], dtype=float)
        re = np.array(landmarks['right_elbow'], dtype=float)
        lw = np.array(landmarks['left_wrist'], dtype=float)
        rw = np.array(landmarks['right_wrist'], dtype=float)
        lh = np.array(landmarks['left_hip'], dtype=float)
        rh = np.array(landmarks['right_hip'], dtype=float)
        
        body_height = abs((ls[1] + rs[1]) / 2 - (lh[1] + rh[1]) / 2)
        shoulder_width = abs(ls[0] - rs[0])

        if not hasattr(self.state, '_diag_start_time'):
            self.state._diag_start_time = time.time()
        
        left_result = self._analyze_diagonal_side(
            'left', ls, le, lw, rs, body_height, shoulder_width, 
            self.state._diag_left_state, P
        )
        
        right_result = self._analyze_diagonal_side(
            'right', rs, re, rw, ls, body_height, shoulder_width,
            self.state._diag_right_state, P
        )
        
        if left_result['is_active'] and not right_result['is_active']:
            self.state._diag_active_side = 'left'
            active_result = left_result
        elif right_result['is_active'] and not left_result['is_active']:
            self.state._diag_active_side = 'right'
            active_result = right_result
        elif left_result['is_active'] and right_result['is_active']:
            if left_result['total_disp'] > right_result['total_disp']:
                self.state._diag_active_side = 'left'
                active_result = left_result
            else:
                self.state._diag_active_side = 'right'
                active_result = right_result
        else:
            self.state._diag_active_side = None
            self.state.stage = None
            return
        
        if active_result['is_up']:
            self.state.stage = 'up'
        elif active_result['is_down']:
            self.state.stage = 'down'
        
        self._update_phase(active_result['is_up'], active_result['is_down'])
        
        self.state._overextension_detected = False
        self.state._overextension_type = None

        if active_result['should_count']:
            distance = active_result['total_disp'] * body_height
            self.state.count += 1
            self.state.total_distance += distance
            self.state.total_energy += self.state.BAND_RESISTANCE_N * distance
            
            self._on_rep_completed()
            
            self.state._last_completion_category = 'standard'
            
            self.state._completed_this_frame = True
            self.state._completed_hold_frames = 2

    def _analyze_diagonal_side(self, side, shoulder, elbow, wrist, opp_shoulder,
                           body_height, shoulder_width, state, params):

        current_time = time.time()

        y_diff = (wrist[1] - shoulder[1]) / body_height
        x_diff = abs(wrist[0] - shoulder[0]) / shoulder_width

        is_down = y_diff > params['start_threshold_y']
        is_up = y_diff < params['end_threshold_y']

        if state['start_wrist_y'] is None and is_down:
            state['start_wrist_y'] = wrist[1]
            state['start_wrist_x'] = wrist[0]

        vertical_disp = 0.0
        horizontal_disp = 0.0
        total_disp = 0.0
        total_disp_weighted = 0.0
        if state['start_wrist_y'] is not None:
            vertical_disp = abs(wrist[1] - state['start_wrist_y']) / body_height
            horizontal_disp = abs(wrist[0] - state['start_wrist_x']) / max(shoulder_width, 1e-6)
            total_disp = vertical_disp + horizontal_disp

            total_disp_weighted = 0.75 * vertical_disp + 0.25 * horizontal_disp

        is_active = (total_disp >= params['min_distance'] * 0.5) or is_up or is_down

        diagonal_angle_ok = True
        if state['start_wrist_y'] is not None:
            dy = (wrist[1] - state['start_wrist_y']) / body_height
            dx = (wrist[0] - state['start_wrist_x']) / max(shoulder_width, 1e-6)
            angle_deg = abs(math.degrees(math.atan2(-dy, dx)))
            diagonal_angle_ok = (params['min_diagonal_angle'] <= angle_deg <= params['max_diagonal_angle'])

        is_diagonal = (horizontal_disp >= params['min_horizontal_disp']) or diagonal_angle_ok

        should_count = False
        debounce_time = 0.5 # 对角线动作保持 0.5s

        if (state['movement_state'] == 'down' and is_up
            and current_time - state['last_count_time'] > debounce_time):

            path_a = (total_disp_weighted >= max(params['min_distance'] * 0.75, 0.14)) and is_diagonal

            vertical_ok = vertical_disp >= max(params.get('min_vertical_disp', 0.10), 0.10)
            horiz_min_ok = horizontal_disp >= max(params['min_horizontal_disp'] * 0.4, 0.035)
            path_b = vertical_ok and horiz_min_ok

            if path_a or path_b:
                should_count = True
                state['movement_state'] = 'up'
                state['last_count_time'] = current_time

        elif state['movement_state'] == 'up' and is_down:
            state['movement_state'] = 'down'
            state['start_wrist_y'] = wrist[1]
            state['start_wrist_x'] = wrist[0]

        return {
            'side': side,
            'is_active': is_active,
            'is_up': is_up,
            'is_down': is_down,
            'vertical_disp': vertical_disp,
            'horizontal_disp': horizontal_disp,
            'total_disp': total_disp,
            'is_diagonal': is_diagonal,
            'should_count': should_count,
        }

    def _generate_feedback(self):
        if self.state._overextension_detected:
            if self.state._overextension_type == 'height_up':
                self.state.feedback = "手臂舉得太高了，請放低一些！"
            elif self.state._overextension_type == 'depth':
                self.state.feedback = "手臂太靠後了，請往前一些！"
            else:
                self.state.feedback = "動作幅度過大，請小心一點！"
            return
        
        if self.exercise_id == 'diagonal_lift':
            side_name = '左側' if self.state._diag_active_side == 'left' else '右側'
            if self.state.stage == 'down':
                self.state.feedback = f"✓ {side_name}準備好，開始對角線拉"
            elif self.state.stage == 'up':
                self.state.feedback = f"✓ {side_name}很好！保持對角線方向"
            else:
                self.state.feedback = "請開始動作"
            return
        
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
                self.state.feedback = "你已準備好，隨時開始運動"
        
        elif self.style == 'beginner':
            if self.state.stage == 'up':
                self.state.feedback = "漂亮！你太棒了！"
            elif self.state.stage == 'down':
                self.state.feedback = "加油！再來一個！"
            else:
                self.state.feedback = "你已準備好，隨時開始運動"

        if self.state._overextension_detected:
            if self.state._overextension_type == 'over_extension':
                self.state.feedback = "動作過度伸展，請減小幅度！"
            elif self.state._overextension_type == 'out_of_range':
                self.state.feedback = "動作偏離標準範圍，請注意姿勢！"
            elif self.state._overextension_type == 'height_up':
                self.state.feedback = "手臂舉得太高了，請放低一些！"
            elif self.state._overextension_type == 'depth':
                self.state.feedback = "手臂太靠後了，請往前一些！"
            else:
                self.state.feedback = "動作幅度過大，請小心一點！"
            return