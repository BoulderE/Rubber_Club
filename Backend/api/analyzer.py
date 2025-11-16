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
                # 'over_extension_threshold_y': -0.30,
                'min_distance': 0.015
            },
            'beginner': {
                'start_threshold_y': 0.75,
                'end_threshold_y': 0.30,
                # 'over_extension_threshold_y': -0.20,
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

        'min_horizontal_disp': 0.08,   # 原 0.20
        'min_vertical_disp':   0.10,
        'min_distance':        0.18,

        'min_diagonal_angle':  8,
        'max_diagonal_angle':  85,

        'max_shoulder_z_diff': 0.20,   # 原 0.10
        'shoulder_z_window':   15,     # 原 10
    },
    'beginner': {
        'start_threshold_y': 0.22,
        'end_threshold_y': 0.05,

        # 初学更宽松
        'min_horizontal_disp': 0.08,   # 原 0.20
        'min_vertical_disp':   0.10,
        'min_distance':        0.18,

        'min_diagonal_angle':  8,
        'max_diagonal_angle':  85,

        'max_shoulder_z_diff': 0.20,   # 原 0.10
        'shoulder_z_window':   15,     # 原 10
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
        self.feedback = "請做 👍 手勢開始運動"  # ← 修改提示
        self.is_paused = True
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
        self._diag_left_state = {
            'movement_state': 'down',
            'start_wrist_y': None,
            'start_wrist_x': None,
            'shoulder_z_history': [],
            'last_count_time': 0,
        }
        self._diag_right_state = {
            'movement_state': 'down',
            'start_wrist_y': None,
            'start_wrist_x': None,
            'shoulder_z_history': [],
            'last_count_time': 0,
        }
        self._diag_active_side = None

        #new
        self.up_durations = []      
        self.down_durations = []    
        self.current_phase = None   
        self.phase_start_time = None  
        

class ExerciseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
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

    def _now(self):
        return time.time()
    
    def _finalize_open_phase(self):
        """
        在离开追踪或暂停时，结束当前正在进行的阶段
        """
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
        
        # 重置阶段状态
        self.state.current_phase = None
        self.state.phase_start_time = None

    def _update_phase(self, is_up: bool, is_down: bool):
        """
        更新运动阶段并记录各阶段时长
        
        Args:
            is_up: 是否在 up 区域
            is_down: 是否在 down 区域
        """
        now_t = self._now()
        
        # 🔹 确定当前应该处于的阶段
        target_phase = None
        if is_up:
            target_phase = 'up'
        elif is_down:
            target_phase = 'down'
        else:
            # 🔹 在中间过渡区域
            # 选项 A: 保持当前阶段（不做任何事）
            # 选项 B: 结束当前阶段（这里我们选择 A，更稳定）
            return
        
        # 🔹 检测阶段切换
        if self.state.current_phase != target_phase:
            
            # 🔸 结束旧阶段，记录时长
            if self.state.current_phase is not None and self.state.phase_start_time is not None:
                duration = now_t - self.state.phase_start_time
                
                # 🔸 过滤异常值（太短或太长的都不记录）
                if 0.1 <= duration <= 10.0:
                    if self.state.current_phase == 'up':
                        self.state.up_durations.append(duration)
                        # 限制列表长度
                        if len(self.state.up_durations) > 50:
                            self.state.up_durations.pop(0)
                        print(f"[DEBUG] UP 阶段时长: {duration:.2f}秒")
                    
                    elif self.state.current_phase == 'down':
                        self.state.down_durations.append(duration)
                        if len(self.state.down_durations) > 50:
                            self.state.down_durations.pop(0)
                        print(f"[DEBUG] DOWN 阶段时长: {duration:.2f}秒")
            
            # 🔸 开始新阶段
            self.state.current_phase = target_phase
            self.state.phase_start_time = now_t


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
        基于 up 和 down 阶段时长的稳定性评分（分阶段版本）
        
        流程：
        1. 分别取最近 10-15 个 up 和 down 阶段时长
        2. 对每个阶段，用 median 归一化并计算标准差
        3. 用 sigmoid 将标准差映射到分数
        4. 按样本数加权平均得到最终分数
        """
        import statistics
        
        def phase_score(durations, phase_name):
            """
            计算单个阶段的稳定性分数
            
            Args:
                durations: 时长列表 [1.2, 1.3, 1.1, ...]
                phase_name: 阶段名称（用于调试输出）
            
            Returns:
                int: 10-100 的分数，或 None（样本不足）
            """
            if len(durations) < 3:
                return None
            
            if len(durations) >= 5:  # 样本足够多时才过滤
                q1 = statistics.quantiles(durations, n=4)[0]  # 25分位
                q3 = statistics.quantiles(durations, n=4)[2]  # 75分位
                iqr = q3 - q1
                
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                filtered = [d for d in durations if lower <= d <= upper]

                if len(filtered) >= 3:
                    durations = filtered
                            
            # 🔹 计算 median（中位数更稳健，不受极端值影响）
            med = statistics.median(durations)
            if med <= 1e-6 or not math.isfinite(med):
                return None
            
            # 🔹 归一化每个值：normalized = value / median
            # 这样可以消除"up 天然比 down 快"的影响
            normalized = [d / med for d in durations]
            
            # 🔹 计算归一化标准差
            # 标准差越小 = 节奏越稳定 = 分数越高
            std_normalized = statistics.pstdev(normalized)
            
            print(f"[DEBUG] {phase_name} 阶段:")
            print(f"        中位数: {med:.2f}秒")
            print(f"        归一化标准差: {std_normalized:.3f}")
            
            # 🔹 Sigmoid 映射到 10-100
            # 参数说明：
            # - x0: 中心点（标准差为 x0 时得 55 分）
            # - k: 陡峭度（越大越敏感）
            x0 = 0.20  # 标准差 < 20% 为优秀
            k = 15     # 陡峭度系数
            
            sigmoid_value = 1.0 / (1.0 + math.exp(k * (std_normalized - x0)))
            score = 10 + 90 * sigmoid_value
            
            print(f"        分数: {int(round(score))}")
            
            return int(round(score))
        
        # 🔸 获取最近的时长数据
        recent_up = self.state.up_durations[-15:]    # 最近 15 个 up
        recent_down = self.state.down_durations[-15:]  # 最近 15 个 down
        
        # 🔸 计算各阶段分数
        up_score = phase_score(recent_up, "UP")
        down_score = phase_score(recent_down, "DOWN")
        
        # 🔸 合并分数
        if up_score is None and down_score is None:
            return 100  # 样本不足，返回满分
        
        if up_score is None:
            return down_score
        
        if down_score is None:
            return up_score
        
        # 🔸 加权平均（按样本数加权）
        w_up = len(recent_up)
        w_down = len(recent_down)
        final_score = (up_score * w_up + down_score * w_down) / (w_up + w_down)
        
        print(f"[DEBUG] 最终 Smoothness: {int(round(final_score))}")
        
        return int(round(final_score))

    def get_metrics(self):
        # 供路由返回
        return {
            "rep_count": self.rep_count,
            "smoothness": self.smoothness_score,
            "rep_durations": self.repetition_durations[-20:],  # 可限长返回
        }

    def setup(self, exercise_type: str, style: str):
        if exercise_type not in EXERCISE_CONFIG:
            raise ValueError(f"不支援的運動類型:{exercise_type}")
        
        self.config = EXERCISE_CONFIG[exercise_type]
        self.style = style if style in ['intermediate', 'beginner'] else 'intermediate'
        self.exercise_id = exercise_type
        self.reset()
        print(f"分析器已設定為: 運動='{self.config['name']}', 模式='{self.style}' (使用專屬評價標準)")

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
            self._finalize_open_phase()
            self.state.feedback = "已暫停，做按讚手勢繼續"
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
            self._finalize_open_phase()
            self.state.feedback = "請確保身體關鍵部位在鏡頭內"

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
            'rep_durations': self.repetition_durations[-20:],
            'phase_timings': self.phase_timings[-20:],
            'up_durations': self.state.up_durations[-10:],
            'down_durations': self.state.down_durations[-10:],
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

        is_down = y_diff > params['end_threshold_y']          # 手较低
        is_up = y_diff < params['start_threshold_y']          # 手较高（靠近肩上方）

        self._update_phase(is_up, is_down)

        if is_up:
            self.state.stage = 'up'
        elif is_down:
            self.state.stage = 'down'

        if not self.state._action_active:
            if is_up:
                self.state._action_active = True
                self.state._start_position = wrist
                self.state._action_overextended = False
        elif is_down:
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

        self._update_phase(is_up, is_down)

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
        修正后的侧平举分析逻辑 - 修复过伸时重复计数问题
        """
        params = self.config['params'][self.style]
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']

        x_abs_diff = abs(wrist[0] - shoulder[0])
        y_diff = wrist[1] - shoulder[1]

        # ========== 1. 过伸检测 ==========
        self.state._overextension_detected = False
        self.state._overextension_type = None
        
        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'
            if self.state._action_active:
                self.state._action_overextended = True

        # ========== 2. 阶段判断 ==========
        is_up = x_abs_diff > params['end_threshold_x']
        is_down = x_abs_diff < params['start_threshold_x']
        
        # 🔧 新增：安全区域判断 - 手腕必须低于肩膀才算真正的 down
        is_safely_down = is_down and y_diff > -0.05  # y_diff > -0.05 表示手腕不高于肩膀太多

        self._update_phase(is_up, is_down)
        
        if is_up:
            self.state.stage = 'up'
        elif is_down:
            self.state.stage = 'down'

        # ========== 3. 计数逻辑 ==========
        
        # 🔧 修复：只有在"安全的 down 区域"才能开始新动作
        if is_safely_down and not self.state._action_active:
            self.state._action_active = True
            self.state._in_up_phase = False
            self.state._action_overextended = False
            self.state._start_position = wrist.copy()

        # 到达顶点
        if self.state._action_active and not self.state._in_up_phase and is_up:
            self.state._in_up_phase = True

        # 🔧 修复：完成动作时，也使用"安全的 down"判断
        if self.state._action_active and self.state._in_up_phase and is_safely_down:
            self.state._end_position = wrist.copy()
            distance = float(np.linalg.norm(self.state._end_position - self.state._start_position))
            
            if distance > params.get('min_distance', 0.02):
                self.state.count += 1
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

                self._on_rep_completed()

                is_non_standard = bool(getattr(self.state, '_action_overextended', False))
                self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'

                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2

            # 重置状态
            self.state._action_active = False
            self.state._in_up_phase = False
            self.state._action_overextended = False

    def _analyze_front_raise_logic(self, landmarks):
        """
        重构后的前平举分析逻辑（距离归一化版本）
        使用躯干长度归一化，适应不同距离的用户
        """
        params = self.config['params'][self.style]
        shoulder = landmarks['right_shoulder']
        wrist = landmarks['right_wrist']
        elbow = landmarks['right_elbow']
        hip = landmarks['right_hip']
        
        # ========== 归一化处理 ==========
        # 使用躯干长度作为参考，使阈值不受用户与摄像头距离影响
        torso_length = abs(shoulder[1] - hip[1])
        
        # 防御性编程：如果躯干长度异常小（检测错误），使用默认值
        if torso_length < 0.05:
            torso_length = 0.3
        
        # 计算手腕相对肩膀的垂直距离（原始值）
        y_diff_raw = wrist[1] - shoulder[1]
        
        # 归一化：除以躯干长度，得到相对距离
        y_diff = y_diff_raw / torso_length
        
        # ========== 1. 过伸检测 ==========
        self.state._overextension_detected = False
        self.state._overextension_type = None
        
        if 'over_extension_threshold_y' in params and y_diff < params['over_extension_threshold_y']:
            self.state._overextension_detected = True
            self.state._overextension_type = 'height_up'
            if self.state._action_active:
                self.state._action_overextended = True
        
        # ========== 2. 阶段判断 ==========
        # is_up: 手臂是否举到高位（y_diff 小于阈值，手腕高于肩膀）
        # is_down: 手臂是否放到低位（y_diff 大于阈值，手腕低于肩膀）
        is_up = y_diff < params['end_threshold_y']
        is_down = y_diff > params['start_threshold_y']
        
        # 更新相位追踪（用于 smoothness 计算）
        self._update_phase(is_up, is_down)
        
        # 更新显示的阶段状态
        if is_up:
            self.state.stage = 'up'
        elif is_down:
            self.state.stage = 'down'
        
        # ========== 3. 初始化状态变量 ==========
        if not hasattr(self.state, '_in_up_phase'):
            self.state._in_up_phase = False
        
        # ========== 4. 计数逻辑 - 状态机 ==========
        # 状态机流程: down -> up -> down (完整的一次动作)
        
        # 4.1 动作开始：手臂在 down 位置，且没有正在进行的动作
        if is_down and not self.state._action_active:
            self.state._action_active = True
            self.state._in_up_phase = False
            self.state._action_overextended = False
            self.state._start_position = wrist.copy()
        
        # 4.2 到达顶点：动作进行中，首次达到 up 位置
        if self.state._action_active and not self.state._in_up_phase and is_up:
            self.state._in_up_phase = True
        
        # 4.3 完成动作：已到达顶点，现在回到 down 位置
        if self.state._action_active and self.state._in_up_phase and is_down:
            self.state._end_position = wrist.copy()
            
            # 计算位移距离
            distance = float(np.linalg.norm(self.state._end_position - self.state._start_position))
            
            # 只有位移足够大才计数（防止误触发）
            if distance > params.get('min_distance', 0.010):
                # 计数
                self.state.count += 1
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance
                
                # 触发 rep 完成回调（计算 smoothness）
                self._on_rep_completed()
                
                # 分类：标准 vs 非标准（是否过伸）
                is_non_standard = bool(getattr(self.state, '_action_overextended', False))
                self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'
                
                # 触发完成动画
                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2
            
            # 重置状态，准备下一次动作
            self.state._action_active = False
            self.state._in_up_phase = False
            self.state._action_overextended = False

    def _analyze_overhead_press_logic(self, landmarks):
        params = self.config['params'][self.style]
        shoulder, wrist, elbow = landmarks['right_shoulder'], landmarks['right_wrist'], landmarks['right_elbow']
        
        y_diff = wrist[1] - shoulder[1]
        
        is_up = y_diff < params['end_threshold_y']
        is_down = abs(y_diff) < params['start_threshold_y']

        self._update_phase(is_up, is_down)

        if is_up:
            self.state.stage = 'up'
        elif is_down:
            self.state.stage = 'down'

        if not self.state._action_active:
            self.state._in_up_phase = False
        # 進入起始區且未激活 → 準備開始一輪
        if is_down and not self.state._action_active:
            self.state._action_active = True
            self.state._in_up_phase = False
            self.state._start_position = wrist.copy()
            self.state._action_overextended = False  # 本動作是否過伸（此動作暫不檢測，可保留結構）

        # 首次到達 up 區
        if self.state._action_active and not self.state._in_up_phase and is_up:
            self.state._in_up_phase = True

        # 已到達過 up，現在回到 down → 嘗試完成一次
        if self.state._action_active and self.state._in_up_phase and is_down:
            self.state._end_position = wrist.copy()
            distance = float(np.linalg.norm(self.state._end_position - self.state._start_position))

            if distance > params.get('min_distance', 0.02):
                self.state.count += 1
                self.state.total_distance += distance
                self.state.total_energy += self.state.BAND_RESISTANCE_N * distance

                self._on_rep_completed()
                self.state._last_completion_category = 'standard'
                self.state._completed_this_frame = True
                self.state._completed_hold_frames = 2

            # 本輪結束，復位
            self.state._action_active = False
            self.state._in_up_phase = False
            self.state._action_overextended = False

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

    def _analyze_diagonal_lift_logic(self, landmarks):
        P = self.config['params'][self.style]
        
        # ========== 1. 准备数据 ==========
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
        startup_grace = (time.time() - self.state._diag_start_time) < 0.8
        
        # ========== 2. 调用双侧分析 ==========
        left_result = self._analyze_diagonal_side(
            'left', ls, le, lw, rs, body_height, shoulder_width, 
            self.state._diag_left_state, P  # ← 传入左侧状态
        )
        
        right_result = self._analyze_diagonal_side(
            'right', rs, re, rw, ls, body_height, shoulder_width,
            self.state._diag_right_state, P  # ← 传入右侧状态
        )
        
        # ========== 3. 决策逻辑 ==========
        if left_result['is_active'] and not right_result['is_active']:
            self.state._diag_active_side = 'left'
            active_result = left_result
        elif right_result['is_active'] and not left_result['is_active']:
            self.state._diag_active_side = 'right'
            active_result = right_result
        elif left_result['is_active'] and right_result['is_active']:
            # 兩邊都活躍，選幅度大的
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
        
        # ========== 4. 更新全局状态 ==========
        if active_result['is_up']:
            self.state.stage = 'up'
        elif active_result['is_down']:
            self.state.stage = 'down'
        
        self._update_phase(active_result['is_up'], active_result['is_down'])
        
        self.state._overextension_detected = (not active_result['shoulder_stable']) and (not startup_grace)
        if not active_result['shoulder_stable'] and not startup_grace:
            self.state._overextension_type = 'shoulder_rotation'

        if active_result['should_count']:
            distance = active_result['total_disp'] * body_height
            self.state.count += 1
            self.state.total_distance += distance
            self.state.total_energy += self.state.BAND_RESISTANCE_N * distance
            
            self._on_rep_completed()
            
            is_non_standard = not active_result['shoulder_stable']
            self.state._last_completion_category = 'non_standard' if is_non_standard else 'standard'
            
            self.state._completed_this_frame = True
            self.state._completed_hold_frames = 2

    def _analyze_diagonal_side(self, side, shoulder, elbow, wrist, opp_shoulder,
                           body_height, shoulder_width, state, params):
        """分析單側對角線動作（更贴身、更宽容的版本）"""
        current_time = time.time()

        # 归一化位移
        y_diff = (wrist[1] - shoulder[1]) / body_height
        x_diff = abs(wrist[0] - shoulder[0]) / shoulder_width

        # 分区
        is_down = y_diff > params['start_threshold_y']
        is_up = y_diff < params['end_threshold_y']

        # 肩膀 z 轴稳定性（放宽 + 平滑）
        shoulder_z_diff = abs(shoulder[2] - opp_shoulder[2]) / max(shoulder_width, 1e-6)
        state['shoulder_z_history'].append(shoulder_z_diff)
        if len(state['shoulder_z_history']) > params['shoulder_z_window']:
            state['shoulder_z_history'].pop(0)

        # 基础容差偏移：对极小值直接容忍
        base_tolerance = 0.02  # 额外宽容
        if len(state['shoulder_z_history']) >= 3:
            z_vals = state['shoulder_z_history']
            z_range = max(z_vals) - min(z_vals)
            # 放宽：允许一个基础容差后再比较
            shoulder_stable = (z_range <= (params['max_shoulder_z_diff'] + base_tolerance))
        else:
            shoulder_stable = True

        # 初始化起点
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
            # 竖直权重更高，贴身也能过
            total_disp_weighted = 0.75 * vertical_disp + 0.25 * horizontal_disp

        # 活跃性（用于 UI 阶段/提示，不影响正确性归类）
        is_active = (total_disp >= params['min_distance'] * 0.5) or is_up or is_down

        # 角度辅助（放宽范围）
        diagonal_angle_ok = True
        if state['start_wrist_y'] is not None:
            dy = (wrist[1] - state['start_wrist_y']) / body_height
            dx = (wrist[0] - state['start_wrist_x']) / max(shoulder_width, 1e-6)
            angle_deg = abs(math.degrees(math.atan2(-dy, dx)))  # 0=水平，90=竖直
            diagonal_angle_ok = (params['min_diagonal_angle'] <= angle_deg <= params['max_diagonal_angle'])

        # 放宽对角特征：水平或角度满足其一
        is_diagonal = (horizontal_disp >= params['min_horizontal_disp']) or diagonal_angle_ok

        should_count = False
        debounce_time = 0.5

        if (state['movement_state'] == 'down' and is_up
            and current_time - state['last_count_time'] > debounce_time):
            # 路径A：加权总位移达标 + 有对角线特征
            path_a = (total_disp_weighted >= max(params['min_distance'] * 0.75, 0.14)) and is_diagonal
            # 路径B：更偏竖直的贴身路径：竖直位移达到阈值 + 水平位移有个最小量
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
            'shoulder_stable': shoulder_stable,
            'should_count': should_count,
        }

    def _generate_feedback(self):
        if self.state._overextension_detected:
            if self.state._overextension_type == 'height':
                self.state.feedback = "手臂舉得太高了，請放低一些！"
            elif self.state._overextension_type == 'depth':
                self.state.feedback = "手臂太靠後了，請往前一些！"
            elif self.state._overextension_type == 'shoulder_rotation':
                self.state.feedback = "⚠️ 軀幹過度轉動，保持肩膀穩定"
            else:
                self.state.feedback = "動作幅度過大，請小心一點！"
            return
        
         # 對角線動作專屬反饋
        if self.exercise_id == 'diagonal_lift':
            side_name = '左側' if self.state._diag_active_side == 'left' else '右側'
            if self.state.stage == 'down':
                self.state.feedback = f"✓ {side_name}準備好，開始對角線拉"
            elif self.state.stage == 'up':
                self.state.feedback = f"✓ {side_name}很好！保持對角線方向"
            else:
                self.state.feedback = "請開始動作"
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
                self.state.feedback = "你已準備好，隨時開始運動"
        
        elif self.style == 'beginner':
            if self.state.stage == 'up':
                self.state.feedback = "漂亮！你太棒了！"
            elif self.state.stage == 'down':
                self.state.feedback = "加油！再來一個！"
            else:
                self.state.feedback = "你已準備好，隨時開始運動"