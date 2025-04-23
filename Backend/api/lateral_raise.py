import mediapipe as mp
import numpy as np
import math

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0)

# 定义常量
CHEST_PULL_THRESHOLD_X = 0.05  # 手腕或手肘抬起高度达到肩膀的阈值（完成动作）
OVER_EXTENSION_THRESHOLD_Y = -0.2  # 手腕或手肘高度超过肩膀的阈值（检测过伸）
BAND_RESISTANCE_N = 25 * 9.81

# 训练状态类
class WorkoutState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.lateral_rise_count = 0
        self.pull_active = False
        self.overextension_detected = False
        self.start_position = None
        self.end_position = None
        self.total_distance = 0
        self.total_energy = 0
        self.is_paused = False

# 初始化全局状态
workout_state = WorkoutState()

def analyze_movement(right_shoulder, right_elbow, right_wrist):
    """分析侧平举动作"""
    if workout_state.is_paused:
        return
    
    # 计算手腕和手肘与肩膀的水平距离
    wrist_to_shoulder_x_diff = abs(right_wrist[0] - right_shoulder[0])
    elbow_to_shoulder_x_diff = abs(right_elbow[0] - right_shoulder[0])
    
    # Overextension 检测
    if (right_elbow[1] < right_shoulder[1] + OVER_EXTENSION_THRESHOLD_Y or
        right_wrist[1] < right_shoulder[1] + OVER_EXTENSION_THRESHOLD_Y):
        workout_state.overextension_detected = True
    else:
        workout_state.overextension_detected = False
    
    if not workout_state.pull_active:
        # 起始状态：手腕和手肘的水平距离接近肩膀
        if wrist_to_shoulder_x_diff < CHEST_PULL_THRESHOLD_X and elbow_to_shoulder_x_diff < CHEST_PULL_THRESHOLD_X:
            workout_state.pull_active = True
            workout_state.start_position = right_wrist
    
    elif workout_state.pull_active:
        # 完成状态：手腕和手肘的水平距离达到阈值
        if wrist_to_shoulder_x_diff > CHEST_PULL_THRESHOLD_X and elbow_to_shoulder_x_diff > CHEST_PULL_THRESHOLD_X:
            workout_state.pull_active = False
            workout_state.end_position = right_wrist
            
            # 确保起始和结束位置都不为空
            if workout_state.start_position is not None and workout_state.end_position is not None:
                # 计算移动距离
                # distance = math.sqrt((workout_state.end_position[0] - workout_state.start_position[0])**2 +
                #                     (workout_state.end_position[1] - workout_state.start_position[1])**2)
                distance = float(np.linalg.norm(workout_state.end_position - workout_state.start_position))
                workout_state.total_distance += distance
                
                # 计算能量消耗
                energy = float(BAND_RESISTANCE_N * distance)
                workout_state.total_energy += energy
                
                workout_state.lateral_rise_count += 1