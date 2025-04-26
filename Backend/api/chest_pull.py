import cv2
import mediapipe as mp
import numpy as np
import math
# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0)

# 定义常量
# CHEST_PULL_THRESHOLD_Y = 0.2  # 手腕抬起到肩膀以上的阈值
CHEST_PULL_THRESHOLD_Y = 0.05
OVER_EXTENSION_THRESHOLD_Y = -0.2  # 手腕或手肘高度显著高于肩膀的阈值
BAND_RESISTANCE_N = 25 * 9.81
# 状态变量
pull_active = False

class WorkoutState:
    def __init__(self):
        self.reset()

    def reset(self):
        # Chest Pull 状态
        self.chest_pull_count = 0  # 完成次数
        self.chest_pull_active = False  # 是否处于动作执行状态
        self.chest_pull_start_position = None  # 动作起始点
        self.chest_pull_end_position = None  # 动作结束点

        # 通用状态
        self.overextension_detected = False  # 是否检测到过伸
        self.total_distance = 0  # 累计移动距离
        self.total_energy = 0  # 累计能量消耗
        self.is_paused = False  # 是否暂停

workout_state = WorkoutState()

def analyze_chest_pull(right_shoulder, right_elbow, right_wrist):
    if workout_state.is_paused:
        return

    wrist_to_shoulder_y_diff = right_wrist[1] - right_shoulder[1]
    elbow_to_shoulder_y_diff = right_elbow[1] - right_shoulder[1]

    workout_state.overextension_detected = (
        right_elbow[1] < right_shoulder[1] + OVER_EXTENSION_THRESHOLD_Y 
        or right_wrist[1] < right_shoulder[1] + OVER_EXTENSION_THRESHOLD_Y
    )

    if not workout_state.chest_pull_active:
        if wrist_to_shoulder_y_diff < -CHEST_PULL_THRESHOLD_Y and elbow_to_shoulder_y_diff < -CHEST_PULL_THRESHOLD_Y:
            workout_state.chest_pull_active = True
            workout_state.chest_pull_start_position = right_wrist
    elif workout_state.chest_pull_active:
        
       if wrist_to_shoulder_y_diff > CHEST_PULL_THRESHOLD_Y and elbow_to_shoulder_y_diff > CHEST_PULL_THRESHOLD_Y:
            workout_state.chest_pull_active = False
            workout_state.chest_pull_end_position = right_wrist

            # 确保起始和结束位置都不为空
            if workout_state.chest_pull_start_position is not None and workout_state.chest_pull_end_position is not None:
                # 计算移动距离
                distance = math.sqrt((workout_state.chest_pull_end_position[0] - workout_state.chest_pull_start_position[0])**2 +
                                     (workout_state.chest_pull_end_position[1] - workout_state.chest_pull_start_position[1])**2)
                workout_state.total_distance += distance

                # 计算能量消耗
                energy = BAND_RESISTANCE_N * distance
                workout_state.total_energy += energy

                # Chest Pull 动作计数增加
                workout_state.chest_pull_count += 1