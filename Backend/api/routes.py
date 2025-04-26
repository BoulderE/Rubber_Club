from flask import Blueprint, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

from .chest_pull import analyze_chest_pull, workout_state as chest_pull_state
from .lateral_raise import analyze_movement, workout_state as lateral_raise_state
from .gesture_classification import GestureDetector, is_wait_gesture, is_five_fingers_open

# 创建蓝图
mediapipe_bp = Blueprint('mediapipe', __name__)

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, model_complexity=0)

# 手势检测器
wait_gesture_detector = GestureDetector(buffer_size=5)
five_fingers_detector = GestureDetector(buffer_size=5)

# 路由：分析视频流
@mediapipe_bp.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # 保存临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_file.name)
    temp_file.close()
    
    # 读取图像
    frame = cv2.imread(temp_file.name)
    os.unlink(temp_file.name)  # 删除临时文件
    
    if frame is None:
        return jsonify({'error': 'Invalid image file'}), 400
    # 初始化响应数据
    response_data = {
        'lateral_raise': {
            'count': lateral_raise_state.lateral_rise_count,
            'energy': lateral_raise_state.total_energy,
            'overextension': lateral_raise_state.overextension_detected,
            'paused': lateral_raise_state.is_paused
        },
        'chest_pull': {
            'count': chest_pull_state.chest_pull_count,
            'energy': chest_pull_state.total_energy,
            'overextension': chest_pull_state.overextension_detected,
            'paused': chest_pull_state.is_paused
        },
        'gesture_detected': None
    }

    # 转换为 RGB 图像
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手势检测
    hands_results = hands.process(image_rgb)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # 检测暂停手势
            if wait_gesture_detector.detect_stable_gesture(is_wait_gesture, landmarks):
                lateral_raise_state.is_paused = True
                chest_pull_state.is_paused = True
                response_data['gesture_detected'] = 'wait'

            # 检测恢复手势
            elif five_fingers_detector.detect_stable_gesture(is_five_fingers_open, landmarks):
                lateral_raise_state.is_paused = False
                chest_pull_state.is_paused = False
                response_data['gesture_detected'] = 'five_fingers_open'

    # 姿势分析
    pose_results = pose.process(image_rgb)
    if not pose_results.pose_landmarks:
        print("Error: No pose landmarks detected")
        return jsonify({'error': 'No landmarks detected'}), 400

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        # 获取右肩、右肘、右手腕的坐标
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])

        # 分别分析两个动作
        if not lateral_raise_state.is_paused:
            analyze_movement(right_shoulder, right_elbow, right_wrist)
        if not chest_pull_state.is_paused:
            analyze_chest_pull(right_shoulder, right_elbow, right_wrist)

        keypoints = {
            'right_shoulder': {'x': float(right_shoulder[0]), 'y': float(right_shoulder[1])},
            'right_elbow': {'x': float(right_elbow[0]), 'y': float(right_elbow[1])},
            'right_wrist': {'x': float(right_wrist[0]), 'y': float(right_wrist[1])}
        }
    
    # 将关键点数据添加到响应中
        response_data['landmarks'] = keypoints

        # 更新响应数据
        response_data['lateral_raise'].update({
            'count': int(lateral_raise_state.lateral_rise_count),
            'energy': float(lateral_raise_state.total_energy),
            'overextension': bool(lateral_raise_state.overextension_detected),
            'paused': bool(lateral_raise_state.is_paused)
        })
        response_data['chest_pull'].update({
            'count': int(chest_pull_state.chest_pull_count),
            'energy': float(chest_pull_state.total_energy),
            'overextension': bool(chest_pull_state.overextension_detected),
            'paused': bool(chest_pull_state.is_paused)
        })

    return jsonify(response_data)

# 路由：控制训练状态
@mediapipe_bp.route('/control', methods=['POST'])
def control_workout():
    """控制训练状态（暂停/继续/重置）"""
    data = request.json
    if not data or 'action' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    action = data['action']
    if action == 'pause':
        lateral_raise_state.is_paused = True
        chest_pull_state.is_paused = True
    elif action == 'resume':
        lateral_raise_state.is_paused = False
        chest_pull_state.is_paused = False
    elif action == 'reset':
        lateral_raise_state.reset()
        chest_pull_state.reset()

    return jsonify({'status': 'success'})

# 路由：获取训练状态
@mediapipe_bp.route('/status', methods=['GET'])
def get_status():
    """获取当前训练状态"""
    return jsonify({
        'lateral_raise': {
            'count': lateral_raise_state.lateral_rise_count,
            'energy': lateral_raise_state.total_energy,
            'overextension': lateral_raise_state.overextension_detected,
            'paused': lateral_raise_state.is_paused
        },
        'chest_pull': {
            'count': chest_pull_state.chest_pull_count,
            'energy': chest_pull_state.total_energy,
            'overextension': chest_pull_state.overextension_detected,
            'paused': chest_pull_state.is_paused
        }
    })