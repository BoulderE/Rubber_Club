from flask import request, jsonify
import cv2
import numpy as np
import os
import tempfile

from ..models.workout import WorkoutState
from ..utils.image_utils import save_temp_image, read_and_delete_image

# 模块级变量
workout_state = WorkoutState()
mp_pose = None
pose = None

def init_mediapipe(mp):
    """初始化MediaPipe"""
    global mp_pose, pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def analyze_movement(right_shoulder, right_elbow, right_wrist):
    """分析动作并更新workout_state"""
    from ..models.workout import CHEST_PULL_THRESHOLD_X, OVER_EXTENSION_THRESHOLD_Y, BAND_RESISTANCE_N, ENERGY_TO_CALORIES
    
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
            if workout_state.start_position is not None and workout_state.end_position is not None:
                # 计算移动距离
                distance = math.sqrt((workout_state.end_position[0] - workout_state.start_position[0])**2 +
                                    (workout_state.end_position[1] - workout_state.start_position[1])**2)
                workout_state.total_distance += distance
                # 计算能量消耗
                energy = BAND_RESISTANCE_N * distance
                workout_state.total_energy += energy
                # 计算卡路里消耗
                calories = energy * ENERGY_TO_CALORIES
                workout_state.total_calories += calories
                workout_state.chest_pull_count += 1

def register_routes(app, mp):
    """注册所有路由到Flask应用"""
    init_mediapipe(mp)
    
    @app.route('/analyze', methods=['POST'])
    def analyze_frame():
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # 保存临时文件
        temp_file_path = save_temp_image(file)
        
        # 读取图像
        frame = read_and_delete_image(temp_file_path)
        
        if frame is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # 转换为 RGB 图像并处理
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 提取关键点
            landmarks = results.pose_landmarks.landmark
            # 获取右肩、右肘、右手腕的坐标
            right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
            right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
            right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
            
            # 运动分析逻辑
            analyze_movement(right_shoulder, right_elbow, right_wrist)
        
        return jsonify({
            'count': workout_state.chest_pull_count,
            'distance': workout_state.total_distance,
            'energy': workout_state.total_energy,
            'calories': workout_state.total_calories,
            'overextension': workout_state.overextension_detected,
            'paused': workout_state.is_paused
        })
    
    @app.route('/analyze-stream', methods=['POST'])
    def analyze_stream():
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # 保存临时文件
        temp_file_path = save_temp_image(file)
        
        # 读取图像
        frame = read_and_delete_image(temp_file_path)
        
        if frame is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # 转换为 RGB 图像并处理
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        response_data = {
            'count': workout_state.chest_pull_count,
            'distance': workout_state.total_distance,
            'energy': workout_state.total_energy,
            'calories': workout_state.total_calories,
            'overextension': workout_state.overextension_detected,
            'paused': workout_state.is_paused,
            'landmarks': None
        }
        
        if results.pose_landmarks:
            # 提取关键点
            landmarks = results.pose_landmarks.landmark
            # 获取右肩、右肘、右手腕的坐标
            right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
            right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
            right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
            
            # 运动分析逻辑
            analyze_movement(right_shoulder, right_elbow, right_wrist)
            
            # 添加关键点数据，方便前端可视化
            keypoints = {
                'right_shoulder': {'x': float(right_shoulder[0]), 'y': float(right_shoulder[1])},
                'right_elbow': {'x': float(right_elbow[0]), 'y': float(right_elbow[1])},
                'right_wrist': {'x': float(right_wrist[0]), 'y': float(right_wrist[1])}
            }
            response_data['landmarks'] = keypoints
        
        return jsonify(response_data)

    @app.route('/control', methods=['POST'])
    def control_workout():
        data = request.json
        if not data or 'action' not in data:
            return jsonify({'error': 'Invalid request'}), 400
        
        if data['action'] == 'pause':
            workout_state.is_paused = True
        elif data['action'] == 'resume':
            workout_state.is_paused = False
        elif data['action'] == 'reset':
            workout_state.reset()
        
        return jsonify({'status': 'success'})

    @app.route('/status', methods=['GET'])
    def get_status():
        return jsonify({
            'count': workout_state.chest_pull_count,
            'distance': workout_state.total_distance,
            'energy': workout_state.total_energy,
            'calories': workout_state.total_calories,
            'overextension': workout_state.overextension_detected,
            'paused': workout_state.is_paused
        })