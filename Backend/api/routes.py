from flask import Blueprint, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np

from .analyzer import ExerciseAnalyzer 
from .gesture_classification import GestureDetector, is_wait_gesture, is_thumbs_up

analyzer = ExerciseAnalyzer() 
# 创建蓝图
mediapipe_bp = Blueprint('mediapipe', __name__)
CORS(mediapipe_bp)

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, model_complexity=0)

# 手势检测器
wait_gesture_detector = GestureDetector(buffer_size=5)
thumb_up_detector = GestureDetector(buffer_size=5)

@mediapipe_bp.route('/control', methods=['POST'])
def control_workout():
    data = request.json
    action = data.get('action')

    if action == 'start':
        exercise_id = data.get('exercise') # e.g., 'lateral_raise'
        style = data.get('style')
        
        if not exercise_id:
            return jsonify({"error": "Exercise type is required"}), 400
            
        try:
            analyzer.setup(exercise_id, style)
            return jsonify({"status": "started", "exercise": exercise_id})
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    elif action == 'reset':
        analyzer.reset()
        print("Analyzer reset.")
        return jsonify({"status": "reset"})
        
    return jsonify({"error": "Invalid action"}), 400

# 路由：分析视频流
# @mediapipe_bp.route('/analyze-stream', methods=['POST'])
# def analyze_stream():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']

#     # 【优化】直接从内存读取图像，避免磁盘I/O，效率更高
#     try:
#         in_memory_file = np.frombuffer(file.read(), np.uint8)
#         frame = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
#         if frame is None:
#             return jsonify({'error': 'Invalid image file'}), 400
#     except Exception as e:
#         return jsonify({'error': f'Failed to process image: {str(e)}'}), 400

#     if not analyzer.config:
#         return jsonify({'error': 'Analyzer not configured. Please send a "start" command first.'}), 400

#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     gesture_detected_type = None

#     hands_results = hands.process(image_rgb)
#     if hands_results.multi_hand_landmarks:
#         for hand_landmarks in hands_results.multi_hand_landmarks:
#             landmarks = hand_landmarks.landmark
#             if wait_gesture_detector.detect_stable_gesture(is_wait_gesture, landmarks):
#                 analyzer.state.is_paused = True # 直接控制状态
#                 gesture_detected_type = 'wait'
#             elif thumb_up_detector.detect_stable_gesture(is_thumbs_up, landmarks):
#                 analyzer.state.is_paused = False # 直接控制状态
#                 gesture_detected_type = 'thumbs_up'

#     if analyzer.state.is_paused:
#         analysis_results = {
#             'count': analyzer.state.count,
#             'stage': analyzer.state.stage,
#             'feedback': "已暂停",
#             'paused': True,
#             'energy': analyzer.state.total_energy
#         }
#     else:
#         # 调用您正确的 process 方法
#         analysis_results = analyzer.process(image_rgb)

#     response_data = {
#         # 动态地使用当前运动的名称作为键
#         analyzer.exercise_id: analysis_results,
#         'gesture_detected': None
#     }

#     return jsonify(response_data)

@mediapipe_bp.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    # 【优化】直接从内存读取图像，避免磁盘I/O，效率更高
    try:
        in_memory_file = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image file'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400

    if not analyzer.config:
        return jsonify({'error': 'Analyzer not configured. Please send a "start" command first.'}), 400

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gesture_detected_type = None

    hands_results = hands.process(image_rgb)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            if wait_gesture_detector.detect_stable_gesture(is_wait_gesture, landmarks):
                analyzer.state.is_paused = True  # 直接控制状态
                gesture_detected_type = 'wait'
            elif thumb_up_detector.detect_stable_gesture(is_thumbs_up, landmarks):
                analyzer.state.is_paused = False  # 直接控制状态
                gesture_detected_type = 'thumbs_up'

    if analyzer.state.is_paused:
        analysis_results = {
            'count': analyzer.state.count,
            'stage': analyzer.state.stage,
            'feedback': "已暂停",
            'paused': True,
            'energy': analyzer.state.total_energy
        }
    else:
        analysis_results = analyzer.process(image_rgb)

    # 兼容：保留“动态键”；同时提供稳定键 'exercise' 与 'analysis'
    response_data = {
        analyzer.exercise_id: analysis_results,   # 旧：动态键（向后兼容）
        'exercise': analyzer.exercise_id,         # 新：稳定字段
        'analysis': analysis_results,             # 新：稳定字段
        'gesture_detected': gesture_detected_type
    }

    return jsonify(response_data)

# 路由：获取训练状态
@mediapipe_bp.route('/status', methods=['GET'])
def get_status():
    """【升级】获取统一分析器的当前状态"""
    if not analyzer.config:
        return jsonify({"status": "not_configured"})

    return jsonify({
        "status": "configured",
        "exercise": analyzer.config['name'],
        "style": analyzer.style,
        "count": analyzer.state.count,
        "stage": analyzer.state.stage,
        "is_paused": analyzer.state.is_paused,
        "current_feedback": analyzer.state.feedback,
        "total_energy": analyzer.state.total_energy
    })