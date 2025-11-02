from flask import Blueprint, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np

from .analyzer import ExerciseAnalyzer 
from .gesture_recognizer import get_gesture_recognizer  # ← 新增

analyzer = ExerciseAnalyzer() 

# 创建蓝图
mediapipe_bp = Blueprint('mediapipe', __name__)
CORS(mediapipe_bp)

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0)

# ========== 移除旧的 MediaPipe Hands ==========
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(...)
# wait_gesture_detector = GestureDetector(...)
# thumb_up_detector = GestureDetector(...)

# ========== 新增：HuggingFace 手势识别器（懒加载）==========
hf_gesture_recognizer = None


@mediapipe_bp.route('/control', methods=['POST'])
def control_workout():
    data = request.json
    action = data.get('action')

    if action == 'start':
        exercise_id = data.get('exercise')
        style = data.get('style')
        
        if not exercise_id:
            return jsonify({"error": "Exercise type is required"}), 400
            
        try:
            analyzer.setup(exercise_id, style)
            # analyzer.state.is_paused 已经通过 reset() 设置为 True
            return jsonify({
                "status": "started", 
                "exercise": exercise_id,
                "paused": True,
                "message": "請做 👍 手勢開始運動"
            })
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    elif action == 'reset':
        analyzer.reset()
        # analyzer.state.is_paused 已经通过 reset() 设置为 True
        print("Analyzer reset.")
        return jsonify({
            "status": "reset",
            "paused": True,
            "message": "請做 👍 手勢開始運動"
        })
        
    return jsonify({"error": "Invalid action"}), 400


@mediapipe_bp.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    """
    分析视频流：
    1. 检测手势（like → 继续，stop → 暂停）
    2. 分析运动姿态
    """
    global hf_gesture_recognizer
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    # 读取图像
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
    
    # ========== 1. 手势识别（HuggingFace + MPS）==========
    gesture_detected_type = None
    
    # 懒加载手势识别器
    if hf_gesture_recognizer is None:
        hf_gesture_recognizer = get_gesture_recognizer()
    
    # 检测手势
    gesture_result = hf_gesture_recognizer.predict(image_rgb, confidence_threshold=0.6)
    current_gesture = gesture_result['gesture']
    
    # 检测稳定的 stop 手势（暂停）
    if hf_gesture_recognizer.detect_stable_gesture(
        image_rgb, 
        target_gesture='stop',  # ← 根据你的模型，可能是 'stop' 或 'palm'
        confidence_threshold=0.6
    ):
        analyzer.state.is_paused = True
        gesture_detected_type = 'stop'
        hf_gesture_recognizer.reset_buffer()
    
    # 检测稳定的 like 手势（继续）
    elif hf_gesture_recognizer.detect_stable_gesture(
        image_rgb,
        target_gesture='like',
        confidence_threshold=0.6
    ):
        analyzer.state.is_paused = False
        gesture_detected_type = 'like'
        hf_gesture_recognizer.reset_buffer()
    
    # ========== 2. 运动分析 ==========
    if analyzer.state.is_paused:
        analysis_results = {
            'count': analyzer.state.count,
            'stage': analyzer.state.stage,
            'feedback': "已暂停，请做 👍 手势继续",
            'paused': True,
            'energy': analyzer.state.total_energy
        }
    else:
        analysis_results = analyzer.process(image_rgb)

    # ========== 3. 返回结果 ==========
    response_data = {
        analyzer.exercise_id: analysis_results,   # 旧格式（向后兼容）
        'exercise': analyzer.exercise_id,         # 新格式
        'analysis': analysis_results,             # 新格式
        'gesture_detected': gesture_detected_type,
        'current_gesture': current_gesture,       # 新增：显示当前识别的手势（调试用）
        'gesture_confidence': gesture_result['confidence']  # 新增：置信度
    }

    return jsonify(response_data)


@mediapipe_bp.route('/status', methods=['GET'])
def get_status():
    """获取统一分析器的当前状态"""
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


@mediapipe_bp.route('/device-info', methods=['GET'])
def get_device_info():
    """获取手势识别器的设备信息（自动初始化）"""
    global hf_gesture_recognizer
    
    # 自动初始化
    if hf_gesture_recognizer is None:
        try:
            hf_gesture_recognizer = get_gesture_recognizer()
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"初始化失败: {str(e)}"
            }), 500
    
    device_info = hf_gesture_recognizer.get_device_info()
    return jsonify({
        "status": "loaded",
        **device_info
    })