from flask import Blueprint, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np

from .analyzer import ExerciseAnalyzer 
from .gesture_recognizer import get_gesture_recognizer 

analyzer = ExerciseAnalyzer() 

mediapipe_bp = Blueprint('mediapipe', __name__)
CORS(mediapipe_bp)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

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
        print("Analyzer reset.")
        return jsonify({
            "status": "reset",
            "paused": True,
            "message": "請做 👍 手勢開始運動"
        })
        
    return jsonify({"error": "Invalid action"}), 400


@mediapipe_bp.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    global hf_gesture_recognizer
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

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
    
    if hf_gesture_recognizer is None:
        hf_gesture_recognizer = get_gesture_recognizer()
    
    gesture_result = hf_gesture_recognizer.predict(image_rgb, confidence_threshold=0.6)
    current_gesture = gesture_result['gesture']
    
    if analyzer.state.is_paused:
        if hf_gesture_recognizer.detect_stable_gesture(
            image_rgb,
            target_gesture='like',
            confidence_threshold=0.6
        ):
            analyzer.state.is_paused = False
            gesture_detected_type = 'like'
            hf_gesture_recognizer.reset_buffer()
        
        analysis_results = {
            'count': analyzer.state.count,
            'stage': analyzer.state.stage,
            'feedback': "請做 👍 手勢開始運動", 
            'paused': True,
            'energy': analyzer.state.total_energy
        }
    else:
        analysis_results = analyzer.process(image_rgb)
        analysis_results['paused'] = False

    response_data = {
        analyzer.exercise_id: analysis_results,   
        'exercise': analyzer.exercise_id,         
        'analysis': analysis_results,             
        'gesture_detected': gesture_detected_type,
        'current_gesture': current_gesture,       
        'gesture_confidence': gesture_result['confidence']  
    }

    return jsonify(response_data)


@mediapipe_bp.route('/status', methods=['GET'])
def get_status():
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
    global hf_gesture_recognizer
    
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