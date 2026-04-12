from flask import Blueprint, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import time

from .analyzer import ExerciseAnalyzer 
from .gesture_recognizer import get_gesture_recognizer 

analyzer = ExerciseAnalyzer() 

mediapipe_bp = Blueprint('mediapipe', __name__)
CORS(mediapipe_bp)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

hf_gesture_recognizer = None

# Track session timing
session_start_time = None


@mediapipe_bp.route('/control', methods=['POST'])
def control_workout():
    global session_start_time
    
    data = request.json
    action = data.get('action')

    if action == 'start':
        exercise_id = data.get('exercise')
        style = data.get('style', 'intermediate')
        target_count = data.get('target_count')
        task_id = data.get('task_id')
        
        if not exercise_id:
            return jsonify({"error": "Exercise type is required"}), 400
            
        try:
            analyzer.setup(exercise_id, style, target_count, task_id)
            session_start_time = time.time()
            
            return jsonify({
                "status": "started", 
                "exercise": exercise_id,
                "style": style,
                "target_count": analyzer.target_count,
                "task_id": task_id,
                "paused": True,
                "message": "請做 👍 手勢開始運動"
            })
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    elif action == 'reset':
        analyzer.reset()
        session_start_time = None
        print("Analyzer reset.")
        return jsonify({
            "status": "reset",
            "paused": True,
            "message": "請做 👍 手勢開始運動"
        })
    
    elif action == 'stop':
        # ── 改動：取得 LSTM 最終評分 ──
        duration = 0
        if session_start_time:
            duration = round(time.time() - session_start_time, 1)
        
        # 先拿 LSTM 分數（在 reset 之前）
        lstm_results = analyzer.get_final_score()
        
        results = {
            'task_id': analyzer.current_task_id,
            'exercise_key': analyzer.exercise_id,
            'exercise_name': analyzer.config['name'] if analyzer.config else None,
            'completed_reps': lstm_results['rep_count'],
            'target_reps': analyzer.target_count,
            'smoothness': lstm_results['smoothness_score'],
            'total_energy': lstm_results['total_energy'],
            'duration': duration,
            'is_complete': lstm_results['rep_count'] >= analyzer.target_count,
            'style': analyzer.style,
            
            # ── LSTM 新增欄位 ──
            'lstm_score': lstm_results['final_lstm_score'],
            'lstm_avg': lstm_results['avg_lstm_score'],
            'lstm_min': lstm_results['min_lstm_score'],
            'lstm_max': lstm_results['max_lstm_score'],
            'lstm_frames_scored': lstm_results['total_frames_scored'],
            'scores_over_time': lstm_results['scores_over_time'],
        }
        
        # get_final_score() 內部已經 reset，這裡再確保
        session_start_time = None
        
        return jsonify({
            "status": "stopped",
            "results": results
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
            'energy': analyzer.state.total_energy,
            'target_count': analyzer.target_count,
            'task_id': analyzer.current_task_id,
            'lstm_score': None,  # ← 新增：暫停時沒有分數
        }
    else:
        analysis_results = analyzer.process(image_rgb)
        # process() 回傳的 dict 已經包含 lstm_score
        analysis_results['paused'] = False
        analysis_results['target_count'] = analyzer.target_count
        analysis_results['task_id'] = analyzer.current_task_id
        
        analysis_results['target_reached'] = analyzer.state.count >= analyzer.target_count

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
        "exercise_key": analyzer.exercise_id,
        "style": analyzer.style,
        "count": analyzer.state.count,
        "target_count": analyzer.target_count,
        "stage": analyzer.state.stage,
        "is_paused": analyzer.state.is_paused,
        "current_feedback": analyzer.state.feedback,
        "total_energy": analyzer.state.total_energy,
        "task_id": analyzer.current_task_id,
        "smoothness": analyzer.smoothness_score,
        "lstm_score": analyzer.state.current_lstm_score,  # ← 新增
    })


@mediapipe_bp.route('/session-results', methods=['GET'])
def session_results():
    """Get current session results without stopping"""
    if not analyzer.config:
        return jsonify({"error": "No active session"}), 400
    
    return jsonify(get_session_results())


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


def get_session_results():
    """Helper function to build session results (不 reset，只讀取)"""
    global session_start_time
    
    duration = 0
    if session_start_time:
        duration = round(time.time() - session_start_time, 1)
    
    # ── 計算當前 LSTM 分數快照 ──
    scores = analyzer.state.lstm_scores
    if scores:
        arr = np.array(scores, dtype=np.float32)
        n = len(arr)
        if n >= 10:
            trim = max(1, n // 10)
            trimmed = np.sort(arr)[trim:-trim]
            lstm_final = int(round(float(np.mean(trimmed))))
        else:
            lstm_final = int(round(float(np.mean(arr))))
    else:
        lstm_final = None
    
    return {
        'task_id': analyzer.current_task_id,
        'exercise_key': analyzer.exercise_id,
        'exercise_name': analyzer.config['name'] if analyzer.config else None,
        'completed_reps': analyzer.state.count,
        'target_reps': analyzer.target_count,
        'smoothness': analyzer.smoothness_score,
        'total_energy': round(analyzer.state.total_energy, 2),
        'duration': duration,
        'is_complete': analyzer.state.count >= analyzer.target_count,
        'style': analyzer.style,
        'lstm_score': lstm_final,                    # ← 新增
        'lstm_realtime': analyzer.state.current_lstm_score,  # ← 新增
    }