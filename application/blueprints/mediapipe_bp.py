from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

mediapipe_bp = Blueprint('mediapipe', __name__)

@mediapipe_bp.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # MediaPipe 姿势检测
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return jsonify({
        'landmarks': results.pose_landmarks.ListFields()[0][1] if results.pose_landmarks else None
    })