from flask import Blueprint, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import tempfile
import os

# 创建蓝图
mediapipe_bp = Blueprint('mediapipe', __name__)

# 初始化MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 定义常量
CHEST_PULL_THRESHOLD_X = 0.05  # 手腕或手肘抬起高度达到肩膀的阈值（完成动作）
OVER_EXTENSION_THRESHOLD_Y = -0.2  # 手腕或手肘高度超过肩膀的阈值（检测过伸）
OVER_EXTENSION_THRESHOLD_X = 0.15  # 手腕或手肘水平偏移肩膀的阈值（检测水平过伸）
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

# 创建全局状态实例
workout_state = WorkoutState()

# 手势检测函数
def is_wait_gesture(landmarks):
    """检测等待手势（竖起食指）"""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    index_finger_extended = index_tip.y < landmarks[6].y
    
    if index_finger_extended:
        return True
    return False

def is_thumbs_up(landmarks):
    """检测竖大拇指手势"""
    thumb_tip = landmarks[4]  # 大拇指指尖
    thumb_ip = landmarks[3]   # 大拇指第二关节
    index_tip = landmarks[8]  # 食指指尖
    index_pip = landmarks[6]  # 食指第二关节
    middle_tip = landmarks[12]  # 中指指尖
    middle_pip = landmarks[10]  # 中指第二关节
    ring_tip = landmarks[16]  # 无名指指尖
    ring_pip = landmarks[14]  # 无名指第二关节
    pinky_tip = landmarks[20]  # 小指指尖
    pinky_pip = landmarks[18]  # 小指第二关节

    # 判断大拇指是否伸展（指尖高于第二关节）
    thumb_extended = thumb_tip.y < thumb_ip.y

    # 判断其他手指是否弯曲（指尖低于第二关节）
    index_finger_bent = index_tip.y > index_pip.y
    middle_finger_bent = middle_tip.y > middle_pip.y
    ring_finger_bent = ring_tip.y > ring_pip.y
    pinky_finger_bent = pinky_tip.y > pinky_pip.y

    # 如果大拇指伸展且其他手指弯曲，则认为是"大拇指"手势
    if (thumb_extended and
        index_finger_bent and
        middle_finger_bent and
        ring_finger_bent and
        pinky_finger_bent):
        return True
    return False

# 动作分析函数
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
                distance = math.sqrt((workout_state.end_position[0] - workout_state.start_position[0])**2 +
                                    (workout_state.end_position[1] - workout_state.start_position[1])**2)
                workout_state.total_distance += distance
                
                # 计算能量消耗
                energy = BAND_RESISTANCE_N * distance
                workout_state.total_energy += energy
                
                workout_state.lateral_rise_count += 1

# API端点
@mediapipe_bp.route('/analyze', methods=['POST'])
def analyze_frame():
    """分析上传的单张图片"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 保存临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_file.name)
    temp_file.close()
    
    # 读取图像
    frame = cv2.imread(temp_file.name)
    os.unlink(temp_file.name)  # 删除临时文件
    
    if frame is None:
        return jsonify({'error': 'Invalid image file'}), 400
    
    # 转换为RGB图像并处理
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
        'count': workout_state.lateral_rise_count,
        'distance': workout_state.total_distance,
        'energy': workout_state.total_energy,
        'overextension': workout_state.overextension_detected,
        'paused': workout_state.is_paused
    })

@mediapipe_bp.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    """分析视频流中的一帧"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
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
        'count': workout_state.lateral_rise_count,
        'distance': workout_state.total_distance,
        'energy': workout_state.total_energy,
        'overextension': workout_state.overextension_detected,
        'paused': workout_state.is_paused,
        'landmarks': None,
        'gesture_detected': None
    }
    
    # 转换为RGB图像并处理
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 分析手势
    hands_results = hands.process(image_rgb)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # 手势检测
            if is_wait_gesture(landmarks):
                workout_state.is_paused = True
                response_data['gesture_detected'] = 'wait'
            elif is_thumbs_up(landmarks):
                workout_state.is_paused = False
                response_data['gesture_detected'] = 'thumbs_up'
    
    # 分析姿势
    pose_results = pose.process(image_rgb)
    if not workout_state.is_paused and pose_results.pose_landmarks:
        # 提取关键点
        landmarks = pose_results.pose_landmarks.landmark
        
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
    
    # 更新响应中的当前状态
    response_data.update({
        'count': workout_state.lateral_rise_count,
        'distance': workout_state.total_distance,
        'energy': workout_state.total_energy,
        'overextension': workout_state.overextension_detected,
        'paused': workout_state.is_paused
    })
    
    return jsonify(response_data)

@mediapipe_bp.route('/control', methods=['POST'])
def control_workout():
    """控制训练状态（暂停/继续/重置）"""
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

@mediapipe_bp.route('/status', methods=['GET'])
def get_status():
    """获取当前训练状态"""
    return jsonify({
        'count': workout_state.lateral_rise_count,
        'distance': workout_state.total_distance,
        'energy': workout_state.total_energy,
        'overextension': workout_state.overextension_detected,
        'paused': workout_state.is_paused
    })