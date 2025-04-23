from collections import deque

class GestureDetector:
    def __init__(self, buffer_size=5, threshold=0.8):
        self.buffer = deque(maxlen=buffer_size)  # 滑动窗口
        self.threshold = threshold

    def detect_stable_gesture(self, gesture_function, landmarks):
        gesture_detected = gesture_function(landmarks)
        self.buffer.append(gesture_detected)

        # 如果滑动窗口内所有值都为 True，则认为手势稳定
        return all(self.buffer)

def is_wait_gesture(landmarks):
    wrist = landmarks[0]  # 手腕点
    index_tip = landmarks[8]  # 食指指尖
    index_pip = landmarks[6]  # 食指第二关节

    # 计算手掌宽度（手腕到食指根部）
    palm_width = abs(landmarks[5].x - landmarks[17].x)

    # 归一化坐标
    index_tip_y_norm = (index_tip.y - wrist.y) / palm_width
    index_pip_y_norm = (index_pip.y - wrist.y) / palm_width

    # 检测食指是否竖直
    index_finger_extended = index_tip_y_norm < index_pip_y_norm - 0.2  # 提高阈值，防止误判

    # 检测其他手指是否弯曲
    other_fingers_bent = all(
        landmarks[finger_tip].y > landmarks[finger_pip].y
        for finger_tip, finger_pip in [(12, 10), (16, 14), (20, 18)]  # 中指、无名指、小指
    )

    return index_finger_extended and other_fingers_bent

def is_five_fingers_open(landmarks):
    wrist = landmarks[0]         # 手腕点
    thumb_tip = landmarks[4]     # 大拇指指尖
    index_tip = landmarks[8]     # 食指指尖
    middle_tip = landmarks[12]   # 中指指尖
    ring_tip = landmarks[16]     # 无名指指尖
    pinky_tip = landmarks[20]    # 小指指尖
    
    # 计算手掌宽度
    palm_width = abs(landmarks[5].x - landmarks[17].x)

    # 检测大拇指是否伸展（指尖远离手腕）
    thumb_extended = abs(thumb_tip.x - wrist.x) > 0.2  # 大拇指横向远离手掌中心

    # 检测其他手指是否完全伸展
    index_extended = index_tip.y < landmarks[6].y - 0.05 * palm_width
    middle_extended = middle_tip.y < landmarks[10].y - 0.05 * palm_width
    ring_extended = ring_tip.y < landmarks[14].y - 0.05 * palm_width
    pinky_extended = pinky_tip.y < landmarks[18].y - 0.05 * palm_width

    return thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended