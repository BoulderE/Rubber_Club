import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
from collections import deque

class GestureRecognizer:
    """
    使用 HuggingFace 的 SigLIP 模型识别手势 (支持 Apple Silicon MPS 加速)
    """
    def __init__(self, model_name="prithivMLmods/Hand-Gesture-19", buffer_size=5):
        print(f"正在加载手势识别模型: {model_name}")
        
        # ========== Apple Silicon MPS 加速 ==========
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ 使用 Apple Silicon MPS 加速")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✅ 使用 CUDA GPU 加速")
        else:
            self.device = torch.device("cpu")
            print("⚠️  使用 CPU")
        
        # 加载模型
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # 移动到设备
        self.model.to(self.device)
        self.model.eval()
        
        # 手势标签映射（19类）
        self.labels = {
            0: "call", 1: "dislike", 2: "fist", 3: "four",
            4: "like", 5: "mute", 6: "no_gesture", 7: "ok",
            8: "one", 9: "palm", 10: "peace", 11: "peace_inverted",
            12: "rock", 13: "stop", 14: "stop_inverted",
            15: "three", 16: "three2", 17: "two_up", 18: "two_up_inverted"
        }
        
        # 稳定性缓冲区
        self.buffer_size = buffer_size
        self.gesture_buffer = deque(maxlen=buffer_size)
        
        print(f"手势识别模型加载完成！设备: {self.device}")
    
    def predict(self, image_np, confidence_threshold=0.7):
        """预测单帧图像的手势"""
        # 转换为 PIL Image
        image = Image.fromarray(image_np).convert("RGB")
        
        # 预处理
        inputs = self.processor(images=image, return_tensors="pt")
        
        # ========== 将输入移到设备 ==========
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
        
        # 移回 CPU
        probs = probs.cpu()
        
        # 获取最高分
        max_prob, max_idx = torch.max(probs, dim=0)
        gesture = self.labels[max_idx.item()]
        confidence = max_prob.item()
        
        # 低置信度返回 no_gesture
        if confidence < confidence_threshold:
            gesture = "no_gesture"
        
        return {
            'gesture': gesture,
            'confidence': confidence,
            'all_scores': {self.labels[i]: probs[i].item() for i in range(len(self.labels))}
        }
    
    def detect_stable_gesture(self, image_np, target_gesture, confidence_threshold=0.7):
        """检测稳定的手势（需要连续多帧识别一致）"""
        result = self.predict(image_np, confidence_threshold)
        detected_gesture = result['gesture']
        
        # 加入缓冲区
        self.gesture_buffer.append(detected_gesture)
        
        # 检查缓冲区是否全部为目标手势
        if len(self.gesture_buffer) == self.buffer_size:
            if all(g == target_gesture for g in self.gesture_buffer):
                return True
        
        return False
    
    def reset_buffer(self):
        """重置缓冲区"""
        self.gesture_buffer.clear()
    
    def get_device_info(self):
        """获取设备信息"""
        info = {
            'device': str(self.device),
            'mps_available': torch.backends.mps.is_available(),
            'cuda_available': torch.cuda.is_available()
        }
        
        if self.device.type == 'mps':
            info['acceleration'] = 'Apple Silicon MPS'
        elif self.device.type == 'cuda':
            info['acceleration'] = f'CUDA GPU ({torch.cuda.get_device_name(0)})'
        else:
            info['acceleration'] = 'CPU (No Hardware Acceleration)'
        
        return info


# 全局实例（懒加载）
_gesture_recognizer = None

def get_gesture_recognizer():
    """获取手势识别器实例（单例模式）"""
    global _gesture_recognizer
    if _gesture_recognizer is None:
        _gesture_recognizer = GestureRecognizer()
    return _gesture_recognizer