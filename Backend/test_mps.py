import torch
import time
import numpy as np
from PIL import Image

def test_mps_acceleration():
    print("=" * 60)
    print("Apple Silicon MPS 加速测试")
    print("=" * 60)
    
    # 1. 检查 MPS 可用性
    print(f"\n1. MPS 可用: {torch.backends.mps.is_available()}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    
    # 2. 选择设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✅ 使用设备: MPS (Apple Silicon 加速)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ 使用设备: CUDA")
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  使用设备: CPU")
    
    # 3. 加载手势识别模型
    print("\n2. 加载手势识别模型...")
    try:
        from transformers import AutoImageProcessor, SiglipForImageClassification
        
        model_name = "prithivMLmods/Hand-Gesture-19"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = SiglipForImageClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        print(f"✅ 模型已加载到 {device}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 4. 性能测试
    print("\n3. 性能测试（推理 100 次）...")
    
    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    # 预热
    inputs = processor(images=test_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        _ = model(**inputs)
    
    # 计时
    start_time = time.time()
    for _ in range(100):
        inputs = processor(images=test_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
    
    elapsed = time.time() - start_time
    fps = 100 / elapsed
    
    print(f"\n4. 性能结果:")
    print(f"   总耗时: {elapsed:.2f} 秒")
    print(f"   平均每帧: {elapsed/100*1000:.2f} 毫秒")
    print(f"   理论 FPS: {fps:.1f}")
    
    if device.type == 'mps':
        print(f"\n✅ MPS 加速正常工作！")
        print(f"   预期性能: 30-60 FPS (M1), 60-120 FPS (M2/M3)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_mps_acceleration()