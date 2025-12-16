# diagnose.py
import torch
import os
from ultralytics import YOLO
import cv2
import numpy as np

print("=== Model Diagnostics ===")

# 1. Check if model file exists
model_path = r"C:\Users\dell\OneDrive\Desktop\Food_AI_Project\best.pt"
print(f"1. Model path exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print(f"   File size: {os.path.getsize(model_path) / 1e6:.1f} MB")
    
    # 2. Try to load the model
    try:
        print("\n2. Loading model...")
        model = YOLO(model_path)
        print(f"   ✅ Model loaded successfully!")
        
        # 3. Check model info
        print(f"   Model type: {type(model)}")
        print(f"   Model device: {model.device}")
        
        if hasattr(model, 'names'):
            print(f"   Classes: {model.names}")
            print(f"   Number of classes: {len(model.names)}")
        else:
            print("   ⚠️ No class names found in model")
        
        # 4. Test inference
        print("\n3. Testing inference...")
        test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        # Test with different confidences
        for conf in [0.9, 0.5, 0.3, 0.1]:
            results = model(test_img, conf=conf, verbose=False)
            boxes = results[0].boxes if results[0].boxes is not None else []
            print(f"   Confidence {conf}: {len(boxes)} detections")
        
        # 5. Test with a simple image
        print("\n4. Testing with colored shapes...")
        simple_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(simple_img, (50, 50), (150, 200), (139, 69, 19), -1)  # Brown
        cv2.circle(simple_img, (250, 150), 50, (30, 136, 229), -1)  # Blue
        
        results = model(simple_img, conf=0.3, verbose=True)
        print(f"   Detections on simple image: {len(results[0].boxes) if results[0].boxes else 0}")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Model file not found!")
    
# Check CUDA
print("\n=== System Info ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Check ultralytics
print(f"\nUltralytics import: Success")