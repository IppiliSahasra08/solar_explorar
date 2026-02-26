import sys
import os
import torch
from PIL import Image
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from model_utils import RoofPredictor

def test_inference():
    print("Testing model inference...")
    model_path = "roof_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    predictor = RoofPredictor(model_path)
    
    # Create a dummy image (RGB, 640x640)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
    
    try:
        mask, percentage, pred = predictor.predict(dummy_img)
        print(f"Success! Roof Percentage: {percentage:.2f}%")
        print(f"Mask shape: {mask.shape}")
        
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    test_inference()
