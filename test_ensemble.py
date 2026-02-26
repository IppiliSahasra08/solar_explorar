import torch
from ensemble_model import EnsemblePredictor
from PIL import Image
import os
import numpy as np

def test_ensemble():
    # Model paths
    ensemble_paths = [
        "best_model_fold1.pth",
        "best_model_fold2.pth",
        "best_model_fold3.pth",
        "best_model_fold4.pth",
        "best_model_fold5.pth",
    ]
    
    # Check if files exist
    missing = [p for p in ensemble_paths if not os.path.exists(p)]
    if missing:
        print(f"Warning: Missing files: {missing}")
    
    # Initialize predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = EnsemblePredictor(ensemble_paths, device=device)
    
    # Load a test image
    test_img_path = "testing/test1.jpg"
    if not os.path.exists(test_img_path):
        print(f"Error: {test_img_path} not found")
        return
        
    img = Image.open(test_img_path).convert("RGB")
    
    # Run prediction
    print(f"Running prediction on {test_img_path}...")
    mask, coverage, avg_pred = predictor.predict(img)
    
    if mask is not None:
        print(f"Success! Coverage: {coverage:.2f}%")
        print(f"Mask shape: {mask.shape}")
        
        # Save output mask for visual check if needed
        output_path = "testing/ensemble_test_output.png"
        Image.fromarray(mask).save(output_path)
        print(f"Mask saved to {output_path}")
    else:
        print("Prediction failed!")

if __name__ == "__main__":
    test_ensemble()
