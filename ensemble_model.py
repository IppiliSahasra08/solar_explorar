import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import torchvision.transforms as T
from PIL import Image

class EnsemblePredictor:
    def __init__(self, model_paths, device="cpu"):
        self.device = torch.device(device)
        self.models = []
        
        print(f"Loading ensemble of {len(model_paths)} models...")
        for path in model_paths:
            if os.path.exists(path):
                model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                ).to(self.device)
                
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.eval()
                self.models.append(model)
                print(f"  [OK] Model loaded from {path}")
            else:
                print(f"  [WARN] Model file not found: {path}")

        if not self.models:
            print("[ERROR] No models loaded for ensemble!")

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def predict(self, image_pil):
        """Run ensemble inference by averaging sigmoid outputs."""
        if not self.models:
            return None, 0.0, None

        inp = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                output = model(inp)
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                all_preds.append(pred)
        
        # Average the predictions
        avg_pred = np.mean(all_preds, axis=0)
        
        # Generate mask
        mask_arr = (avg_pred * 255).astype(np.uint8)
        
        # Coverage percentage
        coverage_pct = float(np.mean(avg_pred)) * 100
        
        return mask_arr, coverage_pct, avg_pred
