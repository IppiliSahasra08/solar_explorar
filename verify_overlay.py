import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2

# --------------------------
# Model Definition
# --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.middle = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m = self.middle(self.pool3(d3))
        u3 = self.up3(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        return self.final(u1)

# --------------------------
# Verification Logic
# --------------------------
def verify_overlay(img_path, model_path="roof_model.pth", out_path="improved_overlay.png"):
    device = torch.device("cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    image = Image.open(img_path).convert("RGB")
    TRANSFORM = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    inp = TRANSFORM(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_tensor = torch.sigmoid(model(inp)).squeeze().cpu()
        pred_np = pred_tensor.numpy()
    
    # Create Analysis Overlay
    orig_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = orig_cv.shape[:2]
    
    pred_resized = cv2.resize(pred_np, (w, h))
    binary_mask = (pred_resized > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = orig_cv.copy()
    cv2.fillPoly(overlay, contours, (255, 182, 6)) # BGR Cyan
    
    alpha = 0.3
    combined = cv2.addWeighted(overlay, alpha, orig_cv, 1 - alpha, 0)
    cv2.drawContours(combined, contours, -1, (255, 182, 6), 2)
    
    cv2.imwrite(out_path, combined)
    print(f"Overlay saved to {out_path}")

if __name__ == "__main__":
    verify_overlay("images/img_004.jpg")
