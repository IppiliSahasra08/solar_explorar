import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

# --------------------------
# DoubleConv Block
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

# --------------------------
# UNet Model
# --------------------------
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

class RoofPredictor:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def predict(self, image_pil):
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.sigmoid(output)
            pred = pred.squeeze().cpu().numpy()
        
        # Calculate roof percentage
        # Using a threshold of 0.5 to define roof pixels
        threshold = 0.5 if pred.max() > 0.5 else (pred.max() * 0.8)
        roof_mask = (pred > threshold).astype(np.uint8) * 255
        
        roof_pixel_count = np.sum(pred > threshold)
        total_pixels = pred.size
        roof_percentage = (roof_pixel_count / total_pixels) * 100
        
        return roof_mask, roof_percentage, pred
