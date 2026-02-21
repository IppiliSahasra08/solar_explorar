import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import time

# --------------------------
# DEVICE
# --------------------------
device = torch.device("cpu")

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


# --------------------------
# LOAD MODEL
# --------------------------
model = UNet().to(device)
model.load_state_dict(torch.load("roof_model.pth", map_location=device))
model.eval()

# --------------------------
# LOAD IMAGE
# --------------------------
img_path = "images/img_004.jpg"
mask_path = "masks/img_004.png"

image = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path)

print("Original image size:", image.size)

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

input_tensor = transform(image).unsqueeze(0).to(device)

print("Input tensor shape:", input_tensor.shape)

# --------------------------
# PREDICT
# --------------------------
start = time.time()

with torch.no_grad():
    output = model(input_tensor)
    pred = torch.sigmoid(output)
    pred = pred.squeeze().cpu().numpy()

end = time.time()
print("Inference time:", end - start, "seconds")
print(f"Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")

# --------------------------
# VISUALIZE
# --------------------------
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Ground Truth")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Raw Sigmoid (Heatmap)")
plt.imshow(pred, cmap="inferno")
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis("off")

plt.subplot(1, 4, 4)
# Use a dynamic threshold if the max confidence is low
threshold = 0.5 if pred.max() > 0.5 else (pred.max() * 0.8)
plt.title(f"Prediction (T={threshold:.2f})")
plt.imshow(pred > threshold, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()