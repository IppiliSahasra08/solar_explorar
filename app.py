import os
import io
import base64
import math

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --------------------------
# CONFIGURATION
# --------------------------
MODEL_PATH = "roof_model.pth"
DEVICE = torch.device("cpu")

app = FastAPI(title="Solar Explorar — Roof Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# MODEL DEFINITION
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
        m  = self.middle(self.pool3(d3))
        u3 = self.conv3(torch.cat([self.up3(m), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))
        return self.final(u1)


# Load model once at startup
model = UNet().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️  Warning: {MODEL_PATH} not found. Segmentation will return blank mask.")

# --------------------------
# HELPERS
# --------------------------
TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

def predict_mask(image: Image.Image):
    """Run U-Net inference. Returns (mask PIL image, roof coverage 0-1)."""
    inp = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
    mask_arr = (pred * 255).astype(np.uint8)
    return Image.fromarray(mask_arr), float(np.mean(pred))

def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --------------------------
# ENDPOINTS
# --------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    try:
        mask_pil, coverage = predict_mask(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Resize original to 256x256 for consistent display
    thumb = image.resize((256, 256))

    return {
        "coverage": round(coverage * 100, 2),           # % of pixels detected as roof
        "original_image": f"data:image/png;base64,{pil_to_base64(thumb)}",
        "mask_image":     f"data:image/png;base64,{pil_to_base64(mask_pil)}",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
