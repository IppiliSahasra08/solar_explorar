import os
import io
import base64

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import requests
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# --------------------------
# CONFIGURATION
# --------------------------
try:
    from config import MAPBOX_TOKEN, MODEL_PATH, SATELLITE_WIDTH, SATELLITE_HEIGHT, SATELLITE_ZOOM
except ImportError:
    MAPBOX_TOKEN     = os.environ.get("MAPBOX_TOKEN", "")
    MODEL_PATH       = os.environ.get("MODEL_PATH", "roof_model.pth")
    SATELLITE_WIDTH  = 600
    SATELLITE_HEIGHT = 600
    SATELLITE_ZOOM   = 18

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
        self.down1  = DoubleConv(3, 64)
        self.pool1  = nn.MaxPool2d(2)
        self.down2  = DoubleConv(64, 128)
        self.pool2  = nn.MaxPool2d(2)
        self.down3  = DoubleConv(128, 256)
        self.pool3  = nn.MaxPool2d(2)
        self.middle = DoubleConv(256, 512)
        self.up3    = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3  = DoubleConv(512, 256)
        self.up2    = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2  = DoubleConv(256, 128)
        self.up1    = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1  = DoubleConv(128, 64)
        self.final  = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m  = self.middle(self.pool3(d3))
        u3 = self.conv3(torch.cat([self.up3(m), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))
        return self.final(u1)


# --------------------------
# MODEL LOADER (swappable)
# --------------------------
def load_model(path: str = MODEL_PATH) -> UNet:
    """Load (or reload) the UNet from the given .pth file."""
    m = UNet().to(DEVICE)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        print(f"[OK] Model loaded from {path}")
    else:
        print(f"[WARN] Warning: {path} not found. Segmentation will return a blank mask.")
    return m


model = load_model()


# --------------------------
# HELPERS
# --------------------------
TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])


def calculate_solar_potential(roof_area_m2: float) -> dict:
    """
    Calculate solar potential metrics based on roof area.
    """
    # Assumptions based on the project breakdown:
    # 75% of roof area is usable (obstructions, shading, etc.)
    usable_area = roof_area_m2 * 0.75
    
    # Standard panel is ~1.7 m2
    panel_count = int(usable_area / 1.7)
    
    # Each panel is ~400W (0.4kW)
    system_capacity_kw = panel_count * 0.4
    
    # Annual energy: Capacity * Peak Sun Hours (4.5 default) * Days
    annual_energy_kwh = system_capacity_kw * 4.5 * 365
    
    # Annual savings: Energy * Local rate ($0.15/kWh default)
    annual_savings = annual_energy_kwh * 0.15
    
    # CO2 offset: Energy * Grid emission factor (0.4 kg/kWh average)
    co2_offset_kg = annual_energy_kwh * 0.4
    
    return {
        "roof_area_m2": round(float(roof_area_m2), 2),
        "panel_count": panel_count,
        "capacity_kw": round(float(system_capacity_kw), 2),
        "annual_energy_kwh": round(float(annual_energy_kwh), 2),
        "annual_savings": round(float(annual_energy_kwh * 0.15), 2),
        "co2_offset_kg": round(float(annual_energy_kwh * 0.4), 2)
    }


def coverage_to_label(pct: float) -> dict:
    """
    Convert roof coverage percentage to a human-readable prediction.
    Returns a dict with 'label', 'emoji', and 'description'.
    """
    if pct < 10:
        return {
            "label":       "Low Solar Potential",
            "emoji":       "⚠️",
            "description": "Very little rooftop area detected. Solar installation may not be viable.",
        }
    elif pct < 30:
        return {
            "label":       "Moderate Solar Potential",
            "emoji":       "🌤️",
            "description": "Some rooftop area detected. A small panel array could be installed.",
        }
    elif pct < 60:
        return {
            "label":       "Good Solar Potential",
            "emoji":       "☀️",
            "description": "Good rooftop coverage detected. Suitable for a standard solar installation.",
        }
    else:
        return {
            "label":       "Excellent Solar Potential",
            "emoji":       "🌟",
            "description": "Large rooftop area detected. Ideal candidate for maximised solar generation.",
        }


def predict_mask(image: Image.Image):
    """Run U-Net inference. Returns (mask PIL, overlay PIL, coverage %, area m2)."""
    # 1. Preprocess & Inference
    inp = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_tensor = torch.sigmoid(model(inp)).squeeze().cpu()
        pred_np = pred_tensor.numpy()
    
    # 2. Basic Mask (for area/coverage)
    mask_arr = (pred_np * 255).astype(np.uint8)
    coverage_pct = float(np.mean(pred_np)) * 100
    
    # Scale calculation for area
    white_pixels = np.sum(pred_np > 0.5)
    pixel_to_m2 = 0.36 * ((SATELLITE_WIDTH / 256) ** 2)
    area_m2 = float(white_pixels * pixel_to_m2)

    # 3. Create Analysis Overlay (Contours on original)
    # Convert PIL image to OpenCV format
    orig_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = orig_cv.shape[:2]
    
    # Resize prediction back to original image size for overlay
    pred_resized = cv2.resize(pred_np, (w, h))
    binary_mask = (pred_resized > 0.5).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create semi-transparent overlay
    overlay = orig_cv.copy()
    # Fill detected regions with semi-transparent Cyan (B=255, G=182, R=6)
    cv2.fillPoly(overlay, contours, (255, 182, 6))
    
    # Blend overlay with original (alpha=0.3)
    alpha = 0.3
    combined = cv2.addWeighted(overlay, alpha, orig_cv, 1 - alpha, 0)
    
    # Draw sharp borders (Cyan)
    cv2.drawContours(combined, contours, -1, (255, 182, 6), 2)
    
    # Convert back to PIL
    overlay_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask_arr)

    return mask_pil, overlay_pil, round(float(coverage_pct), 2), area_m2


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def fetch_satellite_image(lat: float, lng: float, zoom: int = SATELLITE_ZOOM) -> Image.Image:
    """
    Fetch a satellite image from the Mapbox Static Images API.
    """
    if not MAPBOX_TOKEN or MAPBOX_TOKEN.startswith("pk.YOUR"):
        raise HTTPException(
            status_code=500,
            detail="Mapbox token not configured. Please set MAPBOX_TOKEN in config.py.",
        )

    # Mapbox Static API URL
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lng},{lat},{zoom}/{SATELLITE_WIDTH}x{SATELLITE_HEIGHT}"
        f"?access_token={MAPBOX_TOKEN}"
    )
    
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Mapbox satellite request failed: HTTP {resp.status_code}",
        )
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# --------------------------
# ENDPOINTS
# --------------------------
@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path":   MODEL_PATH,
        "mapbox_ready": bool(MAPBOX_TOKEN and not MAPBOX_TOKEN.startswith("pk.YOUR")),
    }


@app.get("/geocode")
async def geocode(address: str = Query(..., description="Address or place to look up")):
    """
    Convert a free-text address into latitude / longitude using Mapbox Geocoding API.
    Returns: { lat, lng, place_name }
    """
    if not MAPBOX_TOKEN or MAPBOX_TOKEN.startswith("pk.YOUR"):
        raise HTTPException(
            status_code=500,
            detail="Mapbox token not configured. Please set MAPBOX_TOKEN in config.py.",
        )

    import urllib.parse
    encoded = urllib.parse.quote(address)
    url = (
        f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded}.json"
        f"?access_token={MAPBOX_TOKEN}&limit=1"
    )
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Geocoding request failed.")

    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise HTTPException(status_code=404, detail="Address not found. Try a more specific query.")

    feat       = features[0]
    lng, lat   = feat["center"]
    place_name = feat.get("place_name", address)

    return {"lat": lat, "lng": lng, "place_name": place_name}


@app.get("/satellite")
async def satellite_preview(
    lat:  float = Query(...),
    lng:  float = Query(...),
    zoom: int   = Query(default=SATELLITE_ZOOM),
):
    """
    Fetch a satellite image for the given coordinates.
    Returns: { image_b64 }  (data URI ready)
    """
    img = fetch_satellite_image(lat, lng, zoom)
    return {
        "image_b64": f"data:image/png;base64,{pil_to_b64(img)}",
        "width":  SATELLITE_WIDTH,
        "height": SATELLITE_HEIGHT,
        "zoom":   zoom,
    }


@app.get("/segment-from-coords")
async def segment_from_coords(
    lat:  float = Query(...),
    lng:  float = Query(...),
    zoom: int   = Query(default=SATELLITE_ZOOM),
):
    """
    Full pipeline: fetch satellite image → run U-Net segmentation → return results.
    """
    # 1. Fetch satellite image
    try:
        sat_img = fetch_satellite_image(lat, lng, zoom)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Satellite fetch failed: {e}")

    # 2. Run segmentation
    try:
        mask_pil, overlay_pil, coverage, area_m2 = predict_mask(sat_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    # 3. Build solar metrics
    metrics = calculate_solar_potential(area_m2)
    prediction = coverage_to_label(coverage)

    return {
        "coverage":        coverage,
        "prediction":      prediction,
        "metrics":         metrics,
        "satellite_image": f"data:image/png;base64,{pil_to_b64(sat_img)}",
        "mask_image":      f"data:image/png;base64,{pil_to_b64(overlay_pil)}",
    }


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Legacy endpoint: upload an image file directly → segmentation results.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    try:
        mask_pil, overlay_pil, coverage, area_m2 = predict_mask(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    metrics    = calculate_solar_potential(area_m2)
    prediction = coverage_to_label(coverage)
    thumb      = image.resize((256, 256))

    return {
        "coverage":       coverage,
        "prediction":     prediction,
        "metrics":        metrics,
        "original_image": f"data:image/png;base64,{pil_to_b64(thumb)}",
        "mask_image":     f"data:image/png;base64,{pil_to_b64(overlay_pil)}",
    }


@app.post("/reload-model")
async def reload_model_endpoint(path: str = Query(default=MODEL_PATH)):
    """
    Hot-swap the model without restarting the server.
    POST /reload-model?path=new_model.pth
    """
    global model
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {path}")
    try:
        model = load_model(path)
        return {"status": "ok", "loaded": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
