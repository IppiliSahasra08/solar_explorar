# ============================================================
# Solar Explorar — Configuration
# ============================================================
# Put your Mapbox public token here (starts with pk.eyJ1...)
# Get one free at https://account.mapbox.com/
MAPBOX_TOKEN = "pk.YOUR_MAPBOX_TOKEN_HERE"

# To swap the model: change this path and restart the server.
# Any .pth file trained with the same UNet architecture works.
MODEL_PATH = "roof_segmentation_model.pth"

# Mapbox Static Image settings
SATELLITE_WIDTH  = 600
SATELLITE_HEIGHT = 600
SATELLITE_ZOOM   = 18   # 18 = detailed rooftop level for Mapbox
