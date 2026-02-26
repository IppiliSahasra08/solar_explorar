import requests
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def get_satellite_image(address, zoom=20, size="640x640"):
    """
    Fetches a satellite image for a given address using Google Static Maps API.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

    # 1. Geocoding (Address to Lat/Lng)
    # Note: For simplicity, we can also just pass the address string directly to Static Maps,
    # but Geocoding is safer for complex addresses.
    
    static_maps_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": address,
        "zoom": zoom,
        "size": size,
        "maptype": "satellite",
        "key": GOOGLE_API_KEY
    }
    
    response = requests.get(static_maps_url, params=params)
    
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    else:
        raise Exception(f"Failed to fetch image from Google Maps: {response.status_code} - {response.text}")
