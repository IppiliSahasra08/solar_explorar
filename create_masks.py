import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw

RAW_FOLDER = "raw_images"
MASK_FOLDER = "masks"

os.makedirs(MASK_FOLDER, exist_ok=True)

for file in os.listdir(RAW_FOLDER):
    if file.endswith(".json"):
        json_path = os.path.join(RAW_FOLDER, file)

        with open(json_path) as f:
            data = json.load(f)

        image_path = os.path.join(RAW_FOLDER, os.path.basename(data["imagePath"]))
        image = Image.open(image_path)
        width, height = image.size

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for shape in data["shapes"]:
            if shape["label"] == "roof":
                points = [tuple(point) for point in shape["points"]]
                draw.polygon(points, outline=255, fill=255)

        mask_name = file.replace(".json", ".png")
        mask.save(os.path.join(MASK_FOLDER, mask_name))

print("✅ Masks created successfully!")