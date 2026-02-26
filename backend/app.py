from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from PIL import Image
import io
import base64
from model_utils import RoofPredictor
from maps_utils import get_satellite_image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="../static", static_url_path="")
CORS(app)

# Initialize Predictor
# Note: Using absolute path or relative to project root
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "roof_model.pth")
predictor = RoofPredictor(MODEL_PATH)

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    address = data.get("address")
    
    if not address:
        return jsonify({"error": "No address provided"}), 400
    
    try:
        # 1. Get Satellite Image
        image = get_satellite_image(address)
        
        # 2. Run Inference
        mask, roof_percentage, raw_pred = predictor.predict(image)
        
        # 3. Encode images to Base64 for response
        # Original Image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        # Mask Image
        mask_pil = Image.fromarray(mask)
        mask_byte_arr = io.BytesIO()
        mask_pil.save(mask_byte_arr, format='PNG')
        mask_base64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')
        
        return jsonify({
            "address": address,
            "roof_percentage": round(roof_percentage, 2),
            "image_base64": img_base64,
            "mask_base64": mask_base64,
            "metrics": {
                "estimated_area_sqm": round(roof_percentage * 0.1, 2) # Rough placeholder calculation
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
