from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import sys
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.inference_pipeline import process_image  
app = Flask(__name__)
CORS(app)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'img_folder')
ANNOTATED_FOLDER = os.path.join(PROJECT_ROOT, 'output_folder')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return "✅ Captioning backend is running."

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(image.filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(img_path)

    try:
        result = process_image(filename)
    except Exception as e:
        print("❌ Exception occurred during image processing:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

@app.route("/annotated/<filename>", methods=["GET"])
def get_annotated(filename):
    annotated_path = os.path.join(ANNOTATED_FOLDER, filename)
    if not os.path.exists(annotated_path):
        return jsonify({"error": "Annotated image not found"}), 404
    return send_file(annotated_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
