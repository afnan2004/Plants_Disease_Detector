import os
import sys
import urllib.request

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

MODEL_PATH = "deepfake_model.h5"
MODEL_URL = os.getenv("HF_MODEL_URL")

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded")

# Load model
model = load_model(MODEL_PATH)
print("Model loaded successfully")

def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="No file selected")

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        input_data = preprocess_image(filepath)
        prediction = model.predict(input_data)[0][0]

        label = "REAL" if prediction >= 0.5 else "FAKE"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        return render_template(
            "index.html",
            filename=filename,
            label=label,
            confidence=f"{confidence * 100:.2f}%"
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run()
