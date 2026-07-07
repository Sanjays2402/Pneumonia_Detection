"""Flask web app for pneumonia detection from chest X-ray images.

Loads a trained Keras CNN (``cnn_model.h5``) and serves a simple upload-and-predict
UI. The model is loaded lazily on first use so the app can boot (and be smoke-tested)
without the weights present.

NOTE: This is an educational project, not a medical device. See DISCLAIMER.md.
"""

import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "cnn_model.h5"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
IMG_SIZE = (75, 75)  # must match the training input shape
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap

# Lazily-initialised model handle (populated on first prediction).
_model = None


def get_model():
    """Load and cache the Keras model on first use."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH!r}. Train it with "
                "CNN_model.ipynb (saves cnn_model.h5) or set MODEL_PATH."
            )
        _model = load_model(MODEL_PATH)
    return _model


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def predict_xray(filepath: str) -> dict:
    """Run the CNN on a single image and return label + confidence."""
    img = keras_image.load_img(filepath, target_size=IMG_SIZE, color_mode="grayscale")
    x = keras_image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Sigmoid output in [0, 1]; class 1 == pneumonia (see class_indices in notebook).
    prob = float(get_model().predict(x, verbose=0)[0][0])
    is_pneumonia = prob >= 0.5
    confidence = prob if is_pneumonia else 1.0 - prob
    return {
        "label": "Pneumonia detected" if is_pneumonia else "Normal (no pneumonia)",
        "pneumonia": is_pneumonia,
        "confidence": round(confidence * 100, 2),
    }


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Liveness probe — reports whether the model file is available."""
    return jsonify(status="ok", model_present=os.path.exists(MODEL_PATH))


@app.route("/predict", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(error="No image field in request."), 400

    f = request.files["image"]
    if not f or f.filename == "":
        return jsonify(error="No file selected."), 400
    if not _allowed(f.filename):
        return jsonify(error="Unsupported file type. Use PNG or JPG."), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
    f.save(filepath)

    try:
        result = predict_xray(filepath)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the UI
        return jsonify(error=f"Prediction failed: {exc}"), 500
    finally:
        # Don't retain uploaded medical images on disk.
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
