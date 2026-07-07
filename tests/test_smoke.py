"""Smoke tests for the pneumonia-detection Flask app.

These tests mock TensorFlow/Keras so CI stays fast and does not need the ~1 GB
TF wheel or the trained ``cnn_model.h5`` weights. They verify routing, upload
validation, error handling, and the happy-path prediction wiring — not model
accuracy (which requires the real weights + dataset).
"""

import io
import sys
import types

import numpy as np
import pytest


def _install_fake_tensorflow():
    """Inject a minimal fake ``tensorflow.keras`` into sys.modules.

    app.py imports ``load_model`` and ``preprocessing.image`` at module load
    time, so the fakes must exist before app is imported.
    """
    tf = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    models = types.ModuleType("tensorflow.keras.models")
    preprocessing = types.ModuleType("tensorflow.keras.preprocessing")
    image = types.ModuleType("tensorflow.keras.preprocessing.image")

    def load_model(path):  # pragma: no cover - replaced per-test
        raise RuntimeError("fake load_model should be monkeypatched in tests")

    class _FakeImg:
        pass

    def load_img(path, target_size=None, color_mode=None):
        return _FakeImg()

    def img_to_array(img):
        # 75x75x1 grayscale, values in [0, 255]
        return np.zeros((75, 75, 1), dtype="float32")

    models.load_model = load_model
    image.load_img = load_img
    image.img_to_array = img_to_array
    keras.models = models
    keras.preprocessing = preprocessing
    preprocessing.image = image
    tf.keras = keras

    sys.modules.update({
        "tensorflow": tf,
        "tensorflow.keras": keras,
        "tensorflow.keras.models": models,
        "tensorflow.keras.preprocessing": preprocessing,
        "tensorflow.keras.preprocessing.image": image,
    })


_install_fake_tensorflow()

import app as app_module  # noqa: E402


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "model_present" in body


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"PNEUMONIA DETECTION" in resp.data


def test_predict_no_file(client):
    resp = client.post("/predict", data={})
    assert resp.status_code == 400


def test_predict_bad_extension(client):
    data = {"image": (io.BytesIO(b"not-an-image"), "notes.txt")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_happy_path(client, monkeypatch):
    """A pneumonia-positive prediction returns label + confidence."""
    class _FakeModel:
        def predict(self, x, verbose=0):
            return np.array([[0.87]])  # >= 0.5 -> pneumonia

    monkeypatch.setattr(app_module, "get_model", lambda: _FakeModel())

    data = {"image": (io.BytesIO(b"\x89PNG fake bytes"), "xray.png")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["pneumonia"] is True
    assert body["label"].startswith("Pneumonia")
    assert body["confidence"] == pytest.approx(87.0, abs=0.5)


def test_predict_normal_path(client, monkeypatch):
    class _FakeModel:
        def predict(self, x, verbose=0):
            return np.array([[0.12]])  # < 0.5 -> normal

    monkeypatch.setattr(app_module, "get_model", lambda: _FakeModel())

    data = {"image": (io.BytesIO(b"\x89PNG fake bytes"), "xray.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["pneumonia"] is False
    assert body["confidence"] == pytest.approx(88.0, abs=0.5)
