# Pneumonia Detection from Chest X-Ray Images

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A deep-learning web app that classifies a chest X-ray as **Normal** or **Pneumonia**
using a Convolutional Neural Network (CNN). Upload an X-ray and get a prediction with a
confidence score.

> ⚠️ **Educational project — not a medical device.** Do not use for diagnosis or
> treatment. See [DISCLAIMER.md](DISCLAIMER.md).

## About

Pneumonia inflames the air sacs in one or both lungs and is diagnosed clinically, often
with the help of a chest X-ray. CNNs are a strong fit for image classification, and this
project trains one to recognise pneumonia patterns in X-rays as a learning exercise in
applied deep learning.

## Dataset

**Chest X-Ray Images (Pneumonia)** — [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
(~1.24 GB, ~5,860 images across `train` / `val` / `test`, classes `NORMAL` and `PNEUMONIA`).

## Model

A compact Keras `Sequential` CNN trained on 75×75 grayscale X-rays:

`Conv2D(64) → MaxPooling2D → Flatten → Dense(512) → Dense(256) → Dense(1, sigmoid)`

Binary cross-entropy loss, Adam optimizer. The full pipeline is in
[`CNN_model.ipynb`](CNN_model.ipynb), which saves the trained weights to `cnn_model.h5`.

## Tech stack

| Component | Technology |
|-----------|-----------|
| Model | TensorFlow / Keras (CNN) |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript (jQuery) |
| Image processing | NumPy, Pillow |

## Project structure

```
├── app.py                 # Flask web application
├── CNN_model.ipynb        # Model training notebook
├── requirements.txt       # Python dependencies
├── DISCLAIMER.md          # Medical disclaimer
├── tests/
│   └── test_smoke.py      # Smoke tests (TensorFlow mocked)
├── templates/
│   └── index.html         # Web interface
└── static/
    ├── css/  ├── js/  └── images/
```

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a trained model

Either train it yourself — download the [dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
and run `CNN_model.ipynb` (set `DATA_DIR` to your local dataset path); it saves
`cnn_model.h5` — or drop an existing `cnn_model.h5` in the project root.

### 3. Run the web app

```bash
python app.py                 # serves on http://localhost:5000
```

Open `http://localhost:5000`, upload a chest X-ray, and click **Predict**.

## API

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Liveness probe → `{"status":"ok","model_present":bool}` |
| `/predict` | POST | multipart `image` file → `{"label","pneumonia","confidence"}` |

```bash
curl -F "image=@chest_xray.png" http://localhost:5000/predict
# {"label":"Pneumonia detected","pneumonia":true,"confidence":87.3}
```

## Tests

Smoke tests mock TensorFlow, so they run in seconds without the weights or dataset:

```bash
pip install pytest && pytest -q
```

## Limitations

- Trained only on the Kaggle dataset; performance on X-rays from other equipment,
  populations, or preprocessing pipelines is not guaranteed.
- 75×75 grayscale input discards fine detail a clinical model would retain.
- Binary output only (Normal vs Pneumonia) — it does not identify pneumonia type,
  severity, or location, and gives no uncertainty calibration.
- **Not validated for clinical use.** See [DISCLAIMER.md](DISCLAIMER.md).

## License

[MIT](LICENSE) © Sanjay Santhanam
