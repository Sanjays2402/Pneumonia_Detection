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

## Terms & Disclaimer

By using this project you agree to the following.

- **No warranty.** This software is provided "as is", without warranty of any kind, express or implied, including merchantability, fitness for a particular purpose, and non-infringement.
- **No liability.** To the maximum extent permitted by law, the author and contributors are not liable for any claim, loss, or damage (direct, indirect, incidental, or consequential, including lost data or profits) arising from its use.
- **Your responsibility.** You use it at your own risk and are responsible for complying with the laws that apply to you.
- **Not medical advice.** This is not a medical device and has not been cleared by the FDA, Health Canada, or CDSCO. It does not replace a doctor or pharmacist. In an emergency, call 911 (USA/Canada) or 112 (India).
- **Model results.** Models are trained for learning and research, may be inaccurate or biased, and are not validated for real-world decisions about people.
- **Not for diagnosis.** This is a research and learning project. It is not a medical device, has not been clinically validated, and must not be used to diagnose or treat anyone.
- **Third parties.** Names, logos, APIs, and services of others belong to their owners and are used for identification only; no affiliation or endorsement is implied.
- **Your statutory rights.** Nothing here limits rights that cannot be waived by agreement, including under consumer-protection laws in the USA, Canada (including Quebec), and India (Consumer Protection Act, 2019).
- **License.** Use, copying, and distribution are governed by the [LICENSE](LICENSE) file; if anything here conflicts with it, the license controls.

## Content, Assets & Fair Use

- **Original work.** Unless noted otherwise (including below), the code, text, design, and images in this repository were created by the author.
- **Libraries, fonts, and icons.** Third-party libraries, fonts (for example Google Fonts), and icon sets belong to their creators and are used under their own licenses; see the dependency manifests and each source for terms.
- **Logos, brands, and screenshots.** Third-party names, logos, and screenshots appear only to identify or describe their products (nominative use). They remain the property of their owners and do not imply affiliation or endorsement.
- **Ideas and inspiration.** Where this project is inspired by another product, game, or idea, it is an independent implementation. It borrows only general ideas, mechanics, or styles, which are not anyone's exclusive property, and is meant to contain no third-party code or artwork used without permission.
- **Datasets.** Datasets belong to their publishers and are used under their licenses or terms for learning and research. They may be excluded from the repository; get them from the original source and follow its license.
- **Fair use and fair dealing.** Any third-party material is used in limited ways for purposes such as education, research, review, commentary, or identification, relying on fair use in the USA (17 U.S.C. § 107), fair dealing in Canada (Copyright Act, ss. 29–29.2), and fair dealing in India (Copyright Act, 1957, s. 52), where they apply.
- **Concerns and takedown requests.** If you own rights in something here and believe it is used improperly, [open an issue](https://github.com/Sanjays2402/Pneumonia_Detection/issues) or contact the author through their GitHub profile with a link to the material and your claim. It will be reviewed promptly and credited, changed, or removed as appropriate. You may also use GitHub's DMCA process.
- **Contributions.** By contributing, you confirm you have the right to submit your contribution and agree it is licensed under this repository's license.
