# 🫁 Pneumonia Detection from X-Ray Images

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)

A deep learning web application that detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). Upload an X-ray and get an instant prediction.

## 📖 About

Pneumonia is an infection in the lungs that inflames the air sacs in one or both lungs. It is caused by a bacteria called *Streptococcus pneumoniae* that usually lives in the upper respiratory tract. More than **10 million people in India** are affected by this annually. Doctors use a chest X-ray to confirm the infection and its location. Thus developing an automatic system for detecting the disease can help to treat patients quickly. With powerful Artificial Intelligence algorithms, the medical field has evolved in a very vast manner.

**Deep Learning Convolutional Neural Networks (CNN)** is the go-to algorithm when it comes to image classification. CNN models are so powerful that they are already in use for many disease detections. Moreover, CNN models can be used to analyze and find patterns in images. We build a CNN model to analyze the given X-ray image patterns and predict whether the patient is infected or not.

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** — [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) (1.24 GB)

## ✨ Features

- 🧠 **CNN Model** — Custom convolutional neural network trained on chest X-ray images
- 🌐 **Web Interface** — Flask-based UI for easy image upload and prediction
- ⚡ **Instant Results** — Upload an X-ray and get immediate classification
- 📓 **Training Notebook** — Jupyter notebook with full model training pipeline

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Model** | TensorFlow, Keras (CNN) |
| **Backend** | Flask, Gevent |
| **Frontend** | HTML, CSS, JavaScript |
| **Image Processing** | NumPy, Keras Preprocessing |

## 📁 Project Structure

```
├── CNN_model.ipynb    # Model training notebook
├── app.py             # Flask web application
├── templates/
│   └── index.html     # Web interface
└── static/
    ├── css/
    ├── js/
    └── images/
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow keras flask numpy gevent
```

### Train the Model

1. Download the [dataset from Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
2. Run `CNN_model.ipynb` to train and save the model as `cnn_model.h5`

### Run the Web App

```bash
python app.py
```

Navigate to `http://localhost:5000`, upload a chest X-ray image, and get the prediction.

## 👤 Author

**Sanjay Santhanam**

---

⭐ Star this repo if you found it useful!
