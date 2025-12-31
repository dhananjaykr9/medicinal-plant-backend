# ============================================================
# Medicinal Plant Prediction API
# ResNet50 (Fine-Tuned) + QPSO + SVM
# AUTO-DOWNLOAD MODELS AT STARTUP (Render-safe)
# ============================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import json
import numpy as np
import joblib
import requests

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D

# --------- local utility modules ---------
from preprocess import preprocess_image
from plant_info import get_plant_info

# =====================================================
# MODEL DOWNLOAD CONFIG (Render-safe)
# =====================================================
BASE_MODEL_DIR = "/tmp/models"

MODEL_URLS = {
    "resnet": "https://drive.google.com/uc?id=1CSGs9CpNK6_Re43wHxl5tyWv6Qbcmi4z&export=download",
    "svm": "https://drive.google.com/uc?id=1i7OPtM4hgHDJAMfn3qamPL76_rOiPMbP&export=download",
    "scaler": "https://drive.google.com/uc?id=1KYuS4_2PxgI52pDUvMeD1wFH1ORiDnyN&export=download",
    "indices": "https://drive.google.com/uc?id=1X_r6ypUKMKE2MUYWyb5A6NDa8c1FqHcM&export=download",
    "classes": "https://drive.google.com/uc?id=1Dc0Hits0RP8qH7A4B-j50EOjFHRErsE_&export=download",
}

MODEL_PATHS = {
    "resnet": f"{BASE_MODEL_DIR}/resnet_finetuned_model.h5",
    "svm": f"{BASE_MODEL_DIR}/qpso_svm_model_finetuned.pkl",
    "scaler": f"{BASE_MODEL_DIR}/qpso_scaler_finetuned.pkl",
    "indices": f"{BASE_MODEL_DIR}/selected_indices_finetuned.npy",
    "classes": f"{BASE_MODEL_DIR}/class_names.npy",
}

def download_models():
    """
    Downloads models from Google Drive if not already present.
    Called once at server startup.
    """
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)

    for key, url in MODEL_URLS.items():
        path = MODEL_PATHS[key]

        if not os.path.exists(path):
            print(f"⬇️ Downloading {key}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✅ {key} downloaded")
        else:
            print(f"✔️ {key} already exists")

# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(title="Medicinal Plant Identification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Android / local access
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Load models and assets (ONCE at startup)
# =====================================================
download_models()   # 🔥 AUTO-DOWNLOAD FIRST

print("🔄 Loading models and assets...")

# -------------------------------
# 1️⃣ Load fine-tuned ResNet50
# -------------------------------
RESNET_MODEL_PATH = MODEL_PATHS["resnet"]

resnet_model = load_model(RESNET_MODEL_PATH)
resnet_model.trainable = False

# -------------------------------
# 2️⃣ Build GAP-aligned feature extractor
# -------------------------------
def build_feature_extractor(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPool2D)):
            print(f"✅ Using pooling layer for features: {layer.name}")
            return Model(inputs=model.input, outputs=layer.output)

    print("⚠️ No GAP layer found; using second-last layer")
    return Model(inputs=model.input, outputs=model.layers[-2].output)

feature_extractor = build_feature_extractor(resnet_model)

# -------------------------------
# 3️⃣ Load QPSO + SVM artifacts
# -------------------------------
svm_model = joblib.load(MODEL_PATHS["svm"])
scaler = joblib.load(MODEL_PATHS["scaler"])
selected_indices = np.load(MODEL_PATHS["indices"])

# -------------------------------
# 4️⃣ Load class names
# -------------------------------
class_names = np.load(MODEL_PATHS["classes"], allow_pickle=True).tolist()

print("📋 Loaded class names:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print("✅ All models and assets loaded successfully.")

# =====================================================
# Prediction endpoint
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Input : Leaf image
    Output: Plant name, confidence, Grad-CAM, SHAP, description
    """

    # ---------- Load image ----------
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ---------- Preprocess ----------
    img_array = preprocess_image(image)   # (1, 224, 224, 3)

    # ---------- Feature extraction ----------
    deep_features = feature_extractor.predict(img_array)

    # ---------- QPSO feature selection ----------
    deep_features = deep_features[:, selected_indices]

    # ---------- Scaling ----------
    deep_features = scaler.transform(deep_features)

    # ---------- SVM prediction ----------
    probabilities = svm_model.predict_proba(deep_features)[0]
    pred_index = int(np.argmax(probabilities))

    plant_name = class_names[pred_index]
    confidence = float(probabilities[pred_index])

    # ---------- Explainability (placeholders) ----------
    gradcam_img = ""
    shap_img = ""

    # ---------- External plant knowledge ----------
    description = get_plant_info(plant_name)

    return {
        "plant_name": plant_name,
        "confidence": round(confidence, 4),
        "description": description,
        "gradcam_image": gradcam_img,
        "shap_image": shap_img
    }

# =====================================================
# Health check
# =====================================================
@app.get("/")
def health():
    return {"status": "API running successfully"}
