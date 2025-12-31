# ============================================================
# Medicinal Plant Prediction API
# ResNet50 (Fine-Tuned) + QPSO + SVM
# AUTO-DOWNLOAD MODELS AT STARTUP (Render-safe, Google Drive safe)
# ============================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import numpy as np
import joblib
import gdown

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D

# --------- local utility modules ---------
from preprocess import preprocess_image
from plant_info import get_plant_info

# =====================================================
# MODEL DOWNLOAD CONFIG
# =====================================================
BASE_MODEL_DIR = "models"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

MODEL_IDS = {
    "resnet": "1CSGs9CpNK6_Re43wHxl5tyWv6Qbcmi4z",
    "svm": "1i7OPtM4hgHDJAMfn3qamPL76_rOiPMbP",
    "scaler": "1KYuS4_2PxgI52pDUvMeD1wFH1ORiDnyN",
    "indices": "1X_r6ypUKMKE2MUYWyb5A6NDa8c1FqHcM",
    "classes": "1Dc0Hits0RP8qH7A4B-j50EOjFHRErsE_",
}

MODEL_PATHS = {
    "resnet": f"{BASE_MODEL_DIR}/resnet_finetuned_model.h5",
    "svm": f"{BASE_MODEL_DIR}/qpso_svm_model_finetuned.pkl",
    "scaler": f"{BASE_MODEL_DIR}/qpso_scaler_finetuned.pkl",
    "indices": f"{BASE_MODEL_DIR}/selected_indices_finetuned.npy",
    "classes": f"{BASE_MODEL_DIR}/class_names.npy",
}

def download_model(file_id: str, output_path: str):
    """
    Download a file from Google Drive using gdown (safe & reliable).
    """
    if not os.path.exists(output_path):
        print(f"⬇️ Downloading {output_path} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Downloaded: {output_path}")
    else:
        print(f"✔️ Exists: {output_path}")

# =====================================================
# DOWNLOAD MODELS (BEFORE LOADING)
# =====================================================
download_model(MODEL_IDS["resnet"], MODEL_PATHS["resnet"])
download_model(MODEL_IDS["svm"], MODEL_PATHS["svm"])
download_model(MODEL_IDS["scaler"], MODEL_PATHS["scaler"])
download_model(MODEL_IDS["indices"], MODEL_PATHS["indices"])
download_model(MODEL_IDS["classes"], MODEL_PATHS["classes"])

print("🔄 Loading models and assets...")

# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(title="Medicinal Plant Identification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 1️⃣ Load fine-tuned ResNet50
# =====================================================
resnet_model = load_model(
    MODEL_PATHS["resnet"],
    compile=False
)
resnet_model.trainable = False

# =====================================================
# 2️⃣ Build GAP-aligned feature extractor
# =====================================================
def build_feature_extractor(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPool2D)):
            print(f"✅ Using pooling layer: {layer.name}")
            return Model(inputs=model.input, outputs=layer.output)

    print("⚠️ No GAP found; using penultimate layer")
    return Model(inputs=model.input, outputs=model.layers[-2].output)

feature_extractor = build_feature_extractor(resnet_model)

# =====================================================
# 3️⃣ Load QPSO + SVM artifacts
# =====================================================
svm_model = joblib.load(MODEL_PATHS["svm"])
scaler = joblib.load(MODEL_PATHS["scaler"])
selected_indices = np.load(MODEL_PATHS["indices"])

# =====================================================
# 4️⃣ Load class names
# =====================================================
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

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_array = preprocess_image(image)

    deep_features = feature_extractor.predict(img_array)
    deep_features = deep_features[:, selected_indices]
    deep_features = scaler.transform(deep_features)

    probabilities = svm_model.predict_proba(deep_features)[0]
    pred_index = int(np.argmax(probabilities))

    plant_name = class_names[pred_index]
    confidence = float(probabilities[pred_index])

    description = get_plant_info(plant_name)

    return {
        "plant_name": plant_name,
        "confidence": round(confidence, 4),
        "description": description,
        "gradcam_image": "",
        "shap_image": ""
    }

# =====================================================
# Health check
# =====================================================
@app.get("/")
def health():
    return {"status": "API running successfully"}
