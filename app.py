# app.py
# ============================================================
# Medicinal Plant Prediction API
# ResNet50 (Fine-Tuned) + QPSO + SVM
# ============================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import numpy as np
import joblib

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D

# --------- local utility modules ---------
from preprocess import preprocess_image
from plant_info import get_plant_info

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
print("🔄 Loading models and assets...")

# -------------------------------
# 1️⃣ Load fine-tuned ResNet50
# -------------------------------
RESNET_MODEL_PATH = "models/resnet_finetuned_model.h5"

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
SVM_MODEL_PATH = "models/qpso_svm_model_finetuned.pkl"
SCALER_PATH = "models/qpso_scaler_finetuned.pkl"
QPSO_INDICES_PATH = "models/selected_indices_finetuned.npy"

svm_model = joblib.load(SVM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
selected_indices = np.load(QPSO_INDICES_PATH)

# -------------------------------
# 4️⃣ Load class names (index → label)
# -------------------------------
CLASS_NAMES_PATH = "models/class_names.npy"
class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()

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
    img_array = preprocess_image(image)   # shape: (1, 224, 224, 3)

    # ---------- Feature extraction ----------
    deep_features = feature_extractor.predict(img_array)  # (1, N)

    # ---------- QPSO feature selection ----------
    deep_features = deep_features[:, selected_indices]

    # ---------- Scaling ----------
    deep_features = scaler.transform(deep_features)

    # ---------- SVM prediction ----------
    probabilities = svm_model.predict_proba(deep_features)[0]
    pred_index = int(np.argmax(probabilities))

    plant_name = class_names[pred_index]
    confidence = float(probabilities[pred_index])

    # ---------- Explainability ----------
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
