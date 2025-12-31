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
import h5py

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D

# --------- local utility modules ---------
from preprocess import preprocess_image
from plant_info import get_plant_info


os.environ["TF_USE_LEGACY_KERAS"] = "1"

# =====================================================
# MODEL DOWNLOAD CONFIG
# =====================================================
BASE_MODEL_DIR = "/tmp/models"   # Render-safe
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

# =====================================================
# SAFE DOWNLOAD + VALIDATION
# =====================================================
def is_valid_h5(path: str) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def download_model(file_id: str, output_path: str, is_h5=False):
    must_download = True

    if os.path.exists(output_path):
        if is_h5:
            if is_valid_h5(output_path):
                print(f"✔️ Valid model exists: {output_path}")
                must_download = False
            else:
                print(f"❌ Invalid H5 detected, re-downloading: {output_path}")
                os.remove(output_path)
        else:
            print(f"✔️ Exists: {output_path}")
            must_download = False

    if must_download:
        print(f"⬇️ Downloading {output_path} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Downloaded {output_path}")

# =====================================================
# DOWNLOAD MODELS (BEFORE LOADING)
# =====================================================
download_model(MODEL_IDS["resnet"], MODEL_PATHS["resnet"], is_h5=True)
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
# Load ResNet50
# =====================================================

resnet_model = tf.keras.models.load_model(
    MODEL_PATHS["resnet"],
    compile=False
)


#resnet_model = load_model(MODEL_PATHS["resnet"])
resnet_model.trainable = False

# =====================================================
# Build feature extractor
# =====================================================
def build_feature_extractor(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPool2D)):
            return Model(inputs=model.input, outputs=layer.output)
    return Model(inputs=model.input, outputs=model.layers[-2].output)

feature_extractor = build_feature_extractor(resnet_model)

# =====================================================
# Load QPSO + SVM assets
# =====================================================
svm_model = joblib.load(MODEL_PATHS["svm"])
scaler = joblib.load(MODEL_PATHS["scaler"])
selected_indices = np.load(MODEL_PATHS["indices"])
class_names = np.load(MODEL_PATHS["classes"], allow_pickle=True).tolist()

print("✅ All models loaded successfully.")

# =====================================================
# Prediction endpoint
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_array = preprocess_image(image)

    features = feature_extractor.predict(img_array)
    features = features[:, selected_indices]
    features = scaler.transform(features)

    probs = svm_model.predict_proba(features)[0]
    idx = int(np.argmax(probs))

    return {
        "plant_name": class_names[idx],
        "confidence": round(float(probs[idx]), 4),
        "description": get_plant_info(class_names[idx]),
        "gradcam_image": "",
        "shap_image": ""
    }

# =====================================================
# Health check
# =====================================================
@app.get("/")
def health():
    return {"status": "API running successfully"}
