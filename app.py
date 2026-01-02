# ============================================================
# Medicinal Plant Prediction API
# ResNet50 (Fine-Tuned .keras) + QPSO + SVM
# Render-safe | Google Drive auto-download
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D

# --------- local utility modules ---------
from preprocess import preprocess_image
from plant_info import get_plant_info


# =====================================================
# RENDER SAFE DIRECTORIES
# =====================================================
BASE_MODEL_DIR = "/tmp/models"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)


# =====================================================
# GOOGLE DRIVE FILE IDS
# =====================================================
MODEL_IDS = {
    "resnet": "1O1Wa1Pvhsp2khZAsRZ2r9pirOqUcez2c",  # TF-Keras compatible
    "svm": "1i7OPtM4hgHDJAMfn3qamPL76_rOiPMbP",
    "scaler": "1KYuS4_2PxgI52pDUvMeD1wFH1ORiDnyN",
    "indices": "1X_r6ypUKMKE2MUYWyb5A6NDa8c1FqHcM",
    "classes": "1Dc0Hits0RP8qH7A4B-j50EOjFHRErsE_",
}


# =====================================================
# LOCAL MODEL PATHS
# =====================================================
MODEL_PATHS = {
    "resnet": f"{BASE_MODEL_DIR}/resnet_finetuned_tf213.keras",
    "svm": f"{BASE_MODEL_DIR}/qpso_svm_model_finetuned.pkl",
    "scaler": f"{BASE_MODEL_DIR}/qpso_scaler_finetuned.pkl",
    "indices": f"{BASE_MODEL_DIR}/selected_indices_finetuned.npy",
    "classes": f"{BASE_MODEL_DIR}/class_names.npy",
}


# =====================================================
# SAFE DOWNLOAD FUNCTION
# =====================================================
def download_if_missing(file_id: str, out_path: str):
    if os.path.exists(out_path):
        print(f"✔️ Exists: {out_path}")
        return

    print(f"⬇️ Downloading {out_path}")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=False, fuzzy=True)
    print(f"✅ Downloaded: {out_path}")


# =====================================================
# DOWNLOAD ALL FILES BEFORE LOADING
# =====================================================
download_if_missing(MODEL_IDS["resnet"], MODEL_PATHS["resnet"])
download_if_missing(MODEL_IDS["svm"], MODEL_PATHS["svm"])
download_if_missing(MODEL_IDS["scaler"], MODEL_PATHS["scaler"])
download_if_missing(MODEL_IDS["indices"], MODEL_PATHS["indices"])
download_if_missing(MODEL_IDS["classes"], MODEL_PATHS["classes"])

print("🔄 Loading models...")


# =====================================================
# LOAD RESNET (.keras) — FIXED & COMPATIBLE
# =====================================================
resnet_model = tf.keras.models.load_model(
    MODEL_PATHS["resnet"],
    compile=False
)
resnet_model.trainable = False


# =====================================================
# BUILD FEATURE EXTRACTOR
# =====================================================
def build_feature_extractor(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPool2D)):
            return Model(inputs=model.input, outputs=layer.output)

    return Model(inputs=model.input, outputs=model.layers[-2].output)


feature_extractor = build_feature_extractor(resnet_model)


# =====================================================
# LOAD QPSO + SVM ASSETS
# =====================================================
svm_model = joblib.load(MODEL_PATHS["svm"])
scaler = joblib.load(MODEL_PATHS["scaler"])
selected_indices = np.load(MODEL_PATHS["indices"])
class_names = np.load(MODEL_PATHS["classes"], allow_pickle=True).tolist()

print("✅ All models loaded successfully.")


# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="Medicinal Plant Identification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_array = preprocess_image(image)

    features = feature_extractor.predict(img_array, verbose=0)
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
# HEALTH CHECK
# =====================================================
@app.get("/")
def health():
    return {"status": "API running successfully"}
