# ============================================================
# Medicinal Plant Prediction API
# ResNet50 (Fine-Tuned .keras) + QPSO + SVM
# Render-safe (models downloaded at build time)
# ============================================================

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import numpy as np
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D

# --------- local utility modules ---------
from preprocess import preprocess_image
from plant_info import get_plant_info


# =====================================================
# MODEL DIRECTORY (POPULATED BY render-build.sh)
# =====================================================
BASE_MODEL_DIR = "models"

MODEL_PATHS = {
    "resnet": os.path.join(BASE_MODEL_DIR, "resnet_finetuned_tf213.keras"),
    "svm": os.path.join(BASE_MODEL_DIR, "qpso_svm_model_finetuned.pkl"),
    "scaler": os.path.join(BASE_MODEL_DIR, "qpso_scaler_finetuned.pkl"),
    "indices": os.path.join(BASE_MODEL_DIR, "selected_indices_finetuned.npy"),
    "classes": os.path.join(BASE_MODEL_DIR, "class_names.npy"),
}


# =====================================================
# SAFETY CHECK (FAIL FAST IF MODEL MISSING)
# =====================================================
for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        raise RuntimeError(f"❌ Required model file missing: {path}")


print("🔄 Loading models...")


# =====================================================
# LOAD RESNET (.keras, TF 2.13 compatible)
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
