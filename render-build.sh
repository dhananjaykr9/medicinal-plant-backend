#!/usr/bin/env bash
set -e

mkdir -p models

echo "⬇️ Downloading models during build..."

curl -L -o models/resnet_finetuned_tf213.keras \
"https://drive.google.com/uc?export=download&id=1O1Wa1Pvhsp2khZAsRZ2r9pirOqUcez2c"

curl -L -o models/qpso_svm_model_finetuned.pkl \
"https://drive.google.com/uc?export=download&id=1i7OPtM4hgHDJAMfn3qamPL76_rOiPMbP"

curl -L -o models/qpso_scaler_finetuned.pkl \
"https://drive.google.com/uc?export=download&id=1KYuS4_2PxgI52pDUvMeD1wFH1ORiDnyN"

curl -L -o models/selected_indices_finetuned.npy \
"https://drive.google.com/uc?export=download&id=1X_r6ypUKMKE2MUYWyb5A6NDa8c1FqHcM"

curl -L -o models/class_names.npy \
"https://drive.google.com/uc?export=download&id=1Dc0Hits0RP8qH7A4B-j50EOjFHRErsE_"

echo "✅ Models downloaded"
