import streamlit as st
import joblib
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# === Load models ===
heart_model = joblib.load("models/heart_xgboost_ensemble_model.pkl")
ckd_model = joblib.load("models/ckd_model_honest.pkl")
pneumonia_model = load_model("models/pneumonia_mobilenet_balanced.h5")
retino_svm = joblib.load("models/retinopathy_vgg16_svm.pkl")
retino_rf = joblib.load("models/retinopathy_vgg16_rf.pkl")
retino_scaler = joblib.load("models/retinopathy_vgg16_scaler.pkl")

# === Helper functions ===
def preprocess_image(image, size=224):
    image = Image.open(image).convert("RGB")
    image = image.resize((size, size))
    image = np.array(image)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def retino_feature_extraction(image):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)
    features = feature_model.predict(image)
    return features.reshape(1, -1)

def predict_retinopathy(image):
    image = preprocess_image(image)
    features = retino_feature_extraction(image)
    scaled_features = retino_scaler.transform(features)
    svm_prob = retino_svm.predict_proba(scaled_features)[0][1]
    rf_prob = retino_rf.predict_proba(scaled_features)[0][1]
    avg_prob = (svm_prob + rf_prob) / 2
    pred = 1 if avg_prob >= 0.5 else 0
    return pred, avg_prob

# === UI ===
st.set_page_config(page_title="Multi Disease Prediction System", layout="wide")
st.title("🩺 Multi Disease Prediction System")

option = st.sidebar.selectbox(
    "Choose a Prediction Task",
    ["Heart Disease", "Chronic Kidney Disease", "Pneumonia Detection", "Diabetic Retinopathy"]
)

if option == "Heart Disease":
    st.header("🫀 Heart Disease Prediction")
    age = st.number_input("Age", 20, 100, 45)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("CA (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [1, 2, 3])

    if st.button("Predict Heart Disease"):
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                           'restecg', 'thalach', 'exang', 'oldpeak',
                                           'slope', 'ca', 'thal'])
        pred = heart_model.predict(input_data)[0]
        st.success("Heart Disease Detected" if pred == 1 else "No Heart Disease Detected")

elif option == "Chronic Kidney Disease":
    st.header("🧫 Chronic Kidney Disease Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file (same format as training)")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        # Ensure all required features are present
        required_features = ['age', 'bp', 'sg', 'al', 'su', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm']
        if not all(col in input_df.columns for col in required_features):
            st.error(f"CSV must contain these columns: {required_features}")
        else:
            # Convert 'yes'/'no' to 1/0
            input_df['htn'] = input_df['htn'].map({'yes': 1, 'no': 0})
            input_df['dm'] = input_df['dm'].map({'yes': 1, 'no': 0})

            pred = ckd_model.predict(input_df[required_features])[0]
            st.success("CKD Detected" if pred == 1 else "No CKD Detected")

elif option == "Pneumonia Detection":
    st.header("🫁 Pneumonia Detection (Chest X-ray)")
    uploaded_image = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        img = preprocess_image(uploaded_image)
        pred_prob = pneumonia_model.predict(img)[0][0]
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.success("Pneumonia Detected" if pred_prob > 0.5 else "Normal")

elif option == "Diabetic Retinopathy":
    st.header("👁️ Diabetic Retinopathy Detection")
    uploaded_eye = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])
    if uploaded_eye is not None:
        pred, prob = predict_retinopathy(uploaded_eye)
        st.image(uploaded_eye, caption="Uploaded Eye Image", use_column_width=True)
        st.success("Retinopathy Detected" if pred == 1 else "No Retinopathy Detected")
        st.caption(f"Prediction Confidence: {prob:.2f}")
