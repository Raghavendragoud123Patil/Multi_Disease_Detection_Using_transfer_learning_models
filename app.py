import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load Models
heart_model = joblib.load("models/heart_xgboost_ensemble_model.pkl")
ckd_model = joblib.load("models/ckd_model_honest.pkl")
pneumonia_model = load_model("models/pneumonia_mobilenet_balanced.h5")
retino_svm = joblib.load("models/retinopathy_vgg16_svm.pkl")
retino_rf = joblib.load("models/retinopathy_vgg16_rf.pkl")
retino_scaler = joblib.load("models/retinopathy_vgg16_scaler.pkl")

# Preprocessing
def preprocess_image(image, size=224):
    image = Image.open(image).convert("RGB")
    image = image.resize((size, size))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def extract_retino_features(image):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.output)
    features = feature_model.predict(image, verbose=0)
    return features.reshape(1, -1)

def predict_retinopathy(image):
    img = preprocess_image(image)
    features = extract_retino_features(img)
    scaled = retino_scaler.transform(features)
    prob_svm = retino_svm.predict_proba(scaled)[0][1]
    prob_rf = retino_rf.predict_proba(scaled)[0][1]
    avg = (prob_svm + prob_rf) / 2
    return 1 if avg > 0.5 else 0, avg

# App Layout
st.set_page_config(page_title="🩺 Disease Detector", layout="wide")

# Sidebar Selection
st.sidebar.title("🧬 Disease Selector")
disease = st.sidebar.radio("Choose a Prediction Task", [
    "Cardiovascular Disease", "Chronic Kidney Disease", "Pneumonia Detection", "Diabetic Retinopathy"
])

# Title
st.markdown("<h1 style='text-align:center;'>🩺 Multi Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Heart Disease
if disease == "Cardiovascular Disease":
    st.header("🫀 Cardiovascular Disease Detection")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
    with col2:
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Number of Vessels Colored (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [1, 2, 3])

    if st.button("Predict Cardiovascular Disease"):
        X = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]],
                         columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                  'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        result = heart_model.predict(X)[0]
        st.success("Cardiovascular Disease Detected" if result == 1 else "No Cardiovascular Disease")

# CKD
elif disease == "Chronic Kidney Disease":
    st.header("🧫 Chronic Kidney Disease Detection")
    with st.form("CKD Form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 100, 40)
            bp = st.number_input("BP", 50, 200, 80)
            sg = st.selectbox("SG", [1.005, 1.010, 1.015, 1.020, 1.025])
            al = st.selectbox("AL", [0, 1, 2, 3, 4, 5])
            su = st.selectbox("SU", [0, 1, 2, 3, 4, 5])
            rbc = st.selectbox("RBC", ["normal", "abnormal"])
            pc = st.selectbox("PC", ["normal", "abnormal"])
            pcc = st.selectbox("PCC", ["present", "notpresent"])
            ba = st.selectbox("BA", ["present", "notpresent"])
            bgr = st.number_input("BGR", 50, 500, 150)
            bu = st.number_input("BU", 1, 300, 40)
        with col2:
            sc = st.number_input("SC", 0.1, 10.0, 1.2)
            sod = st.number_input("SOD", 100, 200, 140)
            pot = st.number_input("POT", 2.0, 10.0, 4.5)
            hemo = st.number_input("Hemo", 3.0, 17.0, 12.0)
            pcv = st.number_input("PCV", 10.0, 60.0, 40.0)
            wc = st.number_input("WBC", 2000, 20000, 8000)
            rc = st.number_input("RBC Count", 2.5, 6.5, 5.0)
            htn = st.selectbox("Hypertension", ["yes", "no"])
            dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
            cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
            appet = st.selectbox("Appetite", ["good", "poor"])
            pe = st.selectbox("Pedal Edema", ["yes", "no"])
            ane = st.selectbox("Anemia", ["yes", "no"])
        submitted = st.form_submit_button("Predict CKD")

        if submitted:
            input_dict = {
                'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
                'rbc': rbc.lower(), 'pc': pc.lower(), 'pcc': pcc.lower(), 'ba': ba.lower(),
                'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
                'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
                'htn': htn.lower(), 'dm': dm.lower(), 'cad': cad.lower(),
                'appet': appet.lower(), 'pe': pe.lower(), 'ane': ane.lower()
            }
            df = pd.DataFrame([input_dict])
            result = ckd_model.predict(df)[0]
            st.success("CKD Detected" if result == 1 else "🟢 No CKD Detected")

# Pneumonia
elif disease == "Pneumonia Detection":
    st.header("🫁 Pneumonia Detection from Chest X-ray")
    img_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = preprocess_image(img_file)
        prediction = pneumonia_model.predict(img)[0][0]
        st.image(img_file, use_container_width=True, caption="Chest X-ray")
        st.success("Pneumonia Detected" if prediction > 0.5 else "🫁 Normal")
        st.success(f"Detection Confidence: {prediction:.2f}")

# Diabetic Retinopathy
elif disease == "Diabetic Retinopathy":
    st.header("👁️ Diabetic Retinopathy Detection")
    eye_img = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])
    if eye_img:
        @st.cache_resource
        def load_retino_models():
            return (
                joblib.load("models/retinopathy_vgg16_rf.pkl"),
                joblib.load("models/retinopathy_vgg16_svm.pkl"),
                joblib.load("models/retinopathy_vgg16_scaler.pkl")
            )
        rf_model, svm_model, scaler = load_retino_models()

        image = Image.open(eye_img).convert("RGB").resize((224, 224))
        arr = preprocess_input(img_to_array(image))
        features = extract_retino_features(np.expand_dims(arr, axis=0))
        scaled = scaler.transform(features)

        prob_rf = rf_model.predict_proba(scaled)[0][1]
        prob_svm = svm_model.predict_proba(scaled)[0][1]
        avg_prob = (prob_rf + prob_svm) / 2
        result = 1 if avg_prob >= 0.5 else 0

        st.image(image, caption="Fundus Image", use_container_width=True)
        st.success("Retinopathy Detected" if result == 1 else "🟢 No Retinopathy Detected")
        st.success(f"Detection Confidence: {avg_prob:.2f}")
