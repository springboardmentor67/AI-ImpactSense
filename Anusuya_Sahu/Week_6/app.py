import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- CONFIGURATION ---
MODEL_PATH = '/earthquake_model_final.pkl'
SCALER_PATH = '/scaler.pkl'
FEATURE_NAMES_PATH = '/feature_names.pkl'

# --- PAGE SETUP ---
st.set_page_config(page_title="ImpactSense", page_icon="🌍", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
        margin-top: 20px;
    }
    .safe { background-color: #28a745; }
    .caution { background-color: #ffc107; color: black !important; }
    .danger { background-color: #fd7e14; }
    .critical { background-color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # Load Model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Load Scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    # Load Feature Names
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
        
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_resources()
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure Week 2 and Week 4 codes ran successfully.")
    st.stop()

# --- HEADER ---
st.title("🌍 ImpactSense")
st.markdown("### Earthquake Impact Prediction System")
st.write("Enter the seismic parameters below to predict the alert level.")

# --- INPUT FORM ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        magnitude = st.number_input("Magnitude (Richter)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
        depth = st.number_input("Depth (km)", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)
        cdi = st.number_input("CDI (Community Decimal Intensity)", min_value=0.0, max_value=12.0, value=5.0)
        
    with col2:
        mmi = st.number_input("MMI (Modified Mercalli Intensity)", min_value=0.0, max_value=12.0, value=6.0)
        sig = st.number_input("Significance Score (can be negative)", min_value=-1000.0, max_value=1000.0, value=100.0, step=1.0)
    
    submitted = st.form_submit_button("Predict Impact")

# --- PREDICTION LOGIC ---
if submitted:
    # 1. Feature Engineering (Must match Week 2 logic exactly)
    energy_release = 10 ** (1.5 * magnitude)
    impact_factor = magnitude / np.log1p(depth) if depth > 0 else magnitude
    is_shallow = 1 if depth < 70 else 0
    
    # 2. Prepare Data Frame
    input_data = pd.DataFrame({
        'magnitude': [magnitude],
        'depth': [depth],
        'cdi': [cdi],
        'mmi': [mmi],
        'sig': [sig],
        'energy_release': [energy_release],
        'impact_factor': [impact_factor],
        'is_shallow': [is_shallow]
    })
    
    # Ensure columns are in the exact same order as training
    input_data = input_data[feature_names]
    
    # 3. Scale Data
    input_scaled = scaler.transform(input_data)
    
    # 4. Predict
    prediction_idx = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    # 5. Display Result
    labels = ["Green Alert (Low Risk)", "Yellow Alert (Moderate)", "Orange Alert (High Risk)", "Red Alert (Critical)"]
    colors = ["safe", "caution", "danger", "critical"]
    
    result_text = labels[prediction_idx]
    result_class = colors[prediction_idx]
    confidence = probabilities[prediction_idx] * 100
    
    st.markdown(f'<div class="result-box {result_class}">{result_text}</div>', unsafe_allow_html=True)
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    # Explain why
    st.info(f"**Why?** Calculated Impact Factor: {impact_factor:.2f} | Energy: {energy_release:.2e} J")