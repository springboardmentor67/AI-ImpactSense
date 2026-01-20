#!/usr/bin/env python
"""Test script to verify model loading."""
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "earthquake_impact_rf.pkl")
feature_path = os.path.join(script_dir, "feature_order.pkl")

print(f"Model path: {model_path}")
print(f"Feature path: {feature_path}")
print(f"Model exists: {os.path.exists(model_path)}")
print(f"Feature exists: {os.path.exists(feature_path)}")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded successfully: {model}")
    features = joblib.load(feature_path)
    print(f"Features: {features}")

