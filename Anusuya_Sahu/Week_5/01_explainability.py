import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import shap

# --- CONFIGURATION ---
MODEL_PATH = '../Week_4/output/earthquake_model_final.pkl'
DATA_DIR = '../Week_2/output'
OUTPUT_DIR = 'output'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_resources():
    print("--- Loading Model and Data ---")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    
    with open(os.path.join(DATA_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
        
    print(f"Model Loaded: {type(model).__name__}")
    return model, X_test, feature_names

def plot_feature_importance(model, feature_names):
    print("\n--- Generating Feature Importance Plot ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    feat_imp_df = pd.DataFrame({
        'Feature': np.array(feature_names)[indices],
        'Importance': importances[indices]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title('Global Feature Importance')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, '01_feature_importance.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def plot_shap_values(model, X_test, feature_names):
    print("\n--- Generating SHAP Analysis ---")
    print("Calculating SHAP values...")
    
    # FIX: Convert Numpy Array back to DataFrame so SHAP knows column names
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Create Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)
    
    # FIX: Handle different return types of shap_values
    # Random Forest returns a list of arrays (one for each class: Green, Yellow, Orange, Red)
    if isinstance(shap_values, list):
        print(f"SHAP returned values for {len(shap_values)} classes.")
        
        # Plot Red Alert (Class 3)
        print("Generating Red Alert Plot...")
        plt.figure()
        plt.title("Why does the model predict RED ALERT?")
        shap.summary_plot(shap_values[3], X_test_df, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, '02_shap_red_alert.png'), bbox_inches='tight')
        plt.close()
        
        # Plot Green Alert (Class 0)
        print("Generating Green Alert Plot...")
        plt.figure()
        plt.title("Why does the model predict GREEN ALERT?")
        shap.summary_plot(shap_values[0], X_test_df, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, '03_shap_green_alert.png'), bbox_inches='tight')
        plt.close()
        
    else:
        # Fallback for binary/other cases
        print("SHAP returned a single matrix.")
        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, '02_shap_summary.png'), bbox_inches='tight')
        plt.close()

    print("SHAP plots saved successfully.")

if __name__ == "__main__":
    model, X_test, feature_names = load_resources()
    
    # 1. Feature Importance
    plot_feature_importance(model, feature_names)
    
    # 2. SHAP (with fix)
    try:
        plot_shap_values(model, X_test, feature_names)
    except Exception as e:
        print(f"SHAP Error: {e}")