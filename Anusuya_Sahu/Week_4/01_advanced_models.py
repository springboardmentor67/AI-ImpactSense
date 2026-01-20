import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURATION ---
INPUT_DIR = '../Week_2/output'
OUTPUT_DIR = 'output'

# Suppress warnings to keep console clean
warnings.filterwarnings('ignore')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    print("--- Loading Processed Data ---")
    X_train = np.load(os.path.join(INPUT_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(INPUT_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(INPUT_DIR, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def evaluate_and_plot(model, X_test, y_test, model_name):
    print(f"\n--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Green', 'Yellow', 'Orange', 'Red']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Green', 'Yellow', 'Orange', 'Red'],
                yticklabels=['Green', 'Yellow', 'Orange', 'Red'])
    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {acc:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = os.path.join(OUTPUT_DIR, f'cm_{model_name.replace(" ", "_")}.png')
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")
    plt.close()
    
    return acc

def train_random_forest(X_train, y_train):
    print("\n--- Training Random Forest (Full Power) ---")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best RF Params: {grid_search.best_params_}")
    print(f"Best RF Cross-Val Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_xgboost(X_train, y_train):
    print("\n--- Training XGBoost (God Mode) ---")
    
    # Optimized Grid to run faster
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [5, 10],
        'subsample': [0.8], 
        'colsample_bytree': [0.8] 
    }
    
    # REMOVED: use_label_encoder=False (This was causing the warning)
    xgb = XGBClassifier(eval_metric='mlogloss')
    
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best XGB Params: {grid_search.best_params_}")
    print(f"Best XGB Cross-Val Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    # 1. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_acc = evaluate_and_plot(rf_model, X_test, y_test, "Random Forest")
    
    # 2. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_acc = evaluate_and_plot(xgb_model, X_test, y_test, "XGBoost")
    
    # 3. Compare and Save Best
    print("\n--- Final Championship Results ---")
    print(f"Random Forest Accuracy: {rf_acc:.2%}")
    print(f"XGBoost Accuracy:       {xgb_acc:.2%}")
    
    best_model = xgb_model if xgb_acc > rf_acc else rf_model
    best_name = "XGBoost" if xgb_acc > rf_acc else "Random Forest"
    
    # Save the absolute best model for the UI
    with open(os.path.join(OUTPUT_DIR, 'earthquake_model_final.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
        
    print(f"\nWINNER: {best_name} is saved as 'earthquake_model_final.pkl'")