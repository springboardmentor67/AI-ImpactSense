import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURATION ---
INPUT_DIR = '../Week_2/output'
OUTPUT_DIR = 'output'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    """Load the arrays saved in Week 2"""
    print("--- Loading Processed Data ---")
    X_train = np.load(os.path.join(INPUT_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(INPUT_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(INPUT_DIR, 'y_test.npy'))
    
    # Load feature names for better understanding
    with open(os.path.join(INPUT_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
        
    print(f"Features: {feature_names}")
    return X_train, X_test, y_train, y_test, feature_names

def evaluate_model(model, X_test, y_test, model_name):
    """
    Generic function to evaluate any model and save results
    """
    print(f"\n--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    
    # 1. Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # 2. Detailed Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Green', 'Yellow', 'Orange', 'Red']))
    
    # 3. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
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

def train_logistic_regression(X_train, y_train):
    """
    Trains Logistic Regression with Hyperparameter Tuning
    """
    print("\n--- Training Logistic Regression (Full Power) ---")
    
    # Define the "Grid" of settings to test
    # C: Inverse of regularization strength (Smaller = stronger regularization)
    # solver: Algorithm to use for optimization
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'newton-cg'], 
        'max_iter': [1000] # Ensure it converges
    }
    lr = LogisticRegression()
    
    # GridSearchCV tests all combinations and finds the best one using Cross-Validation
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters found: {grid_search.best_params_}")
    print(f"Best Cross-Val Accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_decision_tree(X_train, y_train, feature_names):
    """
    Trains Decision Tree with Hyperparameter Tuning
    """
    print("\n--- Training Decision Tree (Full Power) ---")
    
    # Tuning Depth to prevent overfitting but allow learning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters found: {grid_search.best_params_}")
    print(f"Best Cross-Val Accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    # 1. Load Data
    X_train, X_test, y_train, y_test, features = load_data()
    
    # 2. Train & Eval Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # 3. Train & Eval Decision Tree
    dt_model = train_decision_tree(X_train, y_train, features)
    dt_acc = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    
    # 4. Summary
    print("\n--- Milestone 2 Summary ---")
    print(f"Logistic Regression Accuracy: {lr_acc:.2%}")
    print(f"Decision Tree Accuracy:       {dt_acc:.2%}")
    
    # Save best baseline model for now
    best_model = lr_model if lr_acc > dt_acc else dt_model
    with open(os.path.join(OUTPUT_DIR, 'best_baseline_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    print("Saved the better of the two models to 'best_baseline_model.pkl'")