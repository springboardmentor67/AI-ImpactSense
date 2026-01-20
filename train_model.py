"""
Train and save the earthquake impact prediction model
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import joblib
import os

# Download and prepare data
print("Downloading dataset...")
# Try multiple sources for earthquake data
try:
    df = pd.read_csv("https://raw.githubusercontent.com/anushkadhiman/AI-impactSense/main/dataset/earthquake.csv")
except:
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/holtzy/Data_to_Viz/master/Story/earthquake/earthquake.csv")
    except:
        # Create sample data for testing
        print("Using sample data...")
        np.random.seed(42)
        n_samples = 1000
        # Generate features
        data = {
            'magnitude': np.random.uniform(3, 9, n_samples),
            'depth': np.random.uniform(5, 700, n_samples),
            'cdi': np.random.uniform(1, 10, n_samples),
            'mmi': np.random.uniform(1, 12, n_samples),
            'sig': np.random.uniform(10, 1000, n_samples),
        }
        df = pd.DataFrame(data)
        
        # Generate alert labels based on a risk calculation
        def calculate_alert(row):
            risk = (row['magnitude'] / 9 * 0.35 + 
                    (1 - row['depth'] / 700) * 0.15 + 
                    row['cdi'] / 10 * 0.20 + 
                    row['mmi'] / 12 * 0.20 + 
                    row['sig'] / 1000 * 0.10)
            if risk < 0.35:
                return 'green'
            elif risk < 0.55:
                return 'yellow'
            elif risk < 0.75:
                return 'orange'
            else:
                return 'red'
        
        df['alert'] = df.apply(calculate_alert, axis=1)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Clean data
df = df.drop_duplicates()
df['alert'] = df['alert'].astype(str).str.strip()

# Prepare features and target
X = df[['magnitude', 'depth', 'cdi', 'mmi', 'sig']]
y = df['alert']

# Encode labels
label_mapping = {'green': 0, 'yellow': 3, 'orange': 1, 'red': 2}
y_encoded = y.map(label_mapping)

print(f"Features: {X.columns.tolist()}")
print(f"Label mapping: {label_mapping}")
print(f"Class distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Hyperparameter tuning
print("\nTraining Random Forest with hyperparameter tuning...")
param_dist = {
    "n_estimators": [200, 300, 500],
    "max_depth": [20, 30, 40, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "bootstrap": [True],
}

rf = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1_weighted',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

final_model = rf_search.best_estimator_
print(f"Best parameters: {rf_search.best_params_}")

# Evaluate
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Save model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "earthquake_impact_rf.pkl")
feature_path = os.path.join(script_dir, "feature_order.pkl")

joblib.dump(final_model, model_path)
feature_order = X_train.columns.tolist()
joblib.dump(feature_order, feature_path)

print(f"\nModel saved to: {model_path}")
print(f"Feature order saved to: {feature_path}")
print(f"Feature order: {feature_order}")

