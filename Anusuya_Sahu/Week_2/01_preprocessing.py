import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# --- CONFIGURATION ---
DATA_PATH = '../dataset.csv'
OUTPUT_DIR = 'output'

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Data Loaded. Shape: {df.shape}")
    return df

def feature_engineering(df):
    """
    Creates new features based on seismic physics.
    """
    print("\n--- Feature Engineering ---")
    
    # 1. Seismic Energy (Joule approximation)
    # Formula: log E = 5.24 + 1.44M (simplified version for ML)
    # We use a power transform to make it linear-ish
    df['energy_release'] = 10 ** (1.5 * df['magnitude'])
    print("Created feature: 'energy_release'")
    
    # 2. Impact Factor
    # High magnitude + Low depth = High Risk. 
    # We add 1 to depth to avoid division by zero.
    df['impact_factor'] = df['magnitude'] / (np.log1p(df['depth']))
    print("Created feature: 'impact_factor'")
    
    # 3. Shallow Flag (Boolean)
    # Earthquakes < 70km are considered shallow and dangerous
    df['is_shallow'] = (df['depth'] < 70).astype(int)
    print("Created feature: 'is_shallow'")
    
    return df

def preprocess_and_save(df):
    print("\n--- Preprocessing ---")
    
    # 1. Encode Target (Alert)
    # We map them manually to ensure the order is correct (Ordinal)
    alert_mapping = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}
    df['alert_encoded'] = df['alert'].map(alert_mapping)
    print(f"Mapped Alerts: {alert_mapping}")
    
    # Separate Features (X) and Target (y)
    # Drop original 'alert' and the encoded target from X
    X = df.drop(['alert', 'alert_encoded'], axis=1)
    y = df['alert_encoded']
    
    # 2. Split Data (80% Train, 20% Test)
    # Stratify ensures we keep the perfect balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    # 3. Scaling
    # We fit the scaler ONLY on training data to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data Scaled using StandardScaler.")
    
    # --- SAVE ARTIFACTS ---
    print("\n--- Saving Artifacts ---")
    
    # Save the processed arrays (efficient for ML)
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train.to_numpy())
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test.to_numpy())
    
    # Save column names (for visualization later)
    with open(os.path.join(OUTPUT_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
        
    # Save the scaler (needed for the final UI)
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"All files saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    # 1. Load
    df = load_data()
    
    # 2. Engineer Features
    df = feature_engineering(df)
    
    # 3. Preprocess & Save
    preprocess_and_save(df)