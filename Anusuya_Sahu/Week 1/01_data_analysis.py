import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
DATA_PATH = '../dataset.csv'
OUTPUT_DIR = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_inspect_data():
    """
    Loads data and performs basic inspection.
    """
    print("--- Loading Data ---")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data Loaded Successfully. Shape: {df.shape}")
        
        # 1. Preview
        print("\nFirst 5 rows:")
        print(df.head())
        
        # 2. Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # 3. Check Data Types
        print("\nData Types:")
        print(df.dtypes)
        
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {DATA_PATH}. Please check folder structure.")
        return None

def visualize_target_distribution(df):
    """
    Visualizes how many Green, Yellow, Orange, Red alerts we have.
    Crucial for identifying Class Imbalance.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='alert', data=df, order=['green', 'yellow', 'orange', 'red'], palette='viridis')
    plt.title('Distribution of Earthquake Alerts (Target Variable)')
    plt.xlabel('Alert Level')
    plt.ylabel('Count')
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, '01_target_distribution.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def visualize_correlations(df):
    """
    Shows correlation between numerical features.
    Helps us see which features (mag, depth, mmi) drive the 'sig' score.
    """
    plt.figure(figsize=(10, 8))
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Plot heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    
    save_path = os.path.join(OUTPUT_DIR, '02_correlation_matrix.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def map_risk_space(df):
    """
    Since we don't have Lat/Long, we map the 'Physics Space'.
    We plot Magnitude vs Depth. 
    We color the dots by Alert Level.
    
    Hypothesis: High Mag + Shallow Depth = Red Alert.
    """
    plt.figure(figsize=(10, 6))
    
    # Map colors to alert types manually for consistency
    palette = {'green': 'green', 'yellow': '#FFD700', 'orange': 'orange', 'red': 'red'}
    
    sns.scatterplot(
        data=df, 
        x='magnitude', 
        y='depth', 
        hue='alert', 
        palette=palette,
        hue_order=['green', 'yellow', 'orange', 'red'],
        alpha=0.7,
        s=60
    )
    
    plt.title('Earthquake Risk Space: Magnitude vs. Depth')
    plt.xlabel('Magnitude')
    plt.ylabel('Depth (km)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Invert Y axis because depth is usually visualized downwards
    plt.gca().invert_yaxis()
    
    save_path = os.path.join(OUTPUT_DIR, '03_magnitude_vs_depth.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def analyze_feature_distributions(df):
    """
    Boxplots to see how 'mmi' (shaking intensity) differs per alert.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='alert', y='mmi', data=df, order=['green', 'yellow', 'orange', 'red'], palette='viridis')
    plt.title('Shaking Intensity (MMI) vs Alert Level')
    
    save_path = os.path.join(OUTPUT_DIR, '04_mmi_distribution.png')
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 1. Load
    df = load_and_inspect_data()
    
    if df is not None:
        # 2. Visualize Target (Class Imbalance check)
        visualize_target_distribution(df)
        
        # 3. Visualize Correlations
        visualize_correlations(df)
        
        # 4. Map the Physics (Mag vs Depth)
        map_risk_space(df)
        
        # 5. Feature Distribution
        analyze_feature_distributions(df)
        
        print("\n--- Week 1 Analysis Complete ---")
        print("Check the 'Week_1/output' folder for your graphs.")