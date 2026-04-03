import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path

# 1. Load the engineered features
# Path is based on your project structure
features_path = Path('data/processed/features_engineered.csv')

if not features_path.exists():
    print(f"Error: {features_path} not found. Please run build_features.py first.")
    exit()

df = pd.read_csv(features_path)

# 2. Define the Target and Features
# We exclude the identifiers and all potential targets
id_cols = ['subject_id', 'hadm_id', 'stay_id']
target_cols = [
    'mortality', 'los_days', 'los_category', 'icu_readmit_48h', 
    'icu_readmit_7d', 'discharge_disposition', 'need_vent_any', 
    'need_vasopressor_any', 'need_rrt_any', 'aki_onset', 
    'ards_onset', 'liver_injury_onset', 'sepsis_onset'
]

X = df.drop(columns=[c for c in id_cols + target_cols if c in df.columns])
y = df['mortality'] # Calculating IG relative to Mortality

print(f"Calculating Information Gain for {X.shape[1]} features across {X.shape[0]} samples...")

# 3. Calculate Information Gain (Mutual Information)
# discrete_features=True because most of your 112 features are binary flags
importances = mutual_info_classif(X, y, discrete_features=True, random_state=42)

# 4. Create and Save Results
ig_results = pd.DataFrame({
    'Feature_Index': range(1, len(X.columns) + 1),
    'Feature_Name': X.columns,
    'Information_Gain': importances
})

# Sort by importance
ig_results = ig_results.sort_values(by='Information_Gain', ascending=False)

# Save to CSV for your defense
output_file = 'feature_information_gain.csv'
ig_results.to_csv(output_file, index=False)

print(f"\nSuccess! IG values saved to {output_file}")
print("\nTop 10 Features by Information Gain:")
print(ig_results.head(10).to_string(index=False))