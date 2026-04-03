import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from pathlib import Path

# 1. Load the engineered features
features_path = Path('data/processed/features_engineered.csv')

if not features_path.exists():
    print(f"Error: {features_path} not found.")
    exit()

print("Loading dataset...")
df = pd.read_csv(features_path)

# 2. Define IDs and all Targets
id_cols = ['subject_id', 'hadm_id', 'stay_id']
target_cols = [
    'mortality', 'los_days', 'los_category', 'icu_readmit_48h', 
    'icu_readmit_7d', 'discharge_disposition', 'need_vent_any', 
    'need_vasopressor_any', 'need_rrt_any', 'aki_onset', 
    'ards_onset', 'liver_injury_onset', 'sepsis_onset'
]

# Extract the 112 Input Features (X)
X = df.drop(columns=[c for c in id_cols + target_cols if c in df.columns])

# 3. Initialize a DataFrame to hold the results
ig_matrix = pd.DataFrame({
    'Feature_Name': X.columns
})

print(f"Calculating Information Gain for {X.shape[1]} features across {X.shape[0]} samples...\n")

# 4. Loop through every single target and calculate IG
for target in target_cols:
    if target not in df.columns:
        continue
        
    print(f" -> Processing target: {target}")
    y = df[target]
    
    # Drop any rows where this specific target might be NaN
    valid_idx = y.dropna().index
    X_valid = X.loc[valid_idx]
    y_valid = y.loc[valid_idx]
    
    # Use regression for continuous days, classification for binary/multi-class
    if target == 'los_days':
        ig_values = mutual_info_regression(X_valid, y_valid, random_state=42)
    else:
        ig_values = mutual_info_classif(X_valid, y_valid, discrete_features=True, random_state=42)
        
    # Add the results as a new column in our matrix
    ig_matrix[f"{target}_IG"] = ig_values

# 5. Save the massive matrix
output_file = 'all_targets_ig_matrix.csv'
ig_matrix.to_csv(output_file, index=False)

print(f"\nSuccess! Full matrix saved to {output_file}")
print("You can now open this in Excel to see how feature importance shifts per target.")