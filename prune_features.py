import pandas as pd
from pathlib import Path

# Configuration: The Mathematical Threshold
# A feature must achieve at least 0.001 Information Gain for AT LEAST ONE target to survive.
IG_THRESHOLD = 0.001 

FEATURES_INPUT_PATH = Path('data/processed/features_engineered.csv')
IG_MATRIX_PATH = Path('all_targets_ig_matrix.csv')
FEATURES_OUTPUT_PATH = Path('data/processed/features_pruned.csv')

def main():
    print("=" * 70)
    print(f"DYNAMIC THRESHOLD PRUNING (Minimum IG: {IG_THRESHOLD})")
    print("=" * 70)

    # 1. Load the data
    if not FEATURES_INPUT_PATH.exists() or not IG_MATRIX_PATH.exists():
        print("Error: Required input files not found.")
        return

    features_df = pd.read_csv(FEATURES_INPUT_PATH)
    ig_matrix = pd.read_csv(IG_MATRIX_PATH)
    
    # 2. Calculate the Maximum Information Gain for each feature across ALL targets
    ig_values = ig_matrix.drop(columns=['Feature_Name'])
    ig_matrix['Max_IG_Across_Targets'] = ig_values.max(axis=1)

    # 3. DYNAMIC FILTERING: Find all features below the threshold
    features_to_drop = ig_matrix[ig_matrix['Max_IG_Across_Targets'] < IG_THRESHOLD].sort_values(by='Max_IG_Across_Targets')
    drop_list = features_to_drop['Feature_Name'].tolist()

    print(f"\nIdentified {len(drop_list)} features failing to meet the {IG_THRESHOLD} threshold:")
    print("-" * 70)
    for _, row in features_to_drop.iterrows():
        print(f"- {row['Feature_Name']}: Max IG = {row['Max_IG_Across_Targets']:.6f}")
    print("-" * 70)

    # 4. Drop the features from the main dataset
    actual_drops = [col for col in drop_list if col in features_df.columns]
    features_pruned = features_df.drop(columns=actual_drops)

    # 5. Save the new dataset
    features_pruned.to_csv(FEATURES_OUTPUT_PATH, index=False)
    
    print(f"\nSUCCESS: Pruned dataset saved to {FEATURES_OUTPUT_PATH}")
    print(f"Original features: {features_df.shape[1]}")
    print(f"Final features retained: {features_pruned.shape[1]}")
    print(f"Total features dynamically dropped: {len(actual_drops)}")
    print("=" * 70)

if __name__ == "__main__":
    main()