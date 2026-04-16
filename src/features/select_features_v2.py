import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

def calculate_elbow(scores, feature_names, threshold=0.90):
    """
    Applies the Cumulative Elbow Method silently for loop iterations.
    """
    df_scores = pd.DataFrame({'Feature': feature_names, 'Score': scores})
    df_scores = df_scores.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    total_score = df_scores['Score'].sum()
    if total_score == 0:
        return []
        
    df_scores['Normalized_Score'] = df_scores['Score'] / total_score
    df_scores['Cumulative_Importance'] = df_scores['Normalized_Score'].cumsum()
    
    try:
        cutoff_idx = df_scores[df_scores['Cumulative_Importance'] >= threshold].index[0]
        selected_features = df_scores.iloc[:cutoff_idx + 1]['Feature'].tolist()
    except IndexError:
        selected_features = df_scores['Feature'].tolist()
        
    return selected_features

def calculate_vif(X, threshold=5.0):
    """
    Calculates Variance Inflation Factor iteratively to remove multi-collinearity.
    """
    print("\n[PHASE 1B] Executing VIF Multi-Collinearity Purge...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    sample_X = X_scaled.sample(n=min(10000, len(X_scaled)), random_state=42)
    dropped_features = []
    
    while True:
        vif_data = [variance_inflation_factor(sample_X.values, i) for i in range(sample_X.shape[1])]
        max_vif = max(vif_data)
        
        if max_vif > threshold:
            max_idx = vif_data.index(max_vif)
            feature_to_drop = sample_X.columns[max_idx]
            dropped_features.append((feature_to_drop, max_vif))
            sample_X = sample_X.drop(columns=[feature_to_drop])
        else:
            print(f"   -> VIF Check Passed. Maximum remaining VIF: {max_vif:.2f} (Target < {threshold})")
            break
            
    print(f"   -> Dropped {len(dropped_features)} features due to overlapping variance.")
    return sample_X.columns.tolist()

def main():
    print("=" * 70)
    print("STAGE 1: TARGET-SPECIFIC FEATURE AGGREGATION (THE UNION METHOD)")
    print("=" * 70)
    
    # 1. Load Data
    data_path = Path('data/processed/features_engineered_v2.csv')
    df = pd.read_csv(data_path)
    
    id_cols = ['subject_id', 'hadm_id', 'stay_id']
    
    # Note: We exclude 'los_days' (Regression) and use 'los_category' to keep all math Classification-based
    target_cols = [
        'mortality', 'los_category', 'icu_readmit_48h', 'icu_readmit_7d', 
        'discharge_disposition', 'need_vent_any', 'need_vasopressor_any', 'need_rrt_any',
        'aki_onset', 'ards_onset', 'liver_injury_onset', 'sepsis_onset'
    ]
    
    all_original_targets = target_cols + ['los_days'] 
    X_raw = df.drop(columns=id_cols + all_original_targets, errors='ignore')
    
    print(f"[METRIC] Starting Predictor Count: {X_raw.shape[1]}")
    
    # Temporary Imputation for Math computations
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
    
    # ---------------------------------------------------------
    # PHASE 1: The Redundancy Purge
    # ---------------------------------------------------------
    print("\n[PHASE 1A] Executing Spearman Correlation Filter (Threshold: 0.85)...")
    corr_matrix = X_imputed.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.85)]
    X_imputed = X_imputed.drop(columns=to_drop_corr)
    print(f"   -> Dropped {len(to_drop_corr)} pairwise duplicates.")
    
    independent_features = calculate_vif(X_imputed, threshold=5.0)
    X_clean_imputed = X_imputed[independent_features]
    
    print(f"\n[METRIC] Independent Feature Base Established: {len(independent_features)} columns remain.")
    
    # ---------------------------------------------------------
    # PHASE 2: The 12-Target Evaluation Loop
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: THE 12-TARGET EVALUATION LOOP")
    print("Estimated Time: 10-15 Minutes. Please wait...")
    print("=" * 70)
    
    # Initialize empty sets for the Union Method
    union_ig = set()
    union_anova = set()
    union_mi = set()
    union_lasso = set()
    
    scaler = StandardScaler()
    X_clean_scaled = scaler.fit_transform(X_clean_imputed)
    
    for i, target in enumerate(target_cols, 1):
        print(f"   [{i}/12] Evaluating mathematical markers for: {target.upper()}")
        
        # Isolate target and drop NaNs just for the scoring algorithm
        target_data = df[[target]].copy()
        valid_idx = target_data[target].dropna().index
        
        y_valid = target_data.loc[valid_idx, target]
        X_valid = X_clean_imputed.loc[valid_idx]
        X_valid_scaled = X_clean_scaled[valid_idx]
        
        # Encode string targets (like discharge_disposition) to numeric for sklearn
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_valid)
        
        # 1. Information Gain
        dt = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=10)
        dt.fit(X_valid, y_encoded)
        ig_feats = calculate_elbow(dt.feature_importances_, independent_features)
        union_ig.update(ig_feats)
        
        # 2. ANOVA
        f_scores, _ = f_classif(X_valid, y_encoded)
        f_scores = np.nan_to_num(f_scores)
        anova_feats = calculate_elbow(f_scores, independent_features)
        union_anova.update(anova_feats)
        
        # 3. Mutual Information
        mi_scores = mutual_info_classif(X_valid, y_encoded, random_state=42)
        mi_feats = calculate_elbow(mi_scores, independent_features)
        union_mi.update(mi_feats)
        
        # 4. LASSO
        lasso = LogisticRegression(penalty='l1', solver='saga', C=0.1, random_state=42, max_iter=500, n_jobs=-1)
        lasso.fit(X_valid_scaled, y_encoded)
        
        # Handle multi-class LASSO coefficients (take max absolute weight across classes)
        if len(lasso.coef_.shape) > 1 and lasso.coef_.shape[0] > 1:
            lasso_weights = np.max(np.abs(lasso.coef_), axis=0)
        else:
            lasso_weights = np.abs(lasso.coef_[0])
            
        lasso_feats = calculate_elbow(lasso_weights, independent_features)
        union_lasso.update(lasso_feats)

    # ---------------------------------------------------------
    # PHASE 3: Feature Set Union & Deduplication
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: UNION DEDUPLICATION & EXPORT")
    print("=" * 70)
    
    final_ig = list(union_ig)
    final_anova = list(union_anova)
    final_mi = list(union_mi)
    final_lasso = list(union_lasso)
    
    print(f"   -> IG Global Matrix Size: {len(final_ig)} features")
    print(f"   -> ANOVA Global Matrix Size: {len(final_anova)} features")
    print(f"   -> MI Global Matrix Size: {len(final_mi)} features")
    print(f"   -> LASSO Global Matrix Size: {len(final_lasso)} features")
    
    out_dir = Path('data/processed/tournament')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_base = df[id_cols + all_original_targets]
    
    pd.concat([df_base, X_raw[final_ig]], axis=1).to_csv(out_dir / 'X_ig_union.csv', index=False)
    pd.concat([df_base, X_raw[final_anova]], axis=1).to_csv(out_dir / 'X_anova_union.csv', index=False)
    pd.concat([df_base, X_raw[final_mi]], axis=1).to_csv(out_dir / 'X_mi_union.csv', index=False)
    pd.concat([df_base, X_raw[final_lasso]], axis=1).to_csv(out_dir / 'X_lasso_union.csv', index=False)
    
    print(f"\n[SUCCESS] Exported 4 Multi-Target matrices to: {out_dir.absolute()}")
    print("The pipeline is now complete. Ready to begin the Baseline Model Tournament.")

if __name__ == "__main__":
    main()