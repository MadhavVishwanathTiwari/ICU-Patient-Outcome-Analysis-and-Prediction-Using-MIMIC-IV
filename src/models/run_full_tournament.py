"""
Phase 3: The Complete Matrix Tournament
Evaluates 12 Clinical Targets across 4 Mathematical Matrices using 5 Models.
Outputs results in Markdown Tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# Suppress TF logging clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Define the 12 Targets and their Classification Types
TARGETS = {
    'mortality': 'binary',
    'aki_onset': 'binary',
    'sepsis_onset': 'binary',
    'ards_onset': 'binary',
    'liver_injury_onset': 'binary',
    'need_vent_any': 'binary',
    'need_vasopressor_any': 'binary',
    'need_rrt_any': 'binary',
    'icu_readmit_48h': 'binary',
    'icu_readmit_7d': 'binary',
    'los_category': 'multiclass',
    'discharge_disposition': 'multiclass'
}

# The 13th target (regression) that must also be dropped to prevent leakage
REGRESSION_TARGET = 'los_days'

# 2. Define the Matrices mapping
MATRICES = {
    'IG': 'X_ig_union.csv',
    'ANOVA': 'X_anova_union.csv',
    'MI': 'X_mi_union.csv',
    'LASSO': 'X_lasso_union.csv'
}

MODEL_NAMES = ['CatBoost', 'XGBoost', 'Random Forest', 'Logistic Regression', 'Custom MLP']

def get_baseline_models(task_type, n_classes):
    """Dynamically loads baseline models depending on Binary vs Multiclass targets"""
    if task_type == 'binary':
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
            'CatBoost': CatBoostClassifier(iterations=200, random_seed=42, verbose=0, auto_class_weights='Balanced')
        }
    else:
        return {
            # Fix: Removed the deprecated multi_class parameter
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob', num_class=n_classes, random_state=42, n_jobs=-1),
            'CatBoost': CatBoostClassifier(iterations=200, random_seed=42, verbose=0, loss_function='MultiClass')
        }

def build_custom_mlp(input_dim, task_type, n_classes):
    """Dynamically builds the Custom Deep Learning architecture for the specific target"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu')
    ])
    
    if task_type == 'binary':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    else:
        # Multiclass requires Softmax and Sparse Categorical Crossentropy
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    return model

def run_full_tournament():
    print("=" * 80)
    print("INITIALIZING 240-MODEL MATRIX TOURNAMENT")
    print("=" * 80)

    data_dir = Path('data/processed/tournament')
    
    # Initialize the nested results dictionary
    results = {t: {m: {mat: None for mat in MATRICES.keys()} for m in MODEL_NAMES} for t in TARGETS.keys()}

    # All targets that must be dropped from X to prevent Target Leakage
    all_target_cols = list(TARGETS.keys()) + [REGRESSION_TARGET]
    id_cols = ['subject_id', 'hadm_id', 'stay_id']

    for target_name, task_type in TARGETS.items():
        print(f"\n>> EVALUATING TARGET: {target_name.upper()} ({task_type})")
        
        for matrix_name, matrix_file in MATRICES.items():
            matrix_path = data_dir / matrix_file
            if not matrix_path.exists():
                continue
                
            df = pd.read_csv(matrix_path)
            
            # Clean XGBoost illegal characters
            df.columns = [col.replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt') for col in df.columns]

            if target_name not in df.columns:
                print(f"  [-] {matrix_name}: Target missing, skipping.")
                continue

            # Drop rows where the *target outcome itself* is missing
            df_clean = df.dropna(subset=[target_name]).copy()
            
            # Label Encode Y (Required for Multiclass, safe for Binary)
            le = LabelEncoder()
            y_encoded = le.fit_transform(df_clean[target_name])
            n_classes = len(le.classes_)

            # Prevent Target Leakage by dropping IDs and all targets
            drop_cols = [c for c in id_cols + all_target_cols if c in df_clean.columns]
            X = df_clean.drop(columns=drop_cols)
            
            # 1. Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)
            
            # 2. IMPUTE MISSING CLINICAL VALUES
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            # 3. Scale Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_scaled = scaler.transform(X_test_imp)

            # --- TRAIN BASELINES ---
            baselines = get_baseline_models(task_type, n_classes)
            for model_name, model in baselines.items():
                print(f"  [+] Training {model_name:20} on {matrix_name:5}...", end='\r')
                model.fit(X_train_scaled, y_train)
                
                if task_type == 'binary':
                    preds = model.predict_proba(X_test_scaled)[:, 1]
                    auc = roc_auc_score(y_test, preds)
                else:
                    preds = model.predict_proba(X_test_scaled)
                    auc = roc_auc_score(y_test, preds, multi_class='ovr', average='macro')
                
                results[target_name][model_name][matrix_name] = auc

            # --- TRAIN CUSTOM MLP ---
            print(f"  [+] Training Custom MLP          on {matrix_name:5}...", end='\r')
            custom_model = build_custom_mlp(input_dim=X_train_scaled.shape[1], task_type=task_type, n_classes=n_classes)
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            custom_model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=0)
            
            mlp_preds = custom_model.predict(X_test_scaled, verbose=0)
            if task_type == 'binary':
                mlp_auc = roc_auc_score(y_test, mlp_preds.flatten())
            else:
                mlp_auc = roc_auc_score(y_test, mlp_preds, multi_class='ovr', average='macro')
                
            results[target_name]['Custom MLP'][matrix_name] = mlp_auc
            
            # Clear RAM after heavy deep learning iteration
            K.clear_session()
            gc.collect()
            
            print(f"  [*] {matrix_name} Matrix fully evaluated for {target_name}.    ")

    # 3. Output the Final Markdown Tables
    print("\n" + "=" * 80)
    print("TOURNAMENT RESULTS (MARKDOWN TABLES)")
    print("=" * 80)
    
    for target_name in TARGETS.keys():
        print(f"\n### {target_name.upper().replace('_', ' ')}")
        print("| Models | IG | ANOVA | MI | LASSO |")
        print("| :--- | :--- | :--- | :--- | :--- |")
        for model in MODEL_NAMES:
            row = f"| **{model}** "
            for matrix_name in ['IG', 'ANOVA', 'MI', 'LASSO']:
                score = results[target_name][model][matrix_name]
                if score is not None:
                    row += f"| {score:.4f} |"
                else:
                    row += "| N/A |"
            print(row)

if __name__ == "__main__":
    run_full_tournament()