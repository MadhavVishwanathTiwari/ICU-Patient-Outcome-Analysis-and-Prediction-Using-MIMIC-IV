"""
Phase 3: The Complete Matrix Tournament (With ROC Curves)
Evaluates 12 Clinical Targets across 4 Mathematical Matrices using 5 Models.
Outputs results in Markdown Tables and saves winning ROC curves as PNGs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import gc

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
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

REGRESSION_TARGET = 'los_days'

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
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob', num_class=n_classes, random_state=42, n_jobs=-1),
            'CatBoost': CatBoostClassifier(iterations=200, random_seed=42, verbose=0, loss_function='MultiClass')
        }

def build_custom_mlp(input_dim, task_type, n_classes):
    """Dynamically builds the Custom Deep Learning architecture"""
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
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    return model

def plot_winning_roc(target_name, model_name, matrix_name, best_auc, y_test, preds, task_type, n_classes):
    """Plots and saves the ROC curve for the winning configuration"""
    plt.figure(figsize=(8, 6))
    
    if task_type == 'binary':
        fpr, tpr, _ = roc_curve(y_test, preds)
        plt.plot(fpr, tpr, lw=2, color='darkorange', label=f'ROC curve (AUC = {best_auc:.4f})')
    else:
        # For Multiclass: plot One-vs-Rest ROC for each class
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], preds[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f'Class {i} (AUC = {class_auc:.4f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'Winning Configuration: {target_name.upper()}\n{model_name} on {matrix_name} Matrix', fontweight='bold', pad=15)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save the plot
    filepath = f"results/roc_curves/roc_{target_name}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def run_full_tournament():
    print("=" * 80)
    print("INITIALIZING 240-MODEL MATRIX TOURNAMENT (WITH ROC GENERATION)")
    print("=" * 80)

    data_dir = Path('data/processed/tournament')
    
    # Create directory for saving the plots
    os.makedirs('results/roc_curves', exist_ok=True)
    
    results = {t: {m: {mat: None for mat in MATRICES.keys()} for m in MODEL_NAMES} for t in TARGETS.keys()}
    all_target_cols = list(TARGETS.keys()) + [REGRESSION_TARGET]
    id_cols = ['subject_id', 'hadm_id', 'stay_id']

    for target_name, task_type in TARGETS.items():
        print(f"\n>> EVALUATING TARGET: {target_name.upper()} ({task_type})")
        
        # Tracking variables for the "King of the Hill" winning model
        best_auc = -1
        best_model_name = ""
        best_matrix_name = ""
        best_y_test = None
        best_preds = None
        best_n_classes = None
        
        for matrix_name, matrix_file in MATRICES.items():
            matrix_path = data_dir / matrix_file
            if not matrix_path.exists():
                continue
                
            df = pd.read_csv(matrix_path)
            df.columns = [col.replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt') for col in df.columns]

            if target_name not in df.columns:
                continue

            df_clean = df.dropna(subset=[target_name]).copy()
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(df_clean[target_name])
            n_classes = len(le.classes_)

            drop_cols = [c for c in id_cols + all_target_cols if c in df_clean.columns]
            X = df_clean.drop(columns=drop_cols)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)
            
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

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
                    auc_val = roc_auc_score(y_test, preds)
                else:
                    preds = model.predict_proba(X_test_scaled)
                    auc_val = roc_auc_score(y_test, preds, multi_class='ovr', average='macro')
                
                results[target_name][model_name][matrix_name] = auc_val
                
                # Check if this is the new champion
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_model_name = model_name
                    best_matrix_name = matrix_name
                    best_y_test = y_test.copy()
                    best_preds = preds.copy()
                    best_n_classes = n_classes

            # --- TRAIN CUSTOM MLP ---
            print(f"  [+] Training Custom MLP          on {matrix_name:5}...", end='\r')
            custom_model = build_custom_mlp(input_dim=X_train_scaled.shape[1], task_type=task_type, n_classes=n_classes)
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            custom_model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=0)
            
            mlp_preds = custom_model.predict(X_test_scaled, verbose=0)
            
            if task_type == 'binary':
                mlp_preds_clean = mlp_preds.flatten()
                mlp_auc = roc_auc_score(y_test, mlp_preds_clean)
            else:
                mlp_preds_clean = mlp_preds
                mlp_auc = roc_auc_score(y_test, mlp_preds_clean, multi_class='ovr', average='macro')
                
            results[target_name]['Custom MLP'][matrix_name] = mlp_auc
            
            # Check if Custom MLP is the new champion
            if mlp_auc > best_auc:
                best_auc = mlp_auc
                best_model_name = 'Custom MLP'
                best_matrix_name = matrix_name
                best_y_test = y_test.copy()
                best_preds = mlp_preds_clean.copy()
                best_n_classes = n_classes
            
            K.clear_session()
            gc.collect()
            
            print(f"  [*] {matrix_name} Matrix fully evaluated for {target_name}.    ")

        # --- END OF TARGET LOOP: PLOT THE WINNER ---
        if best_preds is not None:
            print(f"\n  🏆 Plotting winning configuration for {target_name}: {best_model_name} on {best_matrix_name}")
            plot_winning_roc(target_name, best_model_name, best_matrix_name, best_auc, best_y_test, best_preds, task_type, best_n_classes)

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