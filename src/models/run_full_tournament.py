"""
Phase 3: The Complete Matrix Tournament (Optimized I/O & Architecture)
Evaluates 12 Clinical Targets across 4 Mathematical Matrices using 8 Models.

Models  : CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression,
          Custom MLP, FT-Transformer, Stacking Ensemble → 12 × 8 × 4 = 384 combinations
Outputs : Markdown tables + winning ROC curves saved as PNGs
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
import gc
import json

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.features.leakage_rules import drop_organ_support_leaky_columns
from src.models.custom_mlp import build_custom_mlp, compute_class_weights
from src.models.stacking_model import run_stacking_for_tournament
from src.models.ft_transformer import build_ftt, train_ftt, predict_ftt, get_device

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TARGETS = {
    'mortality': 'binary', 'aki_onset': 'binary', 'sepsis_onset': 'binary',
    'ards_onset': 'binary', 'liver_injury_onset': 'binary', 'need_vent_any': 'binary',
    'need_vasopressor_any': 'binary', 'need_rrt_any': 'binary', 'icu_readmit_48h': 'binary',
    'icu_readmit_7d': 'binary', 'los_category': 'multiclass', 'discharge_disposition': 'multiclass'
}

MATRICES = {
    'IG': 'X_ig_union.csv', 'ANOVA': 'X_anova_union.csv',
    'MI': 'X_mi_union.csv', 'LASSO': 'X_lasso_union.csv'
}

# FT-Transformer added as 8th model → 12 × 8 × 4 = 384 total combinations
MODEL_NAMES = [
    'CatBoost', 'XGBoost', 'LightGBM', 'Random Forest',
    'Logistic Regression', 'Custom MLP', 'FT-Transformer', 'Stacking Ensemble'
]


def _normalize_results(results: dict) -> dict:
    """
    Back-fills any target / model / matrix keys that are missing from a
    previously-saved JSON file.  This happens whenever a new model (e.g.
    'Stacking Ensemble') or a new matrix is added after some results have
    already been checkpointed.  Missing slots are initialised to None so
    downstream write operations never raise a KeyError.
    """
    for target in TARGETS:
        if target not in results:
            results[target] = {}
        for model in MODEL_NAMES:
            if model not in results[target]:
                results[target][model] = {}
            for matrix in MATRICES:
                if matrix not in results[target][model]:
                    results[target][model][matrix] = None
    return results


def get_baseline_models(task_type, n_classes, y_train):
    if task_type == 'binary':
        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)
        spw = neg / pos if pos > 0 else 1.0

        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
            'XGBoost':             xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=spw, random_state=42, n_jobs=-1),
            'LightGBM':            lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
            'CatBoost':            CatBoostClassifier(iterations=200, random_seed=42, verbose=0, auto_class_weights='Balanced')
        }
    else:
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
            'XGBoost':             xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob', num_class=n_classes, random_state=42, n_jobs=-1),
            'LightGBM':            lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
            'CatBoost':            CatBoostClassifier(iterations=200, random_seed=42, verbose=0, loss_function='MultiClass')
        }


def plot_winning_roc(target_name, model_name, matrix_name, best_auc,
                     y_test, preds, task_type, n_classes):
    plt.figure(figsize=(8, 6))
    if task_type == 'binary':
        fpr, tpr, _ = roc_curve(y_test, preds)
        plt.plot(fpr, tpr, lw=2, color='darkorange', label=f'ROC curve (AUC = {best_auc:.4f})')
    else:
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], preds[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f'Class {i} (AUC = {class_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(
        f'Winning Configuration: {target_name.upper()}\n'
        f'{model_name} on {matrix_name} Matrix',
        fontweight='bold', pad=15
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    filepath = f"results/roc_curves/roc_{target_name}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def run_full_tournament():
    print("=" * 80)
    print("INITIALIZING MATRIX TOURNAMENT  (8 models × 4 matrices × 12 targets = 384 slots)")
    print("=" * 80)

    device = get_device()   # CUDA → MPS → CPU; resolved once, reused across all slots

    data_dir = Path('data/processed/tournament')
    out_dir  = Path('results/roc_curves')
    out_dir.mkdir(parents=True, exist_ok=True)

    results_file = Path('results/tournament_scores.json')
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        print("  [+] Loaded previous tournament scores from JSON.")
        # ── FIX: back-fill any keys that didn't exist when the JSON was saved
        # (e.g. 'Stacking Ensemble' added after an earlier partial run).
        results = _normalize_results(results)
    else:
        results = {
            t: {m: {mat: None for mat in MATRICES.keys()} for m in MODEL_NAMES}
            for t in TARGETS.keys()
        }

    # ── I/O OPTIMISATION: load all matrices once ─────────────────────────────
    print("\n  [!] Pre-loading Matrices into Memory...")
    loaded_matrices = {}
    for matrix_name, matrix_file in MATRICES.items():
        matrix_path = data_dir / matrix_file
        if matrix_path.exists():
            df = pd.read_csv(matrix_path)
            df.columns = [
                col.replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt')
                for col in df.columns
            ]
            loaded_matrices[matrix_name] = df
    print(f"  [+] Loaded {len(loaded_matrices)} matrices.\n")

    all_target_cols = list(TARGETS.keys()) + ['los_days']
    id_cols = ['subject_id', 'hadm_id', 'stay_id']

    for target_name, task_type in TARGETS.items():
        print(f">> EVALUATING TARGET: {target_name.upper()} ({task_type})")

        expected_plot_path = out_dir / f"roc_{target_name}.png"
        if expected_plot_path.exists():
            print(f"  [skip] already complete — delete {expected_plot_path.name} to re-run.\n")
            continue

        best_auc, best_model_name, best_matrix_name = -1, "", ""
        best_y_test, best_preds, best_n_classes = None, None, None

        for matrix_name, df_master in loaded_matrices.items():
            if target_name not in df_master.columns:
                continue

            df_clean = df_master.dropna(subset=[target_name]).copy()
            le = LabelEncoder()
            y_encoded = le.fit_transform(df_clean[target_name])
            n_classes = len(le.classes_)

            if task_type == 'binary' and n_classes > 2:
                print(f"  [!] TYPE GUARD: Skipping {target_name} — unexpected class count.")
                continue

            drop_cols = [c for c in id_cols + all_target_cols if c in df_clean.columns]
            X = df_clean.drop(columns=drop_cols)
            X = drop_organ_support_leaky_columns(X, target_name)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
            )

            imputer = SimpleImputer(strategy='median')
            scaler  = StandardScaler()
            X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
            X_test_scaled  = scaler.transform(imputer.transform(X_test))

            # ── Baseline models (unchanged) ───────────────────────────────
            baselines = get_baseline_models(task_type, n_classes, y_train)
            for model_name, model in baselines.items():
                print(f"  [+] Training {model_name:20} on {matrix_name:5}...", end='\r')

                if model_name == 'XGBoost' and task_type == 'multiclass':
                    weights = compute_sample_weight('balanced', y_train)
                    model.fit(X_train_scaled, y_train, sample_weight=weights)
                else:
                    model.fit(X_train_scaled, y_train)

                preds = (
                    model.predict_proba(X_test_scaled)[:, 1]
                    if task_type == 'binary'
                    else model.predict_proba(X_test_scaled)
                )
                auc_val = (
                    roc_auc_score(y_test, preds)
                    if task_type == 'binary'
                    else roc_auc_score(y_test, preds, multi_class='ovr', average='macro')
                )

                if model_name not in results[target_name]:
                    results[target_name][model_name] = {}
                results[target_name][model_name][matrix_name] = auc_val

                if auc_val > best_auc:
                    best_auc, best_model_name, best_matrix_name = auc_val, model_name, matrix_name
                    best_y_test, best_preds, best_n_classes = y_test.copy(), preds.copy(), n_classes

            # ── Custom MLP ────────────────────────────────────────────────
            print(f"  [+] Training Custom MLP          on {matrix_name:5}...", end='\r')
            cw = compute_class_weights(y_train)
            custom_model = build_custom_mlp(
                input_dim=X_train_scaled.shape[1], task_type=task_type, n_classes=n_classes
            )
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            custom_model.fit(
                X_train_scaled, y_train, epochs=100, batch_size=64,
                validation_split=0.2, callbacks=[early_stop],
                class_weight=cw, verbose=0
            )
            mlp_preds = custom_model.predict(X_test_scaled, verbose=0)
            mlp_preds_clean = mlp_preds.flatten() if task_type == 'binary' else mlp_preds
            mlp_auc = (
                roc_auc_score(y_test, mlp_preds_clean)
                if task_type == 'binary'
                else roc_auc_score(y_test, mlp_preds_clean, multi_class='ovr', average='macro')
            )
            results[target_name]['Custom MLP'][matrix_name] = mlp_auc
            if mlp_auc > best_auc:
                best_auc, best_model_name, best_matrix_name = mlp_auc, 'Custom MLP', matrix_name
                best_y_test, best_preds, best_n_classes = y_test.copy(), mlp_preds_clean.copy(), n_classes

            K.clear_session()
            gc.collect()

            # ── FT-Transformer (8th standalone competitor) ────────────────
            # Rebuilt fresh per (target, matrix) slot — no weight sharing.
            # GPU memory freed immediately after prediction via del + gc.
            print(f"  [+] Training FT-Transformer      on {matrix_name:5}...", end='\r')
            try:
                ftt_model = build_ftt(
                    input_dim=X_train_scaled.shape[1],
                    task_type=task_type,
                    n_classes=n_classes,
                )
                ftt_model = train_ftt(
                    model=ftt_model,
                    X_tr=X_train_scaled,
                    y_tr=y_train,
                    device=device,
                    epochs=50,
                    batch_size=256,
                    patience=5,
                    val_fraction=0.15,
                )
                ftt_preds = predict_ftt(ftt_model, X_test_scaled, device, task_type)
                ftt_auc = (
                    roc_auc_score(y_test, ftt_preds)
                    if task_type == 'binary'
                    else roc_auc_score(y_test, ftt_preds, multi_class='ovr', average='macro')
                )
                results[target_name]['FT-Transformer'][matrix_name] = ftt_auc
                if ftt_auc > best_auc:
                    best_auc, best_model_name, best_matrix_name = ftt_auc, 'FT-Transformer', matrix_name
                    best_y_test, best_preds, best_n_classes = y_test.copy(), ftt_preds.copy(), n_classes
            except Exception as exc:
                print(f"\n  [!] FT-Transformer failed for {target_name}/{matrix_name}: {exc}")
                results[target_name]['FT-Transformer'][matrix_name] = None
            finally:
                # Free GPU/CPU memory regardless of success or failure
                try:
                    del ftt_model
                except NameError:
                    pass
                gc.collect()
                if device.type == 'cuda':
                    import torch
                    torch.cuda.empty_cache()

            # ── Stacking Ensemble (8th competitor, was 7th) ───────────────
            # Level 0: CB, XGB, LGB, RF, LR, MLP  (6 base learners)
            # Level 1: Logistic Regression meta-learner
            # MLP at Level 0 adds neural diversity alongside the 5 tree/linear
            # models — Level 1 can exploit the blind-spot coverage.
            # Runtime: ~5-fold OOF × 6 base learners → estimate 5–10 min/slot.
            print(f"  [+] Training Stacking Ensemble   on {matrix_name:5}...", end='\r')
            try:
                stack_auc, stack_preds, _, _, _, _ = run_stacking_for_tournament(
                    X_train_scaled, X_test_scaled,
                    y_train, y_test,
                    task_type, n_classes,
                    input_dim=X_train_scaled.shape[1]   # required to build MLP graph
                )
                results[target_name]['Stacking Ensemble'][matrix_name] = stack_auc
                if stack_auc > best_auc:
                    best_auc         = stack_auc
                    best_model_name  = 'Stacking Ensemble'
                    best_matrix_name = matrix_name
                    best_y_test      = y_test.copy()
                    best_preds       = stack_preds.copy()
                    best_n_classes   = n_classes
            except Exception as exc:
                print(f"\n  [!] Stacking failed for {target_name}/{matrix_name}: {exc}")
                results[target_name]['Stacking Ensemble'][matrix_name] = None

            gc.collect()
            print(f"  [*] {matrix_name} Matrix fully evaluated for {target_name}.    ")

        # ── Save progress after every target ─────────────────────────────
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        if best_preds is not None:
            print(f"  🏆 Plotting winner: {best_model_name} on {best_matrix_name}")
            plot_winning_roc(
                target_name, best_model_name, best_matrix_name,
                best_auc, best_y_test, best_preds, task_type, best_n_classes
            )

    # ── Print markdown tables ─────────────────────────────────────────────
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
                score = results.get(target_name, {}).get(model, {}).get(matrix_name, None)
                row += f"| {score:.4f} |" if score is not None else "| N/A |"
            print(row)


if __name__ == "__main__":
    run_full_tournament()