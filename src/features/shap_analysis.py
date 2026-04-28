"""
Phase 5: SHAP Explainability Analysis
======================================
Loads best_hyperparams.json, retrains each winning model on the full
training split, then runs the appropriate SHAP explainer.

Per target, outputs (to results/shap/):
  1. <target>_beeswarm.png   — beeswarm summary: feature × SHAP value, colored by feature value
  2. <target>_bar.png        — mean |SHAP| bar chart (top-20 features, global importance)
  3. <target>_waterfall.png  — single-patient waterfall (highest-risk patient in test set)

Explainer selection
-------------------
  CatBoost / XGBoost / Random Forest  → TreeExplainer  (fast, exact)
  Logistic Regression                 → LinearExplainer (fast)
  Custom MLP                          → DeepExplainer   (fast for keras)
  Stacking Ensemble                   → LinearExplainer on the meta-learner.
      Features are the 5 base-learner OOF predictions (binary) or
      5 × n_classes predictions (multiclass).
      This answers "which base learner does the meta-learner trust most?"

Run:
    pip install shap catboost xgboost scikit-learn tensorflow matplotlib
    python shap_analysis.py
"""

import json
import gc
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.features.leakage_rules import drop_organ_support_leaky_columns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import xgboost as xgb
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

from src.models.stacking_model import (           # ← NEW
    precompute_oof, train_meta_learner,
    get_test_meta_features, get_oof_feature_names
)


# ─────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────

PARAMS_PATH = Path('results/best_hyperparams.json')
DATA_DIR    = Path('data/processed/tournament')
OUT_DIR     = Path('results/shap')

MATRIX_FILES = {
    'IG':    'X_ig_union.csv',
    'ANOVA': 'X_anova_union.csv',
    'MI':    'X_mi_union.csv',
    'LASSO': 'X_lasso_union.csv',
}

ALL_TARGET_COLS = [
    'mortality', 'aki_onset', 'sepsis_onset', 'ards_onset', 'liver_injury_onset',
    'need_vent_any', 'need_vasopressor_any', 'need_rrt_any',
    'icu_readmit_48h', 'icu_readmit_7d', 'los_category', 'discharge_disposition',
    'los_days'
]
ID_COLS = ['subject_id', 'hadm_id', 'stay_id']

BACKGROUND_N     = 200
SHAP_EXPLAIN_N   = 2000
SHAP_EXPLAIN_N_RF = 600
TOP_N            = 20


# ─────────────────────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':        'sans-serif',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'figure.dpi':         150,
})

PALETTE = {
    'bar_fill':      '#534AB7',
    'bar_edge':      '#3C3489',
    'beeswarm_lo':   '#3B8BD4',
    'beeswarm_hi':   '#D85A30',
    'waterfall_pos': '#1D9E75',
    'waterfall_neg': '#E24B4A',
}


# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────

def load_data(target_name: str, matrix_name: str):
    path = DATA_DIR / MATRIX_FILES[matrix_name]
    df   = pd.read_csv(path)
    df.columns = [c.replace('[','').replace(']','').replace('<','lt').replace('>','gt')
                  for c in df.columns]
    df = df.dropna(subset=[target_name]).copy()

    le = LabelEncoder()
    y  = le.fit_transform(df[target_name])
    n_classes   = len(le.classes_)
    class_names = [str(c) for c in le.classes_]

    drop_cols = [c for c in ID_COLS + ALL_TARGET_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = drop_organ_support_leaky_columns(X, target_name)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    imputer     = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train_imp)
    X_test_sc   = scaler.transform(X_test_imp)

    return (pd.DataFrame(X_train_sc, columns=feature_names),
            pd.DataFrame(X_test_sc,  columns=feature_names),
            y_train, y_test, n_classes, class_names, feature_names)


# ─────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────

def build_catboost(params, task_type, n_classes):
    p = dict(params, random_seed=42, verbose=0)
    if task_type == 'binary':
        p.update({'loss_function': 'Logloss', 'auto_class_weights': 'Balanced'})
    else:
        p['loss_function'] = 'MultiClass'
    return CatBoostClassifier(**p)


def build_xgboost(params, task_type, n_classes, y_train):
    p = dict(params, random_state=42, n_jobs=-1, use_label_encoder=False)
    if task_type == 'binary':
        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)
        p.update({'eval_metric': 'logloss', 'scale_pos_weight': neg / pos if pos > 0 else 1.0})
    else:
        p.update({'eval_metric': 'mlogloss', 'objective': 'multi:softprob', 'num_class': n_classes})
    return xgb.XGBClassifier(**p)


def build_rf(params, task_type):
    return RandomForestClassifier(**params, class_weight='balanced',
                                  random_state=42, n_jobs=-1)


def build_lr(params, task_type):
    return LogisticRegression(**params, solver='saga', max_iter=2000,
                              class_weight='balanced', random_state=42)


def build_mlp(params, input_dim, task_type, n_classes):
    model = Sequential([
        Dense(params['units_1'], activation='relu', input_shape=(input_dim,)),
        Dropout(params['dropout_1']),
        Dense(params['units_2'], activation='relu'),
        Dropout(params['dropout_2']),
        Dense(params['units_3'], activation='relu'),
    ])
    if task_type == 'binary':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(params['lr']), loss='binary_crossentropy', metrics=['AUC'])
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=Adam(params['lr']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────

def train_model(model_name, params, task_type, n_classes, X_train, y_train):
    """
    Returns (fitted_model, is_keras_model).
    For 'Stacking Ensemble', returns (stacking_dict, False).
    stacking_dict = {'meta': ..., 'trained_bases': ..., ...}
    """
    if model_name == 'CatBoost':
        m = build_catboost(params, task_type, n_classes)
        m.fit(X_train.values, y_train)
        return m, False

    elif model_name == 'XGBoost':
        m = build_xgboost(params, task_type, n_classes, y_train)
        if task_type == 'multiclass':
            weights = compute_sample_weight('balanced', y_train)
            m.fit(X_train.values, y_train, sample_weight=weights)
        else:
            m.fit(X_train.values, y_train)
        return m, False

    elif model_name == 'Random Forest':
        m = build_rf(params, task_type)
        m.fit(X_train.values, y_train)
        return m, False

    elif model_name == 'Logistic Regression':
        m = build_lr(params, task_type)
        m.fit(X_train.values, y_train)
        return m, False

    elif model_name == 'Custom MLP':
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        cw  = dict(zip(classes, weights))
        val_n   = int(0.15 * len(X_train))
        X_tr2, X_val2 = X_train.values[:-val_n], X_train.values[-val_n:]
        y_tr2, y_val2 = y_train[:-val_n], y_train[-val_n:]
        m   = build_mlp(params, X_train.shape[1], task_type, n_classes)
        cb  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        m.fit(X_tr2, y_tr2, epochs=100, batch_size=params['batch_size'],
              validation_data=(X_val2, y_val2), callbacks=[cb],
              class_weight=cw, verbose=0)
        return m, True

    elif model_name == 'Stacking Ensemble':
        # ── NEW ──────────────────────────────────────────────────────────
        meta_C = params.get('meta_C', 1.0)
        oof_matrix, trained_bases, base_names = precompute_oof(
            X_train.values, y_train, task_type, n_classes
        )
        meta           = train_meta_learner(oof_matrix, y_train, meta_C)
        oof_feat_names = get_oof_feature_names(base_names, task_type, n_classes)
        stacking_dict  = {
            'meta':           meta,
            'trained_bases':  trained_bases,
            'base_names':     base_names,
            'oof_feat_names': oof_feat_names,
            'oof_matrix':     oof_matrix,   # kept for LinearExplainer background
            'task_type':      task_type,
            'n_classes':      n_classes,
        }
        return stacking_dict, False

    raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────
# SHAP EXPLAINERS
# ─────────────────────────────────────────────────────────────

def get_shap_values_stacking(stacking_dict, X_test):
    """
    Runs LinearExplainer on the meta-learner.

    The meta-learner's input space is the 5 base-learner predictions
    (or 5 × n_classes for multiclass).  SHAP values here answer:
    'which base learner's opinion shifted the meta-learner's output?'

    Returns (shap_vals, expected_val, X_test_meta_df, y_test_cap)
    — note: X_test_meta_df has base-learner names as column headers so
    downstream plot functions work without modification.
    """
    meta           = stacking_dict['meta']
    trained_bases  = stacking_dict['trained_bases']
    oof_feat_names = stacking_dict['oof_feat_names']
    oof_matrix     = stacking_dict['oof_matrix']    # used as background
    task_type      = stacking_dict['task_type']
    n_classes      = stacking_dict['n_classes']

    # Build test meta-features
    test_meta_np = get_test_meta_features(trained_bases, X_test.values, task_type, n_classes)
    test_meta_df = pd.DataFrame(test_meta_np, columns=oof_feat_names)

    # Background = OOF matrix (all training rows available, no sampling needed
    # because the meta-feature space is tiny: 5 or 5*C columns)
    bg_df = pd.DataFrame(oof_matrix, columns=oof_feat_names)

    explainer = shap.LinearExplainer(meta, bg_df, feature_perturbation='interventional')
    sv        = explainer.shap_values(test_meta_df)

    # Collapse multi-output to most informative class (mirrors existing logic)
    if isinstance(sv, list):
        mean_abs = [np.abs(s).mean() for s in sv]
        best_cls = int(np.argmax(mean_abs))
        shap_vals = sv[best_cls]
        expected  = (explainer.expected_value[best_cls]
                     if hasattr(explainer.expected_value, '__len__')
                     else float(explainer.expected_value))
    else:
        sv_arr = np.asarray(sv)
        if sv_arr.ndim == 3:
            mean_abs  = np.abs(sv_arr).mean(axis=(0, 1))
            best_cls  = int(np.argmax(mean_abs))
            shap_vals = sv_arr[:, :, best_cls]
            expected  = (explainer.expected_value[best_cls]
                         if hasattr(explainer.expected_value, '__len__')
                         else float(explainer.expected_value))
        else:
            shap_vals = sv_arr
            expected  = float(explainer.expected_value)

    return shap_vals, expected, test_meta_df


def get_shap_values(model_name, model, is_keras, X_train, X_test, y_test,
                    task_type, n_classes):
    """
    Dispatcher — returns (shap_values, expected_value, X_test_df, y_test_cap).
    shap_values is always 2D (n_samples × n_features).
    """
    explain_n = SHAP_EXPLAIN_N_RF if model_name == 'Random Forest' else SHAP_EXPLAIN_N

    if len(X_test) > explain_n:
        sample_idx = np.random.default_rng(42).choice(len(X_test), explain_n, replace=False)
        X_test = X_test.iloc[sample_idx].reset_index(drop=True)
        y_test = y_test[sample_idx]

    # ── Stacking Ensemble — special path ─────────────────────────────────
    if model_name == 'Stacking Ensemble':
        shap_vals, expected, X_test_meta_df = get_shap_values_stacking(model, X_test)
        return shap_vals, expected, X_test_meta_df, y_test

    bg = X_train.sample(n=min(BACKGROUND_N, len(X_train)), random_state=42)

    if model_name in ('CatBoost', 'XGBoost', 'Random Forest'):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)

        if isinstance(sv, list):
            mean_abs = [np.abs(s).mean() for s in sv]
            best_cls = int(np.argmax(mean_abs))
            shap_vals = sv[best_cls]
            expected  = (explainer.expected_value[best_cls]
                         if hasattr(explainer.expected_value, '__len__')
                         else explainer.expected_value)
        else:
            sv_arr = np.asarray(sv)
            if sv_arr.ndim == 3:
                mean_abs = np.abs(sv_arr).mean(axis=(0, 1))
                best_cls = int(np.argmax(mean_abs))
                shap_vals = sv_arr[:, :, best_cls]
                expected  = (explainer.expected_value[best_cls]
                             if hasattr(explainer.expected_value, '__len__')
                             else explainer.expected_value)
            else:
                shap_vals = sv_arr
                expected  = (explainer.expected_value[1]
                             if hasattr(explainer.expected_value, '__len__')
                             else explainer.expected_value)

    elif model_name == 'Logistic Regression':
        explainer = shap.LinearExplainer(model, bg, feature_perturbation='interventional')
        sv = explainer.shap_values(X_test)

        if isinstance(sv, list):
            mean_abs  = [np.abs(s).mean() for s in sv]
            best_cls  = int(np.argmax(mean_abs))
            shap_vals = sv[best_cls]
            expected  = explainer.expected_value[best_cls]
        else:
            sv_arr = np.asarray(sv)
            if sv_arr.ndim == 3:
                mean_abs  = np.abs(sv_arr).mean(axis=(0, 1))
                best_cls  = int(np.argmax(mean_abs))
                shap_vals = sv_arr[:, :, best_cls]
                expected  = explainer.expected_value[best_cls]
            else:
                shap_vals = sv_arr
                expected  = explainer.expected_value

    elif model_name == 'Custom MLP':
        bg_tensor = bg.values.astype(np.float32)
        explainer = shap.DeepExplainer(model, bg_tensor)
        sv = explainer.shap_values(X_test.values.astype(np.float32))

        def _extract_ev(ev, idx=None):
            val = ev[idx] if idx is not None else ev
            return float(np.squeeze(np.array(val)).ravel()[0])

        if isinstance(sv, list):
            sv = [np.squeeze(s) if s.ndim == 3 else s for s in sv]
            mean_abs  = [np.abs(s).mean() for s in sv]
            best_cls  = int(np.argmax(mean_abs))
            shap_vals = sv[best_cls]
            ev = explainer.expected_value
            expected  = _extract_ev(ev, best_cls) if hasattr(ev, '__len__') else _extract_ev(ev)
        else:
            sv_arr = np.asarray(sv)
            if sv_arr.ndim == 3 and sv_arr.shape[-1] > 1:
                mean_abs  = np.abs(sv_arr).mean(axis=(0, 1))
                best_cls  = int(np.argmax(mean_abs))
                shap_vals = sv_arr[:, :, best_cls]
                expected  = _extract_ev(explainer.expected_value, best_cls)
            else:
                shap_vals = np.squeeze(sv_arr) if sv_arr.ndim == 3 else sv_arr
                expected  = _extract_ev(explainer.expected_value)

    else:
        raise ValueError(f"No SHAP strategy for {model_name}")

    return shap_vals, expected, X_test, y_test


# ─────────────────────────────────────────────────────────────
# PLOTTERS  (unchanged — work on any (n_samples, n_features) SHAP matrix)
# ─────────────────────────────────────────────────────────────

def _title_str(target_name: str) -> str:
    return target_name.replace('_', ' ').title()


def plot_beeswarm(target_name, shap_vals, X_test, top_n=TOP_N):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    # For stacking, n_features may be <= 5, cap top_n
    top_n    = min(top_n, shap_vals.shape[1])
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    sv_top   = shap_vals[:, top_idx]
    X_top    = X_test.iloc[:, top_idx]
    feat_top = [X_test.columns[i] for i in top_idx]

    fig, ax  = plt.subplots(figsize=(10, 0.45 * top_n + 2))

    X_arr  = X_top.values.astype(float)
    X_norm = (X_arr - X_arr.min(axis=0)) / ((X_arr.max(axis=0) - X_arr.min(axis=0)) + 1e-9)
    cmap   = plt.get_cmap('RdBu_r')

    for row_i in range(top_n):
        feat_i  = top_n - 1 - row_i
        y_base  = row_i
        sv_col  = sv_top[:, feat_i]
        c_col   = X_norm[:, feat_i]
        rng     = np.random.default_rng(feat_i)
        jitter  = rng.uniform(-0.25, 0.25, size=len(sv_col))
        ax.scatter(sv_col, y_base + jitter, c=cmap(c_col), alpha=0.45, s=6, linewidths=0)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_top[top_n - 1 - i] for i in range(top_n)], fontsize=9)
    ax.axvline(0, color='#5F5E5A', lw=0.8, linestyle='--')
    ax.set_xlabel('SHAP value  (impact on model output)', fontsize=10)
    ax.set_title(f'{_title_str(target_name)} — SHAP beeswarm (top {top_n} features)',
                 fontsize=11, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.015, pad=0.02)
    cb.set_label('Feature value\n(low → high)', fontsize=8)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Low', 'High'], fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / f'{target_name}_beeswarm.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [saved] {path.name}")


def plot_bar(target_name, shap_vals, feature_names, top_n=TOP_N):
    top_n    = min(top_n, shap_vals.shape[1])
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    feats    = [feature_names[i] for i in top_idx]
    vals     = mean_abs[top_idx]

    feats, vals = feats[::-1], vals[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.40 * top_n + 2))
    bars = ax.barh(range(top_n), vals, color=PALETTE['bar_fill'],
                   edgecolor=PALETTE['bar_edge'], linewidth=0.4, height=0.65)

    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                f'{v:.4f}', va='center', ha='left', fontsize=8, color='#444441')

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats, fontsize=9)
    ax.set_xlabel('Mean |SHAP value|  (average impact on model output)', fontsize=10)
    ax.set_title(f'{_title_str(target_name)} — Global feature importance (top {top_n})',
                 fontsize=11, fontweight='bold')
    ax.set_xlim([0, max(vals) * 1.18])

    plt.tight_layout()
    path = OUT_DIR / f'{target_name}_bar.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [saved] {path.name}")


def plot_waterfall(target_name, shap_vals, expected_value, X_test, y_test, task_type):
    if task_type == 'binary':
        patient_idx = int(np.argmax(shap_vals.sum(axis=1)))
    else:
        patient_idx = int(np.argmax(np.abs(shap_vals).sum(axis=1)))

    sv_patient = shap_vals[patient_idx]
    feat_vals  = X_test.iloc[patient_idx].values
    feat_names = list(X_test.columns)
    true_label = y_test[patient_idx]

    display_n = min(15, len(feat_names))
    top_idx   = np.argsort(np.abs(sv_patient))[::-1][:display_n]

    sv_top = sv_patient[top_idx][::-1]
    fv_top = feat_vals[top_idx][::-1]
    fn_top = [feat_names[i] for i in top_idx][::-1]

    base    = float(expected_value)
    running = base
    lefts, widths, colors = [], [], []
    for v in sv_top:
        lefts.append(min(running, running + v))
        widths.append(abs(v))
        colors.append(PALETTE['waterfall_pos'] if v >= 0 else PALETTE['waterfall_neg'])
        running += v
    final_pred = running

    fig, ax = plt.subplots(figsize=(10, 0.5 * display_n + 2.5))
    ax.axvline(base, color='#888780', lw=1.2, linestyle='--', label=f'Base value = {base:.4f}')

    bars = ax.barh(range(display_n), widths, left=lefts, color=colors,
                   edgecolor='white', linewidth=0.4, height=0.6)

    for i, (b, v) in enumerate(zip(bars, sv_top)):
        sign  = '+' if v >= 0 else ''
        x_pos = lefts[i] + widths[i] + 0.002 if v >= 0 else lefts[i] - 0.002
        ha    = 'left' if v >= 0 else 'right'
        ax.text(x_pos, i, f'{sign}{v:.4f}', va='center', ha=ha, fontsize=8)

    labels_with_val = [f'{fn_top[i]}  = {fv_top[i]:.3g}' for i in range(display_n)]
    ax.set_yticks(range(display_n))
    ax.set_yticklabels(labels_with_val, fontsize=8.5)
    ax.axvline(final_pred, color='#3C3489', lw=1.5, linestyle='-',
               label=f'Prediction = {final_pred:.4f}')
    ax.set_xlabel('SHAP contribution  (cumulative from base value)', fontsize=10)
    ax.set_title(
        f'{_title_str(target_name)} — Highest-risk patient waterfall\n'
        f'True label: {true_label}  |  Prediction: {final_pred:.4f}  |  Base: {base:.4f}',
        fontsize=10, fontweight='bold'
    )

    from matplotlib.patches import Patch
    ax.legend(handles=[
        ax.axvline(base, color='#888780', lw=1.2, linestyle='--'),
        ax.axvline(final_pred, color='#3C3489', lw=1.5),
        Patch(facecolor=PALETTE['waterfall_pos'], label='Increases risk'),
        Patch(facecolor=PALETTE['waterfall_neg'], label='Decreases risk'),
    ], labels=[
        f'Base value = {base:.4f}',
        f'Prediction = {final_pred:.4f}',
        'Increases prediction',
        'Decreases prediction',
    ], fontsize=9, loc='lower right')

    plt.tight_layout()
    path = OUT_DIR / f'{target_name}_waterfall.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [saved] {path.name}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 5: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PARAMS_PATH) as f:
        all_params = json.load(f)

    task_type_map = {
        'mortality': 'binary', 'aki_onset': 'binary', 'sepsis_onset': 'binary',
        'ards_onset': 'binary', 'liver_injury_onset': 'binary',
        'need_vent_any': 'binary', 'need_vasopressor_any': 'binary',
        'need_rrt_any': 'binary', 'icu_readmit_48h': 'binary',
        'icu_readmit_7d': 'binary',
        'los_category': 'multiclass', 'discharge_disposition': 'multiclass',
    }

    for target_name, cfg in all_params.items():
        model_name  = cfg['model']
        matrix_name = cfg['matrix']
        params      = cfg['params']
        task_type   = task_type_map[target_name]

        print(f"\n{'─'*60}")
        print(f"TARGET : {target_name.upper()}")
        print(f"MODEL  : {model_name}  |  MATRIX : {matrix_name}")
        print(f"{'─'*60}")

        if (OUT_DIR / f'{target_name}_waterfall.png').exists():
            print(f'   [skip] already complete — delete PNG to re-run.')
            continue

        (X_train, X_test, y_train, y_test,
         n_classes, class_names, feat_names) = load_data(target_name, matrix_name)

        print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {len(feat_names)}")

        # Train (or re-assemble) the model
        print(f"   Retraining {model_name}...")
        if model_name == 'Stacking Ensemble':
            print(f"   [Stacking] Running OOF + meta-learner fit (this takes a few minutes)...")
        model, is_keras = train_model(model_name, params, task_type, n_classes, X_train, y_train)

        # Explainer selection info
        print(f"   Computing SHAP values ({model_name} → ", end='')
        if model_name in ('CatBoost', 'XGBoost', 'Random Forest'):
            print("TreeExplainer)...")
        elif model_name in ('Logistic Regression', 'Stacking Ensemble'):
            print("LinearExplainer)...")
        else:
            print(f"DeepExplainer, background n={BACKGROUND_N})...")

        shap_n = SHAP_EXPLAIN_N_RF if model_name == 'Random Forest' else SHAP_EXPLAIN_N
        print(f"   SHAP sampling cap: {shap_n} rows")
        t0 = time.perf_counter()
        shap_vals, expected_val, X_test_df, y_test_shap = get_shap_values(
            model_name, model, is_keras,
            X_train, X_test, y_test, task_type, n_classes
        )
        elapsed = time.perf_counter() - t0

        print(f"   SHAP matrix: {shap_vals.shape}  |  Expected value: {expected_val:.4f}")
        print(f"   SHAP compute time: {elapsed/60:.1f} min")

        if model_name == 'Stacking Ensemble':
            feat_names_shap = list(X_test_df.columns)   # base-learner names
            print(f"   [Stacking] Meta-learner features: {feat_names_shap}")
        else:
            feat_names_shap = feat_names

        print("   Generating plots...")
        plot_beeswarm(target_name, shap_vals, X_test_df)
        plot_bar(target_name, shap_vals, feat_names_shap)
        plot_waterfall(target_name, shap_vals, expected_val, X_test_df, y_test_shap, task_type)

        if is_keras:
            K.clear_session()
        del model, shap_vals
        gc.collect()

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print(f"All plots saved to: {OUT_DIR.absolute()}")
    print("\nFile naming convention:")
    print("  <target>_beeswarm.png  — per-patient SHAP distribution (top 20 features)")
    print("  <target>_bar.png       — global importance (mean |SHAP|)")
    print("  <target>_waterfall.png — highest-risk patient breakdown")
    print("\nNote for Stacking Ensemble targets:")
    print("  Feature names in plots = base learner short codes (CB, XGB, LGB, RF, LR).")
    print("  SHAP values show which base learner the meta-learner relies on most.")
    print("\n[NEXT] Review beeswarm plots for clinical sanity-checking.")


if __name__ == '__main__':
    main()