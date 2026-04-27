"""
Phase 5: SHAP Explainability Analysis
======================================
Loads best_hyperparams.json, retrains each winning model on the full
training split, then runs the appropriate SHAP explainer.

Per target, outputs (to results/shap/):
  1. <target>_beeswarm.png   — beeswarm summary: feature × SHAP value, colored by feature value
  2. <target>_bar.png        — mean |SHAP| bar chart (top-20 features, global importance)
  3. <target>_waterfall.png  — single-patient waterfall (highest-risk patient in test set)

Explainer selection:
  CatBoost / XGBoost / Random Forest → TreeExplainer  (fast, exact)
  Logistic Regression                → LinearExplainer (fast)
  Custom MLP                         → DeepExplainer   (fast for keras)

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
import xgboost as xgb
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K


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

# Background sample size for KernelExplainer / DeepExplainer
# Larger = more accurate but slower. 200 is a reasonable balance.
BACKGROUND_N = 200

# Max test-set rows passed to SHAP.
# Keep higher cap for most models, but use a smaller cap for Random Forest
# because TreeExplainer can become very slow on deep forests.
SHAP_EXPLAIN_N = 2000
SHAP_EXPLAIN_N_RF = 600

# How many top features to show in bar and beeswarm plots
TOP_N = 20

# ─────────────────────────────────────────────────────────────
# PLOT STYLE
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':   'sans-serif',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'figure.dpi':         150,
})

PALETTE = {
    'bar_fill':    '#534AB7',
    'bar_edge':    '#3C3489',
    'beeswarm_lo': '#3B8BD4',
    'beeswarm_hi': '#D85A30',
    'waterfall_pos': '#1D9E75',
    'waterfall_neg': '#E24B4A',
}


# ─────────────────────────────────────────────────────────────
# DATA LOADER  (mirrors tune_winners.py exactly)
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
# MODEL BUILDERS  (identical hyper-params to tune_winners.py)
# ─────────────────────────────────────────────────────────────

def build_catboost(params, task_type, n_classes):
    p = dict(params, random_seed=42, verbose=0)
    if task_type == 'binary':
        p.update({'loss_function': 'Logloss', 'auto_class_weights': 'Balanced'})
    else:
        p['loss_function'] = 'MultiClass'
    return CatBoostClassifier(**p)


def build_xgboost(params, task_type, n_classes):
    p = dict(params, random_state=42, n_jobs=-1, use_label_encoder=False)
    if task_type == 'binary':
        p['eval_metric'] = 'logloss'
    else:
        p.update({'eval_metric': 'mlogloss', 'objective': 'multi:softprob',
                  'num_class': n_classes})
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
        model.compile(optimizer=Adam(params['lr']),
                      loss='binary_crossentropy', metrics=['AUC'])
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=Adam(params['lr']),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model_name, params, task_type, n_classes,
                X_train, y_train):
    """
    Trains the final model on the full training split.
    Returns (fitted_model, is_keras_model).
    """
    if model_name == 'CatBoost':
        m = build_catboost(params, task_type, n_classes)
        m.fit(X_train.values, y_train)
        return m, False

    elif model_name == 'XGBoost':
        m = build_xgboost(params, task_type, n_classes)
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
        val_n = int(0.15 * len(X_train))
        X_tr2, X_val2 = X_train.values[:-val_n], X_train.values[-val_n:]
        y_tr2, y_val2 = y_train[:-val_n], y_train[-val_n:]
        m = build_mlp(params, X_train.shape[1], task_type, n_classes)
        cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        m.fit(X_tr2, y_tr2, epochs=100, batch_size=params['batch_size'],
              validation_data=(X_val2, y_val2), callbacks=[cb], verbose=0)
        return m, True

    raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────
# SHAP EXPLAINERS
# ─────────────────────────────────────────────────────────────

def get_shap_values(model_name, model, is_keras, X_train, X_test, y_test,
                    task_type, n_classes):
    """
    Returns (shap_values, expected_value, X_test_df).
    shap_values is a 2D numpy array (n_samples × n_features).
    For multiclass, we return the class with highest mean |SHAP| across classes.
    """
    # Model-specific cap to keep runtime practical.
    explain_n = SHAP_EXPLAIN_N_RF if model_name == 'Random Forest' else SHAP_EXPLAIN_N

    # Sample test set — SHAP plots remain representative with this cap.
    if len(X_test) > explain_n:
        sample_idx = np.random.default_rng(42).choice(len(X_test), explain_n, replace=False)
        X_test  = X_test.iloc[sample_idx].reset_index(drop=True)
        y_test  = y_test[sample_idx]
    bg = X_train.sample(n=min(BACKGROUND_N, len(X_train)), random_state=42)

    if model_name in ('CatBoost', 'XGBoost', 'Random Forest'):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)

        if isinstance(sv, list):
            # multiclass → list of (n × f) arrays, one per class
            # pick the class with highest mean |SHAP| — most informative
            mean_abs = [np.abs(s).mean() for s in sv]
            best_cls = int(np.argmax(mean_abs))
            shap_vals = sv[best_cls]
            expected  = (explainer.expected_value[best_cls]
                         if hasattr(explainer.expected_value, '__len__')
                         else explainer.expected_value)
        else:
            # Newer SHAP versions may return ndarray with class axis:
            #   binary    -> (n, f, 2)
            #   multiclass-> (n, f, c)
            # We collapse to one class so downstream plotting always gets (n, f).
            sv_arr = np.asarray(sv)
            if sv_arr.ndim == 3:
                mean_abs = np.abs(sv_arr).mean(axis=(0, 1))
                best_cls = int(np.argmax(mean_abs))
                shap_vals = sv_arr[:, :, best_cls]
                expected  = (explainer.expected_value[best_cls]
                             if hasattr(explainer.expected_value, '__len__')
                             else explainer.expected_value)
            else:
                # binary legacy path -> (n, f)
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
                mean_abs = np.abs(sv_arr).mean(axis=(0, 1))
                best_cls = int(np.argmax(mean_abs))
                shap_vals = sv_arr[:, :, best_cls]
                expected  = explainer.expected_value[best_cls]
            else:
                shap_vals = sv_arr
                expected  = explainer.expected_value

    elif model_name == 'Custom MLP':
        # DeepExplainer needs the raw Keras model
        bg_tensor = bg.values.astype(np.float32)
        explainer = shap.DeepExplainer(model, bg_tensor)
        sv = explainer.shap_values(X_test.values.astype(np.float32))

        def _extract_ev(ev, idx=None):
            """Safely pull a scalar from whatever DeepExplainer returns."""
            val = ev[idx] if idx is not None else ev
            return float(np.squeeze(np.array(val)).ravel()[0])

        # DeepExplainer may return (n, features, 1) for binary sigmoid — squeeze trailing dim
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
                mean_abs = np.abs(sv_arr).mean(axis=(0, 1))
                best_cls = int(np.argmax(mean_abs))
                shap_vals = sv_arr[:, :, best_cls]
                expected  = _extract_ev(explainer.expected_value, best_cls)
            else:
                shap_vals = np.squeeze(sv_arr) if sv_arr.ndim == 3 else sv_arr
                expected  = _extract_ev(explainer.expected_value)

    else:
        raise ValueError(f"No SHAP strategy for {model_name}")

    return shap_vals, expected, X_test, y_test


# ─────────────────────────────────────────────────────────────
# PLOTTERS
# ─────────────────────────────────────────────────────────────

def _title_str(target_name: str) -> str:
    return target_name.replace('_', ' ').title()


def plot_beeswarm(target_name, shap_vals, X_test, top_n=TOP_N):
    """
    Beeswarm plot: each point = one patient.
    X-axis = SHAP value (pushes prediction left/right).
    Color   = actual feature value (blue=low, red=high).
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    sv_top   = shap_vals[:, top_idx]
    X_top    = X_test.iloc[:, top_idx]
    feat_top = [X_test.columns[i] for i in top_idx]

    fig, ax  = plt.subplots(figsize=(10, 0.45 * top_n + 2))

    # Normalize feature values to [0,1] for coloring
    X_arr = X_top.values.astype(float)
    X_norm = (X_arr - X_arr.min(axis=0)) / ((X_arr.max(axis=0) - X_arr.min(axis=0)) + 1e-9)

    cmap = plt.get_cmap('RdBu_r')

    for row_i in range(top_n):
        feat_i  = top_n - 1 - row_i   # reverse so top feature is at top
        y_base  = row_i
        sv_col  = sv_top[:, feat_i]
        c_col   = X_norm[:, feat_i]

        # Jitter points vertically so they don't all overlap
        rng    = np.random.default_rng(feat_i)
        jitter = rng.uniform(-0.25, 0.25, size=len(sv_col))

        ax.scatter(sv_col, y_base + jitter,
                   c=cmap(c_col), alpha=0.45, s=6, linewidths=0)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_top[top_n - 1 - i] for i in range(top_n)], fontsize=9)
    ax.axvline(0, color='#5F5E5A', lw=0.8, linestyle='--')
    ax.set_xlabel('SHAP value  (impact on model output)', fontsize=10)
    ax.set_title(f'{_title_str(target_name)} — SHAP beeswarm (top {top_n} features)', fontsize=11, fontweight='bold')

    # Colorbar
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
    """
    Mean |SHAP| bar chart — global feature importance.
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    feats    = [feature_names[i] for i in top_idx]
    vals     = mean_abs[top_idx]

    # Reverse for horizontal bar (highest at top)
    feats, vals = feats[::-1], vals[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.40 * top_n + 2))
    bars = ax.barh(range(top_n), vals,
                   color=PALETTE['bar_fill'], edgecolor=PALETTE['bar_edge'],
                   linewidth=0.4, height=0.65)

    # Value labels on bars
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                f'{v:.4f}', va='center', ha='left', fontsize=8,
                color='var(--color-text-secondary)' if False else '#444441')

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats, fontsize=9)
    ax.set_xlabel('Mean |SHAP value|  (average impact on model output)', fontsize=10)
    ax.set_title(f'{_title_str(target_name)} — Global feature importance (top {top_n})', fontsize=11, fontweight='bold')
    ax.set_xlim([0, max(vals) * 1.18])

    plt.tight_layout()
    path = OUT_DIR / f'{target_name}_bar.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   [saved] {path.name}")


def plot_waterfall(target_name, shap_vals, expected_value, X_test, y_test, task_type):
    """
    Waterfall for the single highest-risk patient in the test set.
    Shows the top features that pushed the prediction up or down from the base value.
    """
    # Pick the patient with the highest total SHAP sum (most "at risk")
    if task_type == 'binary':
        patient_idx = int(np.argmax(shap_vals.sum(axis=1)))
    else:
        patient_idx = int(np.argmax(np.abs(shap_vals).sum(axis=1)))

    sv_patient   = shap_vals[patient_idx]
    feat_vals    = X_test.iloc[patient_idx].values
    feat_names   = list(X_test.columns)
    true_label   = y_test[patient_idx]

    # Sort by absolute SHAP and take top features
    display_n = min(15, len(feat_names))
    top_idx   = np.argsort(np.abs(sv_patient))[::-1][:display_n]

    sv_top    = sv_patient[top_idx][::-1]
    fv_top    = feat_vals[top_idx][::-1]
    fn_top    = [feat_names[i] for i in top_idx][::-1]

    # Cumulative sum for waterfall positioning
    base      = float(expected_value)
    running   = base
    lefts, widths, colors, midpoints = [], [], [], []
    for v in sv_top:
        lefts.append(min(running, running + v))
        widths.append(abs(v))
        colors.append(PALETTE['waterfall_pos'] if v >= 0 else PALETTE['waterfall_neg'])
        midpoints.append(running + v / 2)
        running += v
    final_pred = running

    fig, ax = plt.subplots(figsize=(10, 0.5 * display_n + 2.5))

    # Base value line
    ax.axvline(base, color='#888780', lw=1.2, linestyle='--', label=f'Base value = {base:.4f}')

    bars = ax.barh(range(display_n), widths, left=lefts, color=colors,
                   edgecolor='white', linewidth=0.4, height=0.6)

    # SHAP value labels
    for i, (b, v) in enumerate(zip(bars, sv_top)):
        sign  = '+' if v >= 0 else ''
        x_pos = lefts[i] + widths[i] + 0.002 if v >= 0 else lefts[i] - 0.002
        ha    = 'left' if v >= 0 else 'right'
        ax.text(x_pos, i, f'{sign}{v:.4f}', va='center', ha=ha, fontsize=8)

    # Feature labels with actual values
    labels_with_val = [f'{fn_top[i]}  = {fv_top[i]:.3g}' for i in range(display_n)]
    ax.set_yticks(range(display_n))
    ax.set_yticklabels(labels_with_val, fontsize=8.5)

    # Final prediction marker
    ax.axvline(final_pred, color='#3C3489', lw=1.5, linestyle='-',
               label=f'Prediction = {final_pred:.4f}')

    ax.set_xlabel('SHAP contribution  (cumulative from base value)', fontsize=10)
    ax.set_title(
        f'{_title_str(target_name)} — Highest-risk patient waterfall\n'
        f'True label: {true_label}  |  Prediction: {final_pred:.4f}  |  Base: {base:.4f}',
        fontsize=10, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='lower right')

    # Positive / negative legend patches
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

        # Skip targets already completed in a previous run
        if (OUT_DIR / f'{target_name}_waterfall.png').exists():
            print(f'   [skip] already complete — delete PNG to re-run.')
            continue

        # Load data
        (X_train, X_test, y_train, y_test,
         n_classes, class_names, feat_names) = load_data(target_name, matrix_name)

        print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {len(feat_names)}")

        # Train model
        print(f"   Retraining {model_name}...")
        model, is_keras = train_model(model_name, params, task_type, n_classes,
                                      X_train, y_train)

        # SHAP values
        print(f"   Computing SHAP values ({model_name} → ", end='')
        if model_name in ('CatBoost', 'XGBoost', 'Random Forest'):
            print("TreeExplainer)...")
        elif model_name == 'Logistic Regression':
            print("LinearExplainer)...")
        else:
            print(f"DeepExplainer, background n={BACKGROUND_N})...")

        shap_n = SHAP_EXPLAIN_N_RF if model_name == 'Random Forest' else SHAP_EXPLAIN_N
        print(f"   SHAP sampling cap: {shap_n} rows  |  Background: {min(BACKGROUND_N, len(X_train))}")
        t0 = time.perf_counter()
        shap_vals, expected_val, X_test_df, y_test_shap = get_shap_values(
            model_name, model, is_keras,
            X_train, X_test, y_test, task_type, n_classes
        )
        elapsed = time.perf_counter() - t0

        print(f"   SHAP matrix: {shap_vals.shape}  |  Expected value: {expected_val:.4f}")
        print(f"   SHAP compute time: {elapsed/60:.1f} min")

        # Plots
        print("   Generating plots...")
        plot_beeswarm(target_name, shap_vals, X_test_df)
        plot_bar(target_name, shap_vals, feat_names)
        plot_waterfall(target_name, shap_vals, expected_val,
                       X_test_df, y_test_shap, task_type)

        # Cleanup
        if is_keras:
            K.clear_session()
        del model, shap_vals
        gc.collect()

    # ── SUMMARY TABLE  ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print(f"All plots saved to: {OUT_DIR.absolute()}")
    print("\nFile naming convention:")
    print("  <target>_beeswarm.png  — per-patient SHAP distribution (top 20 features)")
    print("  <target>_bar.png       — global importance (mean |SHAP|)")
    print("  <target>_waterfall.png — highest-risk patient breakdown")
    print("\n[NEXT] Review beeswarm plots for clinical sanity-checking.")
    print("       Cross-reference top features against known clinical literature.")
    print("       Flag any features with unexpectedly high importance (data leakage check).")


if __name__ == '__main__':
    main()