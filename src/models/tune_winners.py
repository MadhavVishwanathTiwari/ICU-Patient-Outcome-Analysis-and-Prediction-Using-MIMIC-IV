"""
Phase 4: Hyperparameter Tuning & Streamlit Deployment Packager
Uses Optuna to tune each target's winning model+matrix pair.
Immediately saves the `.pkl` models and preprocessors for Streamlit deployment.

Stacking Ensemble tuning strategy
----------------------------------
Running 5-fold OOF over all 5 base learners inside every Optuna trial would
be prohibitively slow (50 trials × 25 base-learner fits = 1,250 fits per target).
Instead we use a two-phase approach:

  Phase A — Pre-compute OOF once with default base learners.
  Phase B — Optuna tunes ONLY meta_C (50 LR fits on a 5-column matrix).

This is fast, principled, and captures the most impactful hyperparameter.
The saved artefact contains both the fitted base learners and the tuned
meta-learner so the full stack is available at inference time.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os, gc, json, warnings, joblib

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.features.leakage_rules import drop_organ_support_leaky_columns
from src.models.custom_mlp import build_custom_mlp, compute_class_weights
from src.models.stacking_model import (           # ← NEW
    precompute_oof, train_meta_learner,
    get_test_meta_features, get_oof_feature_names
)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Update WINNING_CONFIGS here once the tournament finishes and you know
# whether Stacking Ensemble won any target.  Example entry:
#   'aki_onset': {'model': 'Stacking Ensemble', 'matrix': 'MI', 'type': 'binary', 'baseline_auc': 0.8251}
# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

WINNING_CONFIGS = {
    'mortality':             {'model': 'Stacking Ensemble', 'matrix': 'LASSO', 'type': 'binary',     'baseline_auc': 0.8987},
    'aki_onset':             {'model': 'Stacking Ensemble', 'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.8301},
    'sepsis_onset':          {'model': 'Stacking Ensemble', 'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.7903},
    'ards_onset':            {'model': 'Stacking Ensemble', 'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.9366},
    'liver_injury_onset':    {'model': 'Stacking Ensemble', 'matrix': 'LASSO', 'type': 'binary',     'baseline_auc': 0.9291},
    'icu_readmit_48h':       {'model': 'Stacking Ensemble', 'matrix': 'ANOVA', 'type': 'binary',     'baseline_auc': 0.6107},
    'icu_readmit_7d':        {'model': 'Stacking Ensemble', 'matrix': 'IG',    'type': 'binary',     'baseline_auc': 0.6266},
    'los_category':          {'model': 'CatBoost',          'matrix': 'LASSO', 'type': 'multiclass', 'baseline_auc': 0.7641},
    'discharge_disposition': {'model': 'CatBoost',          'matrix': 'LASSO', 'type': 'multiclass', 'baseline_auc': 0.8126},
    'need_vent_any':         {'model': 'Stacking Ensemble', 'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.8953},
    'need_vasopressor_any':  {'model': 'Stacking Ensemble', 'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.8867},
    'need_rrt_any':          {'model': 'Stacking Ensemble', 'matrix': 'IG',    'type': 'binary',     'baseline_auc': 0.9488},
}

MATRIX_FILES = {
    'IG': 'X_ig_union.csv', 'ANOVA': 'X_anova_union.csv',
    'MI': 'X_mi_union.csv', 'LASSO': 'X_lasso_union.csv',
}

ALL_TARGET_COLS = list(WINNING_CONFIGS.keys()) + ['los_days']
ID_COLS = ['subject_id', 'hadm_id', 'stay_id']

DATA_DIR    = Path('data/processed/tournament')
RESULTS_DIR = Path('results')
ROC_DIR     = RESULTS_DIR / 'roc_curves_tuned'
DEPLOY_DIR  = Path('models/tuned')

N_TRIALS  = 50
CV_FOLDS  = 3


# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────

def load_data(target_name: str, matrix_name: str):
    path = DATA_DIR / MATRIX_FILES[matrix_name]
    df = pd.read_csv(path)
    df.columns = [c.replace('[','').replace(']','').replace('<','lt').replace('>','gt') for c in df.columns]
    df = df.dropna(subset=[target_name]).copy()

    le = LabelEncoder()
    y  = le.fit_transform(df[target_name])
    n_classes = len(le.classes_)

    drop_cols = [c for c in ID_COLS + ALL_TARGET_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = drop_organ_support_leaky_columns(X, target_name)
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_sc  = scaler.transform(imputer.transform(X_test))

    return X_train_sc, X_test_sc, y_train, y_test, n_classes, imputer, scaler, feature_cols, le


# ─────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVES (unchanged for non-stacking models)
# ─────────────────────────────────────────────────────────────

def _cv_score(model, X, y, task_type, folds=CV_FOLDS):
    scoring = 'roc_auc' if task_type == 'binary' else 'roc_auc_ovr'
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1).mean()


def objective_catboost(trial, X_train, y_train, task_type, n_classes):
    params = {
        'iterations':    trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':         trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_seed': 42, 'verbose': 0,
    }
    if task_type == 'binary':
        params.update({'loss_function': 'Logloss', 'auto_class_weights': 'Balanced'})
    else:
        params['loss_function'] = 'MultiClass'
    return _cv_score(CatBoostClassifier(**params), X_train, y_train, task_type)


def objective_xgboost(trial, X_train, y_train, task_type, n_classes):
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    spw = neg / pos if pos > 0 else 1.0
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':     trial.suggest_int('max_depth', 3, 10),
        'use_label_encoder': False, 'random_state': 42, 'n_jobs': 1,
    }
    if task_type == 'binary':
        params.update({'eval_metric': 'logloss', 'scale_pos_weight': spw})
    else:
        params.update({'eval_metric': 'mlogloss', 'objective': 'multi:softprob', 'num_class': n_classes})
    return _cv_score(xgb.XGBClassifier(**params), X_train, y_train, task_type)


def objective_lgbm(trial, X_train, y_train, task_type, n_classes):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves':    trial.suggest_int('num_leaves', 20, 150),
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': 1, 'verbose': -1
    }
    return _cv_score(lgb.LGBMClassifier(**params), X_train, y_train, task_type)


def objective_rf(trial, X_train, y_train, task_type, n_classes):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth':    trial.suggest_int('max_depth', 5, 30),
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': 1,
    }
    return _cv_score(RandomForestClassifier(**params), X_train, y_train, task_type)


def objective_lr(trial, X_train, y_train, task_type, n_classes):
    params = {
        'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
        'solver': 'saga', 'max_iter': 2000,
        'class_weight': 'balanced', 'random_state': 42
    }
    return _cv_score(LogisticRegression(**params), X_train, y_train, task_type)


def objective_mlp(trial, X_train_full, y_train_full, task_type, n_classes):
    u1 = trial.suggest_categorical('units_1', [64, 128, 256])
    u2 = trial.suggest_categorical('units_2', [32, 64, 128])
    u3 = trial.suggest_categorical('units_3', [16, 32, 64])
    d1 = trial.suggest_float('dropout_1', 0.1, 0.5)
    d2 = trial.suggest_float('dropout_2', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    bs = trial.suggest_categorical('batch_size', [32, 64, 128])

    rng_seed = 42 + trial.number
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=rng_seed, stratify=y_train_full
    )
    model = build_custom_mlp(X_tr.shape[1], task_type, n_classes, u1, u2, u3, d1, d2, lr)
    cw    = compute_class_weights(y_tr)
    cb    = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=60, batch_size=bs,
              validation_data=(X_val, y_val), callbacks=[cb], class_weight=cw, verbose=0)
    preds = model.predict(X_val, verbose=0).flatten() if task_type == 'binary' else model.predict(X_val, verbose=0)
    K.clear_session()
    return (roc_auc_score(y_val, preds)
            if task_type == 'binary'
            else roc_auc_score(y_val, preds, multi_class='ovr', average='macro'))


# ─────────────────────────────────────────────────────────────
# STACKING TUNING  (NEW)
# ─────────────────────────────────────────────────────────────

def objective_stacking(trial, oof_matrix, y_train, task_type):
    """
    Tunes only meta_C.  OOF matrix is pre-computed and passed in as a fixed
    argument so each trial is just one LogisticRegression fit on 5 features —
    this is essentially instant (< 0.1 s per trial).
    """
    meta_C = trial.suggest_float('meta_C', 1e-4, 100.0, log=True)
    meta   = LogisticRegression(
        C=meta_C, max_iter=2000, random_state=42,
        class_weight='balanced', solver='saga'
    )
    scoring = 'roc_auc' if task_type == 'binary' else 'roc_auc_ovr'
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    return cross_val_score(meta, oof_matrix, y_train, cv=cv, scoring=scoring, n_jobs=1).mean()


def final_evaluate_stacking(target_name, config, best_params,
                             X_train, X_test, y_train, y_test, n_classes,
                             precomputed_oof=None, pretrained_bases=None):
    """
    Builds and evaluates the full stacking model using the tuned meta_C.

    If precomputed_oof and pretrained_bases are provided (from the tuning
    phase), they are reused to avoid redundant computation.

    Returns (auc_val, final_preds, stacking_artifact_dict)
    """
    task_type = config['type']

    if precomputed_oof is None or pretrained_bases is None:
        from src.models.stacking_model import precompute_oof
        precomputed_oof, pretrained_bases, _ = precompute_oof(
            X_train, y_train, task_type, n_classes,
            input_dim=X_train.shape[1]   # FIX: required to build MLP + FTT graphs
        )

    meta_C = best_params.get('meta_C', 1.0)
    meta   = train_meta_learner(precomputed_oof, y_train, meta_C)

    test_meta   = get_test_meta_features(pretrained_bases, X_test, task_type, n_classes)
    base_names  = list(pretrained_bases.keys())
    oof_feat_names = get_oof_feature_names(base_names, task_type, n_classes)

    if task_type == 'binary':
        final_preds = meta.predict_proba(test_meta)[:, 1]
        auc_val     = roc_auc_score(y_test, final_preds)
    else:
        final_preds = meta.predict_proba(test_meta)
        auc_val     = roc_auc_score(y_test, final_preds, multi_class='ovr', average='macro')

    stacking_artifact = {
        'meta':            meta,
        'trained_bases':   pretrained_bases,
        'base_names':      base_names,
        'oof_feat_names':  oof_feat_names,
        'task_type':       task_type,
        'n_classes':       n_classes,
        'meta_C':          meta_C,
    }
    return auc_val, final_preds, stacking_artifact


# ─────────────────────────────────────────────────────────────
# NON-STACKING FINAL EVALUATE (unchanged logic)
# ─────────────────────────────────────────────────────────────

def final_evaluate(target_name, config, best_params,
                   X_train, X_test, y_train, y_test, n_classes):
    model_name = config['model']
    task_type  = config['type']

    spw      = 1.0
    sample_w = None
    if task_type == 'binary':
        neg  = int(np.sum(y_train == 0))
        pos  = int(np.sum(y_train == 1))
        spw  = neg / pos if pos > 0 else 1.0
    elif model_name == 'XGBoost':
        sample_w = compute_sample_weight('balanced', y_train)

    if model_name == 'CatBoost':
        p = dict(best_params, random_seed=42, verbose=0)
        if task_type == 'binary':
            p.update({'loss_function': 'Logloss', 'auto_class_weights': 'Balanced'})
        else:
            p['loss_function'] = 'MultiClass'
        model = CatBoostClassifier(**p)
        model.fit(X_train, y_train)

    elif model_name == 'XGBoost':
        p = dict(best_params, random_state=42, n_jobs=-1, use_label_encoder=False)
        if task_type == 'binary':
            p.update({'eval_metric': 'logloss', 'scale_pos_weight': spw})
            model = xgb.XGBClassifier(**p)
            model.fit(X_train, y_train)
        else:
            p.update({'eval_metric': 'mlogloss', 'objective': 'multi:softprob', 'num_class': n_classes})
            model = xgb.XGBClassifier(**p)
            model.fit(X_train, y_train, sample_weight=sample_w)

    elif model_name == 'LightGBM':
        model = lgb.LGBMClassifier(**best_params, class_weight='balanced',
                                    random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train)

    elif model_name == 'Random Forest':
        model = RandomForestClassifier(**best_params, class_weight='balanced',
                                        random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    elif model_name == 'Logistic Regression':
        model = LogisticRegression(**best_params, solver='saga', max_iter=2000,
                                    class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

    elif model_name == 'Custom MLP':
        X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        model = build_custom_mlp(
            X_train.shape[1], task_type, n_classes,
            best_params['units_1'], best_params['units_2'], best_params['units_3'],
            best_params['dropout_1'], best_params['dropout_2'], best_params['lr']
        )
        cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_tr2, y_tr2, epochs=100, batch_size=best_params['batch_size'],
            validation_data=(X_val2, y_val2), callbacks=[cb],
            class_weight=compute_class_weights(y_tr2), verbose=0
        )

    else:
        raise ValueError(f"Unknown model in final_evaluate: {model_name}")

    if model_name == 'Custom MLP':
        preds = model.predict(X_test, verbose=0)
        preds = preds.flatten() if task_type == 'binary' else preds
        K.clear_session()
    else:
        preds = (model.predict_proba(X_test)[:, 1]
                 if task_type == 'binary'
                 else model.predict_proba(X_test))

    auc_val = (roc_auc_score(y_test, preds)
               if task_type == 'binary'
               else roc_auc_score(y_test, preds, multi_class='ovr', average='macro'))

    return auc_val, preds, model


# ─────────────────────────────────────────────────────────────
# SHARED PLOTTERS
# ─────────────────────────────────────────────────────────────

def plot_roc(target_name, model_name, matrix_name, tuned_auc, baseline_auc,
             y_test, preds, task_type, n_classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    if task_type == 'binary':
        fpr, tpr, _ = roc_curve(y_test, preds)
        ax.plot(fpr, tpr, lw=2.5, color='#E8593C', label=f'Tuned AUC = {tuned_auc:.4f}')
    else:
        y_bin = label_binarize(y_test, classes=range(n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], preds[:, i])
            ax.plot(fpr, tpr, lw=1.5, alpha=0.85, label=f'Class {i} AUC = {auc(fpr, tpr):.4f}')

    ax.plot([0, 1], [0, 1], color='#888780', lw=1.2, linestyle='--')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(
        f'{target_name.upper()}\n'
        f'{model_name} | {matrix_name} matrix | '
        f'Baseline {baseline_auc:.4f} → Tuned {tuned_auc:.4f}',
        fontweight='bold', pad=12
    )
    ax.legend(loc='lower right')
    fig.savefig(ROC_DIR / f'roc_tuned_{target_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("PHASE 4: OPTUNA TUNING & DEPLOYMENT EXPORT")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ROC_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    params_path   = RESULTS_DIR / 'best_hyperparams.json'
    summary_path  = RESULTS_DIR / 'tuning_summary.json'
    all_best_params      = json.load(open(params_path))      if params_path.exists()  else {}
    comparison_rows_dict = json.load(open(summary_path))     if summary_path.exists() else {}

    for target_name, config in WINNING_CONFIGS.items():
        already_done = (
            (ROC_DIR   / f'roc_tuned_{target_name}.png').exists() and
            (DEPLOY_DIR / f'{target_name}_prep.pkl').exists()
        )
        if already_done:
            print(f"  [skip] {target_name} already tuned and exported.")
            continue

        print(f"\n{'─'*60}\nTARGET : {target_name.upper()} | {config['model']} | {config['matrix']}\n{'─'*60}")

        (X_train, X_test, y_train, y_test,
         n_classes, imputer, scaler, feature_cols, le) = load_data(target_name, config['matrix'])

        # ── Route to the correct tuning path ─────────────────────────────
        is_stacking = config['model'] == 'Stacking Ensemble'

        if is_stacking:
            # Phase A: pre-compute OOF once (expensive, done only once per target)
            print("   [Stacking] Phase A — pre-computing OOF predictions...")
            oof_matrix, pretrained_bases, base_names = precompute_oof(
                X_train, y_train, config['type'], n_classes,
                input_dim=X_train.shape[1]   # FIX: required to build MLP + FTT graphs
            )
            print(f"   [Stacking] OOF matrix shape: {oof_matrix.shape}  "
                  f"(base learners: {base_names})")

            # Phase B: Optuna tunes only meta_C on the fixed OOF matrix
            print(f"   [Stacking] Phase B — tuning meta_C ({N_TRIALS} trials)...")
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(
                lambda t: objective_stacking(t, oof_matrix, y_train, config['type']),
                n_trials=N_TRIALS
            )
            best_params = study.best_params
            print(f"   [Stacking] Best meta_C = {best_params['meta_C']:.5f} "
                  f"(CV AUC = {study.best_value:.4f})")

            tuned_auc, preds, stack_artifact = final_evaluate_stacking(
                target_name, config, best_params,
                X_train, X_test, y_train, y_test, n_classes,
                precomputed_oof=oof_matrix,
                pretrained_bases=pretrained_bases
            )

        else:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            if config['model'] == 'CatBoost':
                study.optimize(lambda t: objective_catboost(t, X_train, y_train, config['type'], n_classes), n_trials=N_TRIALS)
            elif config['model'] == 'XGBoost':
                study.optimize(lambda t: objective_xgboost(t, X_train, y_train, config['type'], n_classes), n_trials=N_TRIALS)
            elif config['model'] == 'LightGBM':
                study.optimize(lambda t: objective_lgbm(t, X_train, y_train, config['type'], n_classes), n_trials=N_TRIALS)
            elif config['model'] == 'Random Forest':
                study.optimize(lambda t: objective_rf(t, X_train, y_train, config['type'], n_classes), n_trials=N_TRIALS)
            elif config['model'] == 'Logistic Regression':
                study.optimize(lambda t: objective_lr(t, X_train, y_train, config['type'], n_classes), n_trials=N_TRIALS)
            elif config['model'] == 'Custom MLP':
                study.optimize(lambda t: objective_mlp(t, X_train, y_train, config['type'], n_classes), n_trials=N_TRIALS)

            best_params = study.best_params
            tuned_auc, preds, trained_model = final_evaluate(
                target_name, config, best_params,
                X_train, X_test, y_train, y_test, n_classes
            )

        print(f"   Test AUC: {tuned_auc:.4f} (Baseline: {config['baseline_auc']:.4f})")
        plot_roc(target_name, config['model'], config['matrix'],
                 tuned_auc, config['baseline_auc'],
                 y_test, preds, config['type'], n_classes)

        # ── Export to Streamlit deploy folder ────────────────────────────
        joblib.dump(
            {'imputer': imputer, 'scaler': scaler,
             'features': feature_cols, 'label_encoder': le},
            DEPLOY_DIR / f'{target_name}_prep.pkl'
        )

        if is_stacking:
            # Save the full stacking artifact (meta + all base learners)
            joblib.dump(stack_artifact, DEPLOY_DIR / f'{target_name}_stacking.pkl')
            print(f"   [+] Stacking artifact saved → {DEPLOY_DIR.name}/{target_name}_stacking.pkl")
        elif config['model'] == 'Custom MLP':
            trained_model.save(DEPLOY_DIR / f'{target_name}_mlp.keras')
        else:
            safe_name = config['model'].lower().replace(' ', '_')
            joblib.dump(trained_model, DEPLOY_DIR / f'{target_name}_{safe_name}.pkl')

        print(f"   [+] Preprocessors packaged → {DEPLOY_DIR.name}/")

        comparison_rows_dict[target_name] = {
            'Target': target_name, 'Model': config['model'],
            'Matrix': config['matrix'],
            'Baseline AUC': config['baseline_auc'],
            'Tuned AUC': tuned_auc,
            'Delta': tuned_auc - config['baseline_auc']
        }
        all_best_params[target_name] = {
            'model': config['model'], 'matrix': config['matrix'],
            'params': best_params
        }

        with open(params_path,  'w') as f: json.dump(all_best_params, f, indent=2)
        with open(summary_path, 'w') as f: json.dump(comparison_rows_dict, f, indent=2)
        gc.collect()

    # ── Markdown summary ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("TUNING RESULTS: BEFORE vs AFTER")
    print("=" * 75)
    print(f"\n| {'Target':<25} | {'Model':<20} | {'Mtx':<5} | {'Baseline':>8} | {'Tuned':>8} | {'Delta':>8} |")
    print(f"| {'-'*25} | {'-'*20} | {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8} |")
    for t_name in list(WINNING_CONFIGS.keys()):
        if t_name in comparison_rows_dict:
            r     = comparison_rows_dict[t_name]
            d_str = f"+{r['Delta']:.4f}" if r['Delta'] >= 0 else f"{r['Delta']:.4f}"
            print(f"| {r['Target']:<25} | {r['Model']:<20} | {r['Matrix']:<5} | "
                  f"{r['Baseline AUC']:>8.4f} | {r['Tuned AUC']:>8.4f} | {d_str:>8} |")
        else:
            print(f"| {t_name:<25} | {'PENDING':<20} | {'...':<5} | {'...':>8} | {'...':>8} | {'...':>8} |")

    print(f"\n[DONE] Models & Preprocessors saved to {DEPLOY_DIR.absolute()}")


if __name__ == '__main__':
    main()