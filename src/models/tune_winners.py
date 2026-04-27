"""
Phase 4: Hyperparameter Tuning of Winning Configurations (With Smart Skip)
=========================================================
Uses Optuna (Bayesian TPE) to tune each target's winning model+matrix pair.

Outputs:
  - results/best_hyperparams.json        — best params for every target
  - results/tuning_summary.json          — memory checkpoint for the markdown table
  - results/roc_curves_tuned/            — new ROC PNG per target
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os, gc, json, warnings

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.features.leakage_rules import drop_organ_support_leaky_columns
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
import xgboost as xgb
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

WINNING_CONFIGS = {
    'mortality':             {'model': 'CatBoost',            'matrix': 'LASSO', 'type': 'binary',     'baseline_auc': 0.8898},
    'aki_onset':             {'model': 'Custom MLP',          'matrix': 'LASSO', 'type': 'binary',     'baseline_auc': 0.8187},
    'sepsis_onset':          {'model': 'Random Forest',       'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.7805},
    'ards_onset':            {'model': 'CatBoost',            'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.9351},
    'liver_injury_onset':    {'model': 'CatBoost',            'matrix': 'IG',    'type': 'binary',     'baseline_auc': 0.9232},
    'need_vent_any':         {'model': 'CatBoost',            'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.8909}, 
    'need_vasopressor_any':  {'model': 'CatBoost',            'matrix': 'MI',    'type': 'binary',     'baseline_auc': 0.8816}, 
    'need_rrt_any':          {'model': 'Custom MLP',          'matrix': 'IG',    'type': 'binary',     'baseline_auc': 0.9417}, 
    
    'icu_readmit_48h':       {'model': 'Logistic Regression', 'matrix': 'ANOVA', 'type': 'binary',     'baseline_auc': 0.5948},
    'icu_readmit_7d':        {'model': 'Logistic Regression', 'matrix': 'ANOVA', 'type': 'binary',     'baseline_auc': 0.6015},
    'los_category':          {'model': 'Custom MLP',          'matrix': 'LASSO', 'type': 'multiclass', 'baseline_auc': 0.7666},
    'discharge_disposition': {'model': 'XGBoost',             'matrix': 'LASSO', 'type': 'multiclass', 'baseline_auc': 0.8135},
}

MATRIX_FILES = {
    'IG':    'X_ig_union.csv',
    'ANOVA': 'X_anova_union.csv',
    'MI':    'X_mi_union.csv',
    'LASSO': 'X_lasso_union.csv',
}

ALL_TARGET_COLS = list(WINNING_CONFIGS.keys()) + ['los_days']
ID_COLS = ['subject_id', 'hadm_id', 'stay_id']

DATA_DIR    = Path('data/processed/tournament')
RESULTS_DIR = Path('results')
ROC_DIR     = RESULTS_DIR / 'roc_curves_tuned'

N_TRIALS  = 50   # Optuna trials per target
CV_FOLDS  = 3    # Stratified k-folds used inside sklearn objectives


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
    X = drop_organ_support_leaky_columns(X, target_name) # Prevents Leakage!

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc  = scaler.transform(X_test_imp)

    return X_train_sc, X_test_sc, y_train, y_test, n_classes


# ─────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _cv_score(model, X, y, task_type, folds=CV_FOLDS):
    scoring = 'roc_auc' if task_type == 'binary' else 'roc_auc_ovr_weighted'
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()

def objective_catboost(trial, X_train, y_train, task_type, n_classes):
    params = {
        'iterations':         trial.suggest_int('iterations', 200, 1000),
        'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':              trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature':trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength':    trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'border_count':       trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': 0,
    }
    if task_type == 'binary':
        params.update({'loss_function': 'Logloss', 'auto_class_weights': 'Balanced'})
    else:
        params['loss_function'] = 'MultiClass'
    return _cv_score(CatBoostClassifier(**params), X_train, y_train, task_type)

def objective_xgboost(trial, X_train, y_train, task_type, n_classes):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1,
    }
    if task_type == 'binary':
        params['eval_metric'] = 'logloss'
    else:
        params.update({'eval_metric': 'mlogloss', 'objective': 'multi:softprob', 'num_class': n_classes})
    return _cv_score(xgb.XGBClassifier(**params), X_train, y_train, task_type)

def objective_rf(trial, X_train, y_train, task_type, n_classes):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
        'max_depth':        trial.suggest_int('max_depth', 5, 30),
        'min_samples_split':trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
    }
    return _cv_score(RandomForestClassifier(**params), X_train, y_train, task_type)

def objective_lr(trial, X_train, y_train, task_type, n_classes):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    params = {
        'C':           trial.suggest_float('C', 1e-4, 100.0, log=True),
        'penalty':     penalty,
        'solver':      'saga',
        'max_iter':    2000,
        'class_weight':'balanced',
        'random_state': 42,
    }
    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
    return _cv_score(LogisticRegression(**params), X_train, y_train, task_type)

def objective_mlp(trial, X_tr, y_tr, X_val, y_val, task_type, n_classes):
    units_1   = trial.suggest_categorical('units_1',   [64, 128, 256])
    units_2   = trial.suggest_categorical('units_2',   [32, 64, 128])
    units_3   = trial.suggest_categorical('units_3',   [16, 32, 64])
    dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.5)
    dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.4)
    lr        = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_sz  = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = _build_mlp(units_1, units_2, units_3, dropout_1, dropout_2, lr, X_tr.shape[1], task_type, n_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=60, batch_size=batch_sz, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
    
    preds = model.predict(X_val, verbose=0)
    K.clear_session()
    
    if task_type == 'binary':
        return roc_auc_score(y_val, preds.flatten())
    return roc_auc_score(y_val, preds, multi_class='ovr', average='macro')


# ─────────────────────────────────────────────────────────────
# MODEL BUILDER HELPERS
# ─────────────────────────────────────────────────────────────

def _build_mlp(u1, u2, u3, d1, d2, lr, input_dim, task_type, n_classes):
    model = Sequential([
        Dense(u1, activation='relu', input_shape=(input_dim,)),
        Dropout(d1),
        Dense(u2, activation='relu'),
        Dropout(d2),
        Dense(u3, activation='relu'),
    ])
    if task_type == 'binary':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['AUC'])
    else:
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ─────────────────────────────────────────────────────────────
# FINAL EVALUATION (train on full train set, evaluate on test)
# ─────────────────────────────────────────────────────────────

def final_evaluate(target_name, config, best_params, X_train, X_test, y_train, y_test, n_classes):
    model_name = config['model']
    task_type  = config['type']

    if model_name == 'CatBoost':
        model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
        model.fit(X_train, y_train)

    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1, use_label_encoder=False)
        model.fit(X_train, y_train)

    elif model_name == 'Random Forest':
        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    elif model_name == 'Logistic Regression':
        model = LogisticRegression(**best_params, solver='saga', max_iter=2000, random_state=42)
        model.fit(X_train, y_train)

    elif model_name == 'Custom MLP':
        u1 = best_params['units_1'];  u2 = best_params['units_2'];  u3 = best_params['units_3']
        d1 = best_params['dropout_1']; d2 = best_params['dropout_2']
        lr = best_params['lr'];        bs = best_params['batch_size']

        val_n = int(0.15 * len(X_train))
        X_tr2, X_val2 = X_train[:-val_n], X_train[-val_n:]
        y_tr2, y_val2 = y_train[:-val_n], y_train[-val_n:]

        model = _build_mlp(u1, u2, u3, d1, d2, lr, X_train.shape[1], task_type, n_classes)
        cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_tr2, y_tr2, epochs=100, batch_size=bs, validation_data=(X_val2, y_val2), callbacks=[cb], verbose=0)

    # Predict
    if model_name == 'Custom MLP':
        preds = model.predict(X_test, verbose=0)
        if task_type == 'binary':
            preds = preds.flatten()
        K.clear_session()
    else:
        if task_type == 'binary':
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict_proba(X_test)

    if task_type == 'binary':
        auc_val = roc_auc_score(y_test, preds)
    else:
        auc_val = roc_auc_score(y_test, preds, multi_class='ovr', average='macro')

    return auc_val, preds


# ─────────────────────────────────────────────────────────────
# ROC CURVE PLOTTER
# ─────────────────────────────────────────────────────────────

def plot_roc(target_name, model_name, matrix_name, tuned_auc, baseline_auc, y_test, preds, task_type, n_classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#FAFAFA')

    if task_type == 'binary':
        fpr, tpr, _ = roc_curve(y_test, preds)
        ax.plot(fpr, tpr, lw=2.5, color='#E8593C', label=f'Tuned AUC = {tuned_auc:.4f}')
        ax.fill_between(fpr, tpr, alpha=0.07, color='#E8593C')
    else:
        y_bin  = label_binarize(y_test, classes=range(n_classes))
        colors = ['#534AB7', '#1D9E75', '#D85A30', '#3B8BD4', '#639922', '#BA7517']
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], preds[:, i])
            c_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, alpha=0.85, color=colors[i % len(colors)], label=f'Class {i}  AUC = {c_auc:.4f}')

    ax.plot([0, 1], [0, 1], color='#888780', lw=1.2, linestyle='--', label='Random  AUC = 0.5000')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0]) # Square aesthetic
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate',  fontsize=11, fontweight='bold')

    delta     = tuned_auc - baseline_auc
    delta_str = f'+{delta:.4f}' if delta >= 0 else f'{delta:.4f}'

    ax.set_title(
        f'{target_name.upper().replace("_", " ")}\n'
        f'{model_name} | {matrix_name} matrix  |  '
        f'Baseline {baseline_auc:.4f} → Tuned {tuned_auc:.4f}  ({delta_str})',
        fontsize=10, fontweight='bold', pad=12
    )
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    ROC_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROC_DIR / f'roc_tuned_{target_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   [ROC saved] roc_tuned_{target_name}.png")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("PHASE 4: OPTUNA HYPERPARAMETER TUNING (WITH SMART SKIP)")
    print(f"  Strategy : Bayesian optimisation (TPE sampler)")
    print(f"  Trials   : {N_TRIALS} per target")
    print(f"  CV folds : {CV_FOLDS} (sklearn models) | val split (MLP)")
    print("=" * 75)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ROC_DIR.mkdir(parents=True, exist_ok=True)

    params_path = RESULTS_DIR / 'best_hyperparams.json'
    summary_path = RESULTS_DIR / 'tuning_summary.json'

    # Load State Persistence
    if params_path.exists():
        with open(params_path, 'r') as f:
            all_best_params = json.load(f)
        print("  [+] Loaded previous best hyperparameters from JSON.")
    else:
        all_best_params = {}

    if summary_path.exists():
        with open(summary_path, 'r') as f:
            comparison_rows_dict = json.load(f)
        print("  [+] Loaded previous markdown summary from JSON.")
    else:
        comparison_rows_dict = {}

    for target_name, config in WINNING_CONFIGS.items():
        model_name  = config['model']
        matrix_name = config['matrix']
        task_type   = config['type']
        baseline    = config['baseline_auc']

        print(f"\n{'─'*60}")
        print(f"TARGET : {target_name.upper()}")
        
        # SMART SKIP LOGIC
        expected_plot_path = ROC_DIR / f'roc_tuned_{target_name}.png'
        if expected_plot_path.exists():
            print(f"  [skip] already tuned — delete {expected_plot_path.name} to re-run.")
            continue

        print(f"MODEL  : {model_name}  |  MATRIX : {matrix_name}  |  BASELINE : {baseline:.4f}")
        print(f"{'─'*60}")

        X_train, X_test, y_train, y_test, n_classes = load_data(target_name, matrix_name)

        # ── BUILD OPTUNA STUDY ──────────────────────────────
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        if model_name == 'CatBoost':
            study.optimize(lambda t: objective_catboost(t, X_train, y_train, task_type, n_classes), n_trials=N_TRIALS, show_progress_bar=True)
        elif model_name == 'XGBoost':
            study.optimize(lambda t: objective_xgboost(t, X_train, y_train, task_type, n_classes), n_trials=N_TRIALS, show_progress_bar=True)
        elif model_name == 'Random Forest':
            study.optimize(lambda t: objective_rf(t, X_train, y_train, task_type, n_classes), n_trials=N_TRIALS, show_progress_bar=True)
        elif model_name == 'Logistic Regression':
            study.optimize(lambda t: objective_lr(t, X_train, y_train, task_type, n_classes), n_trials=N_TRIALS, show_progress_bar=True)
        elif model_name == 'Custom MLP':
            val_n = int(0.15 * len(X_train))
            X_tr_opt, X_val_opt = X_train[:-val_n], X_train[-val_n:]
            y_tr_opt, y_val_opt = y_train[:-val_n], y_train[-val_n:]
            study.optimize(lambda t: objective_mlp(t, X_tr_opt, y_tr_opt, X_val_opt, y_val_opt, task_type, n_classes), n_trials=N_TRIALS, show_progress_bar=True)

        best_params  = study.best_params
        best_cv_auc  = study.best_value
        print(f"\n   Best CV AUC  : {best_cv_auc:.4f}")
        print(f"   Best params  : {json.dumps(best_params, indent=6)}")

        # ── FINAL TEST-SET EVALUATION ────────────────────────
        tuned_auc, preds = final_evaluate(
            target_name, config, best_params,
            X_train, X_test, y_train, y_test, n_classes
        )

        delta     = tuned_auc - baseline
        delta_str = f'+{delta:.4f}' if delta >= 0 else f'{delta:.4f}'
        print(f"   Test AUC     : {tuned_auc:.4f}  ({delta_str} vs baseline)")

        # ── PLOT & SAVE MEMORY ───────────────────────────────
        plot_roc(target_name, model_name, matrix_name, tuned_auc, baseline, y_test, preds, task_type, n_classes)

        comparison_rows_dict[target_name] = {
            'Target':       target_name,
            'Model':        model_name,
            'Matrix':       matrix_name,
            'Baseline AUC': baseline,
            'Tuned AUC':    tuned_auc,
            'Delta':        delta,
        }

        all_best_params[target_name] = {
            'model': model_name,
            'matrix': matrix_name,
            'params': best_params
        }
        
        # Save checkpoints immediately after each successful target
        with open(params_path, 'w') as f:
            json.dump(all_best_params, f, indent=2)
        with open(summary_path, 'w') as f:
            json.dump(comparison_rows_dict, f, indent=2)
            
        gc.collect()

    # ── MARKDOWN TABLE ────────────────────────────────────────
    print("\n" + "=" * 75)
    print("TUNING RESULTS: BEFORE vs AFTER")
    print("=" * 75)
    print(f"\n| {'Target':<25} | {'Model':<20} | {'Mtx':<5} | {'Baseline':>8} | {'Tuned':>8} | {'Delta':>8} |")
    print(f"| {'-'*25} | {'-'*20} | {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8} |")
    
    # Iterate through ALL targets using the saved dictionary
    for t_name in list(WINNING_CONFIGS.keys()):
        if t_name in comparison_rows_dict:
            r = comparison_rows_dict[t_name]
            d_str = f"+{r['Delta']:.4f}" if r['Delta'] >= 0 else f"{r['Delta']:.4f}"
            print(f"| {r['Target']:<25} | {r['Model']:<20} | {r['Matrix']:<5} | "
                  f"{r['Baseline AUC']:>8.4f} | {r['Tuned AUC']:>8.4f} | {d_str:>8} |")
        else:
            print(f"| {t_name:<25} | {'PENDING':<20} | {'...':<5} | {'...':>8} | {'...':>8} | {'...':>8} |")

    print(f"\n[DONE] ROC curves → {ROC_DIR.absolute()}")
    print("[NEXT] Run SHAP analysis with: python shap_analysis.py")


if __name__ == '__main__':
    main()