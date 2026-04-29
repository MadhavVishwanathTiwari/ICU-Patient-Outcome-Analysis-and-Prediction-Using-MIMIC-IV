"""
Stacking Ensemble — MIMIC-IV Clinical Prediction Tournament
=============================================================
Implements 5-fold Out-of-Fold (OOF) stacking with 6 base learners
(CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression, Custom MLP)
and a Logistic Regression meta-learner.

Design rationale
----------------
• Base learners are 5 sklearn-compatible tree/linear models PLUS the Custom MLP.
  Including the MLP at Level 0 is deliberate: all five tree/linear models partition
  feature space with axis-aligned splits and tend to be wrong in similar regions.
  The MLP learns smooth, non-linear boundaries and will be confident in different
  places, giving the Level 1 meta-learner genuine diversity to exploit.

• MLP is handled separately from the sklearn models inside the OOF loop because:
    – Keras models cannot be reliably deepcopy'd (graph state issues).
    – The model must be rebuilt fresh for each fold to avoid weight leakage.
    – K.clear_session() is called after each fold to prevent GPU/memory buildup.
  Hyperparameters are deliberately conservative (50 epochs, patience=5) to keep
  OOF tractable across 48 (target × matrix) slots.

• Meta-learner is always Logistic Regression: interpretable, SHAP-compatible via
  LinearExplainer, and a 6-dimensional input space will never overfit it.

• OOF generation is the only expensive step. For tuning (Phase 4), OOF is
  pre-computed ONCE with default base learners; Optuna then tunes only meta_C.

Public API
----------
run_stacking_for_tournament(X_train, X_test, y_train, y_test,
                             task_type, n_classes, input_dim, meta_C=1.0)
    → (auc, final_preds, meta, trained_bases, base_names, oof_feat_names)

precompute_oof(X_train, y_train, task_type, n_classes, input_dim)
    → (oof_matrix, trained_bases, base_names)

train_meta_learner(oof_matrix, y_train, meta_C)
    → meta (fitted LogisticRegression)

get_test_meta_features(trained_bases, X_test, task_type, n_classes)
    → test_meta_matrix

get_oof_feature_names(base_names, task_type, n_classes)
    → list[str]
"""

import copy
import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

from src.models.custom_mlp import build_custom_mlp, compute_class_weights

# ── Constants ───────────────────────────────────────────────────────────────
N_FOLDS    = 5
# MLP added as 6th base learner — gives Level 1 a neural signal alongside
# the 5 tree/linear models, improving blind-spot coverage.
BASE_NAMES = ['CB', 'XGB', 'LGB', 'RF', 'LR', 'MLP']


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_sklearn_base_learners(task_type: str, n_classes: int,
                                  y_train: np.ndarray) -> dict:
    """
    Returns fresh untrained instances of the 5 sklearn-compatible base learners.
    MLP is excluded here because it requires input_dim and cannot be deepcopy'd;
    it is handled separately in the OOF loop via _build_mlp_fresh().
    """
    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    spw = neg / pos if pos > 0 else 1.0

    if task_type == 'binary':
        return {
            'CB':  CatBoostClassifier(iterations=150, random_seed=42, verbose=0,
                                      auto_class_weights='Balanced'),
            'XGB': xgb.XGBClassifier(n_estimators=150, use_label_encoder=False,
                                      eval_metric='logloss', scale_pos_weight=spw,
                                      random_state=42, n_jobs=-1),
            'LGB': lgb.LGBMClassifier(n_estimators=150, class_weight='balanced',
                                      random_state=42, n_jobs=-1, verbose=-1),
            'RF':  RandomForestClassifier(n_estimators=100, random_state=42,
                                          class_weight='balanced', n_jobs=-1),
            'LR':  LogisticRegression(max_iter=1000, random_state=42,
                                      class_weight='balanced'),
        }
    else:
        return {
            'CB':  CatBoostClassifier(iterations=150, random_seed=42, verbose=0,
                                      loss_function='MultiClass'),
            'XGB': xgb.XGBClassifier(n_estimators=150, use_label_encoder=False,
                                      eval_metric='mlogloss', objective='multi:softprob',
                                      num_class=n_classes, random_state=42, n_jobs=-1),
            'LGB': lgb.LGBMClassifier(n_estimators=150, class_weight='balanced',
                                      random_state=42, n_jobs=-1, verbose=-1),
            'RF':  RandomForestClassifier(n_estimators=100, random_state=42,
                                          class_weight='balanced', n_jobs=-1),
            'LR':  LogisticRegression(max_iter=1000, random_state=42,
                                      class_weight='balanced'),
        }


def _build_mlp_fresh(input_dim: int, task_type: str, n_classes: int):
    """
    Builds a fresh (uncompiled-graph) Custom MLP for one fold or full retraining.
    Called instead of deepcopy because Keras models cannot be safely deepcopy'd.
    Conservative epochs/patience to keep OOF tractable across 48 tournament slots.
    """
    return build_custom_mlp(
        input_dim=input_dim,
        task_type=task_type,
        n_classes=n_classes
        # uses default units/dropout/lr from custom_mlp.py
    )


def _fit_sklearn(model, bname: str, task_type: str,
                 X_tr: np.ndarray, y_tr: np.ndarray):
    """Fits a single sklearn base learner, handling XGB multiclass weighting."""
    if bname == 'XGB' and task_type == 'multiclass':
        sw = compute_sample_weight('balanced', y_tr)
        model.fit(X_tr, y_tr, sample_weight=sw)
    else:
        model.fit(X_tr, y_tr)
    return model


def _fit_mlp(model, X_tr: np.ndarray, y_tr: np.ndarray):
    """
    Fits the MLP with early stopping and class weighting.
    Fewer epochs (50) and tighter patience (5) vs the standalone tournament MLP
    (100 epochs, patience 10) to keep 5-fold OOF time reasonable.
    """
    cw = compute_class_weights(y_tr)
    early_stop = EarlyStopping(monitor='val_loss', patience=5,
                               restore_best_weights=True)
    model.fit(
        X_tr, y_tr,
        epochs=50, batch_size=64, validation_split=0.15,
        callbacks=[early_stop], class_weight=cw, verbose=0
    )
    return model


def _predict_proba_sklearn(model, bname: str, task_type: str,
                            X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1] if task_type == 'binary' else proba


def _predict_proba_mlp(model, task_type: str, X: np.ndarray) -> np.ndarray:
    preds = model.predict(X, verbose=0)
    return preds.flatten() if task_type == 'binary' else preds


def _oof_matrix_shape(n_rows: int, task_type: str, n_classes: int) -> tuple:
    """
    Binary    → (n_rows, 6)          one probability column per base learner
    Multiclass → (n_rows, 6 * C)     one block of C columns per base learner
    """
    n_bases = len(BASE_NAMES)
    return (n_rows, n_bases) if task_type == 'binary' else (n_rows, n_bases * n_classes)


def _fill_slot(matrix: np.ndarray, row_idx, b_idx: int,
               proba: np.ndarray, task_type: str, n_classes: int):
    """Writes one base learner's probabilities into the correct matrix columns."""
    if task_type == 'binary':
        matrix[row_idx, b_idx] = proba
    else:
        matrix[row_idx, b_idx * n_classes: (b_idx + 1) * n_classes] = proba


# ── Public API ───────────────────────────────────────────────────────────────

def get_oof_feature_names(base_names: list, task_type: str, n_classes: int) -> list:
    """
    Returns human-readable feature names for the meta-learner's input space.
    Binary     : ['CB', 'XGB', 'LGB', 'RF', 'LR', 'MLP']
    Multiclass : ['CB_c0', ..., 'MLP_c{C-1}']
    """
    if task_type == 'binary':
        return list(base_names)
    return [f'{bn}_c{c}' for bn in base_names for c in range(n_classes)]


def precompute_oof(X_train: np.ndarray, y_train: np.ndarray,
                   task_type: str, n_classes: int,
                   input_dim: int) -> tuple:
    """
    Generates OOF predictions for ALL 6 base learners via StratifiedKFold,
    then trains each base learner on the FULL training set.

    The MLP is rebuilt fresh per fold (no deepcopy) and K.clear_session() is
    called after each fold's MLP fit to prevent GPU/memory accumulation.

    Parameters
    ----------
    X_train, y_train : scaled numpy arrays from the tournament pipeline
    task_type        : 'binary' | 'multiclass'
    n_classes        : number of target classes
    input_dim        : number of input features (needed to build MLP graph)

    Returns
    -------
    oof_matrix    : np.ndarray — shape (n_train, 6) or (n_train, 6*C)
    trained_bases : dict  {bname: fitted_model}  trained on FULL X_train
    base_names    : list[str]  — always BASE_NAMES order
    """
    sklearn_defs = _build_sklearn_base_learners(task_type, n_classes, y_train)
    n_train      = len(y_train)
    oof_matrix   = np.zeros(_oof_matrix_shape(n_train, task_type, n_classes),
                            dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # ── sklearn base learners (CB, XGB, LGB, RF, LR) ─────────────────
        for b_idx, (bname, bmodel_template) in enumerate(sklearn_defs.items()):
            fold_model = copy.deepcopy(bmodel_template)
            fold_model = _fit_sklearn(fold_model, bname, task_type, X_tr, y_tr)
            proba      = _predict_proba_sklearn(fold_model, bname, task_type, X_val)
            _fill_slot(oof_matrix, val_idx, b_idx, proba, task_type, n_classes)

        # ── MLP (index 5) — rebuilt fresh, cleared after each fold ────────
        mlp_fold = _build_mlp_fresh(input_dim, task_type, n_classes)
        mlp_fold = _fit_mlp(mlp_fold, X_tr, y_tr)
        mlp_proba = _predict_proba_mlp(mlp_fold, task_type, X_val)
        _fill_slot(oof_matrix, val_idx, 5, mlp_proba, task_type, n_classes)
        K.clear_session()
        gc.collect()

    # ── Retrain everything on the FULL training set ───────────────────────
    trained_bases = {}
    for bname, bmodel_template in sklearn_defs.items():
        full_model = copy.deepcopy(bmodel_template)
        full_model = _fit_sklearn(full_model, bname, task_type, X_train, y_train)
        trained_bases[bname] = full_model

    # MLP full retrain
    mlp_full = _build_mlp_fresh(input_dim, task_type, n_classes)
    mlp_full = _fit_mlp(mlp_full, X_train, y_train)
    trained_bases['MLP'] = mlp_full

    return oof_matrix, trained_bases, BASE_NAMES


def train_meta_learner(oof_matrix: np.ndarray, y_train: np.ndarray,
                       meta_C: float = 1.0) -> LogisticRegression:
    """
    Trains a Logistic Regression meta-learner (Level 1) on the OOF feature matrix.
    Input shape: (n_train, 6) for binary, (n_train, 6*C) for multiclass.
    The 6 columns represent one base learner signal each — MLP included.
    """
    meta = LogisticRegression(
        C=meta_C, max_iter=2000, random_state=42,
        class_weight='balanced', solver='saga'
    )
    meta.fit(oof_matrix, y_train)
    return meta


def get_test_meta_features(trained_bases: dict, X_test: np.ndarray,
                            task_type: str, n_classes: int) -> np.ndarray:
    """
    Builds the (n_test, 6) or (n_test, 6*C) test meta-feature matrix using
    base learners trained on the full training set.
    """
    n_test      = X_test.shape[0]
    test_matrix = np.zeros(_oof_matrix_shape(n_test, task_type, n_classes),
                           dtype=np.float64)

    for b_idx, bname in enumerate(BASE_NAMES):
        model = trained_bases[bname]
        if bname == 'MLP':
            proba = _predict_proba_mlp(model, task_type, X_test)
        else:
            proba = _predict_proba_sklearn(model, bname, task_type, X_test)
        _fill_slot(test_matrix, slice(None), b_idx, proba, task_type, n_classes)

    return test_matrix


def run_stacking_for_tournament(
        X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        task_type: str, n_classes: int,
        input_dim: int,
        meta_C: float = 1.0) -> tuple:
    """
    Complete stacking pipeline for one (target, matrix) slot in the tournament.

    Level 0 base learners : CB, XGB, LGB, RF, LR, MLP  (6 models)
    Level 1 meta-learner  : Logistic Regression

    Parameters
    ----------
    X_train, X_test       : scaled numpy arrays
    y_train, y_test       : encoded label arrays
    task_type             : 'binary' | 'multiclass'
    n_classes             : number of target classes
    input_dim             : X_train.shape[1] — required to build the MLP graph
    meta_C                : regularisation strength for the LR meta-learner

    Returns
    -------
    auc_val        : float
    final_preds    : np.ndarray  (n_test,) binary  |  (n_test, C) multiclass
    meta           : fitted LogisticRegression  (Level 1)
    trained_bases  : dict  {bname: fitted_model}  (Level 0, full-data trained)
    base_names     : list[str]
    oof_feat_names : list[str]
    """
    oof_matrix, trained_bases, base_names = precompute_oof(
        X_train, y_train, task_type, n_classes, input_dim
    )
    meta      = train_meta_learner(oof_matrix, y_train, meta_C)
    test_meta = get_test_meta_features(trained_bases, X_test, task_type, n_classes)

    if task_type == 'binary':
        final_preds = meta.predict_proba(test_meta)[:, 1]
        auc_val     = roc_auc_score(y_test, final_preds)
    else:
        final_preds = meta.predict_proba(test_meta)
        auc_val     = roc_auc_score(y_test, final_preds,
                                    multi_class='ovr', average='macro')

    oof_feat_names = get_oof_feature_names(base_names, task_type, n_classes)
    return auc_val, final_preds, meta, trained_bases, base_names, oof_feat_names