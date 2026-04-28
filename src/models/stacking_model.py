"""
Stacking Ensemble — MIMIC-IV Clinical Prediction Tournament
=============================================================
Implements 5-fold Out-of-Fold (OOF) stacking with 5 base learners
(CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression)
and a Logistic Regression meta-learner.

Design rationale
----------------
• Base learners are the same 5 sklearn-compatible models used in the tournament.
  Custom MLP is excluded as a base learner — OOF over 5 folds × Keras is too
  slow to run 48 times (12 targets × 4 matrices) in a reasonable wall-clock time.
• Meta-learner is always Logistic Regression: interpretable, SHAP-compatible via
  LinearExplainer, and rarely over-fits on a 5-dimensional feature space.
• OOF generation is the only expensive step. For tuning (Phase 4), OOF is
  pre-computed ONCE with default base learners; Optuna then tunes only meta_C.
  This collapses 50 tuning trials from ~25 model fits each to just 1 LR fit each.

Public API
----------
run_stacking_for_tournament(X_train, X_test, y_train, y_test, task_type, n_classes)
    → (auc, final_preds, meta, trained_bases, base_names, oof_feat_names)

precompute_oof(X_train, y_train, task_type, n_classes)
    → (oof_matrix, trained_bases, base_names)

train_meta_learner(oof_matrix, y_train, meta_C)
    → meta (fitted LogisticRegression)

get_test_meta_features(trained_bases, X_test, task_type, n_classes)
    → test_meta_matrix

get_oof_feature_names(base_names, task_type, n_classes)
    → list[str]
"""

import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ── Constants ───────────────────────────────────────────────────────────────
N_FOLDS     = 5          # OOF folds
BASE_NAMES  = ['CB', 'XGB', 'LGB', 'RF', 'LR']   # short keys, stable order


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_base_learners(task_type: str, n_classes: int, y_train: np.ndarray) -> dict:
    """
    Returns a fresh dict of untrained base-learner instances.
    Kept deliberately conservative (fewer iterations) so OOF is fast enough
    to run 48 times in the tournament loop.
    """
    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    spw = neg / pos if pos > 0 else 1.0   # for XGB binary only

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


def _fit_one(model, bname: str, task_type: str, X_tr, y_tr):
    """Fits a single base learner, handling XGB multiclass weighting."""
    if bname == 'XGB' and task_type == 'multiclass':
        sw = compute_sample_weight('balanced', y_tr)
        model.fit(X_tr, y_tr, sample_weight=sw)
    else:
        model.fit(X_tr, y_tr)
    return model


def _predict_proba_base(model, bname: str, task_type: str,
                         X: np.ndarray) -> np.ndarray:
    """Returns probabilities in a shape consistent with our OOF matrix layout."""
    proba = model.predict_proba(X)
    return proba[:, 1] if task_type == 'binary' else proba   # (n,) or (n, C)


def _oof_matrix_shape(n_train: int, n_bases: int, task_type: str, n_classes: int) -> tuple:
    if task_type == 'binary':
        return (n_train, n_bases)
    return (n_train, n_bases * n_classes)


def _fill_oof_slot(matrix, val_idx, b_idx, proba, task_type, n_classes):
    """Writes base-learner probabilities into the correct OOF matrix columns."""
    if task_type == 'binary':
        matrix[val_idx, b_idx] = proba
    else:
        matrix[val_idx, b_idx * n_classes: (b_idx + 1) * n_classes] = proba


def _fill_test_slot(matrix, b_idx, proba, task_type, n_classes):
    if task_type == 'binary':
        matrix[:, b_idx] = proba
    else:
        matrix[:, b_idx * n_classes: (b_idx + 1) * n_classes] = proba


# ── Public API ───────────────────────────────────────────────────────────────

def get_oof_feature_names(base_names: list, task_type: str, n_classes: int) -> list:
    """
    Returns human-readable feature names for the meta-learner's input space.
    Binary  : ['CB', 'XGB', 'LGB', 'RF', 'LR']
    Multiclass: ['CB_c0', 'CB_c1', ..., 'LR_c{C-1}']
    """
    if task_type == 'binary':
        return list(base_names)
    return [f'{bn}_c{c}' for bn in base_names for c in range(n_classes)]


def precompute_oof(X_train: np.ndarray, y_train: np.ndarray,
                   task_type: str, n_classes: int) -> tuple:
    """
    Generates OOF predictions for all base learners via StratifiedKFold,
    then trains each base learner on the FULL training set.

    Parameters
    ----------
    X_train, y_train : scaled numpy arrays from the tournament pipeline
    task_type        : 'binary' | 'multiclass'
    n_classes        : number of target classes

    Returns
    -------
    oof_matrix    : np.ndarray, shape (n_train, n_bases) or (n_train, n_bases * C)
    trained_bases : dict  {bname: fitted_model}  trained on FULL X_train
    base_names    : list[str]  stable order of base learner keys
    """
    base_defs  = _build_base_learners(task_type, n_classes, y_train)
    base_names = list(base_defs.keys())   # ['CB', 'XGB', 'LGB', 'RF', 'LR']
    n_train    = len(y_train)
    oof_shape  = _oof_matrix_shape(n_train, len(base_names), task_type, n_classes)
    oof_matrix = np.zeros(oof_shape, dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        for b_idx, (bname, bmodel_template) in enumerate(base_defs.items()):
            fold_model = copy.deepcopy(bmodel_template)
            fold_model = _fit_one(fold_model, bname, task_type, X_tr, y_tr)
            proba      = _predict_proba_base(fold_model, bname, task_type, X_val)
            _fill_oof_slot(oof_matrix, val_idx, b_idx, proba, task_type, n_classes)

    # Train on full training set (for test-set meta-features)
    trained_bases = {}
    for bname, bmodel_template in base_defs.items():
        full_model = copy.deepcopy(bmodel_template)
        full_model = _fit_one(full_model, bname, task_type, X_train, y_train)
        trained_bases[bname] = full_model

    return oof_matrix, trained_bases, base_names


def train_meta_learner(oof_matrix: np.ndarray, y_train: np.ndarray,
                       meta_C: float = 1.0) -> LogisticRegression:
    """
    Trains a Logistic Regression meta-learner on the OOF feature matrix.
    Works for both binary (input: n×5) and multiclass (input: n×5C).
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
    Builds the test-set meta-feature matrix using base learners trained on
    the full training set.

    Returns
    -------
    np.ndarray shape (n_test, n_bases) for binary
                      (n_test, n_bases * C) for multiclass
    """
    base_names  = list(trained_bases.keys())
    n_test      = X_test.shape[0]
    test_shape  = _oof_matrix_shape(n_test, len(base_names), task_type, n_classes)
    test_matrix = np.zeros(test_shape, dtype=np.float64)

    for b_idx, bname in enumerate(base_names):
        proba = _predict_proba_base(trained_bases[bname], bname, task_type, X_test)
        _fill_test_slot(test_matrix, b_idx, proba, task_type, n_classes)

    return test_matrix


def run_stacking_for_tournament(
        X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        task_type: str, n_classes: int,
        meta_C: float = 1.0) -> tuple:
    """
    Complete stacking pipeline for one (target, matrix) slot in the tournament.

    Steps
    -----
    1. Generate OOF matrix + train base learners on full training set
    2. Train meta-learner on OOF matrix
    3. Build test meta-features from fully-trained base learners
    4. Predict with meta-learner and compute AUC

    Returns
    -------
    auc_val        : float
    final_preds    : np.ndarray  (n_test,) binary  |  (n_test, C) multiclass
    meta           : fitted LogisticRegression  (meta-learner)
    trained_bases  : dict  {bname: fitted_model}
    base_names     : list[str]
    oof_feat_names : list[str]  feature names for the meta-learner's input space
    """
    oof_matrix, trained_bases, base_names = precompute_oof(
        X_train, y_train, task_type, n_classes
    )
    meta       = train_meta_learner(oof_matrix, y_train, meta_C)
    test_meta  = get_test_meta_features(trained_bases, X_test, task_type, n_classes)

    if task_type == 'binary':
        final_preds = meta.predict_proba(test_meta)[:, 1]
        auc_val     = roc_auc_score(y_test, final_preds)
    else:
        final_preds = meta.predict_proba(test_meta)
        auc_val     = roc_auc_score(y_test, final_preds,
                                    multi_class='ovr', average='macro')

    oof_feat_names = get_oof_feature_names(base_names, task_type, n_classes)
    return auc_val, final_preds, meta, trained_bases, base_names, oof_feat_names
