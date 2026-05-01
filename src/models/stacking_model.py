"""
Stacking Ensemble — MIMIC-IV Clinical Prediction Tournament
=============================================================
Implements 5-fold Out-of-Fold (OOF) stacking with 7 base learners
(CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression,
Custom MLP, FT-Transformer) and a Logistic Regression meta-learner.

Change log vs previous version
-------------------------------
v2: Added FT-Transformer (FTT) as the 7th base learner at index 6.

    Motivation:
    ──────────
    The five tree/linear models and the MLP provide complementary signals,
    but all share one characteristic: they process features independently
    or via coarse interactions (splits/dot-products).  The FT-Transformer
    uses full pairwise self-attention across ALL feature tokens on every
    forward pass, learning high-order feature interactions explicitly.
    This is particularly valuable for the hard readmission targets
    (icu_readmit_48h ~0.59, icu_readmit_7d ~0.61) where tree ensembles
    plateau and any additional diversity is worth having at Level 1.

    Implementation constraints respected:
    ──────────────────────────────────────
    • FTT uses PyTorch, MLP uses TF/Keras → no framework conflicts.
    • FTT is rebuilt fresh per fold, deleted + gc.collect() after each fold,
      mirroring the MLP's lifecycle exactly.
    • torch.cuda / MPS auto-detection via get_device() in ft_transformer.py.
    • Public API signatures are UNCHANGED — all callers (run_full_tournament,
      tune_winners, shap_analysis) require zero edits.

    OOF matrix shape change:
    ────────────────────────
    Binary     : (n_train,  6) → (n_train,  7)
    Multiclass : (n_train, 6C) → (n_train, 7C)
    The meta-learner (LogisticRegression) sees one extra column; its 7-column
    input space is still trivially small relative to any training set size.

Design rationale (carried forward)
-----------------------------------
• Base learners are sklearn-compatible tree/linear models PLUS two neural
  models (MLP + FTT).  Axis-aligned splits and smooth non-linear boundaries
  are wrong in different regions; the meta-learner exploits the diversity.

• MLP and FTT are handled separately from the sklearn models in the OOF loop:
    – Neither can be reliably deepcopy'd.
    – Both must be rebuilt fresh per fold to avoid weight leakage.
    – MLP: K.clear_session() + gc.collect() after each fold.
    – FTT: del model + gc.collect() (+ torch.cuda.empty_cache() if CUDA).

• Meta-learner is always Logistic Regression: interpretable, SHAP-compatible
  via LinearExplainer, and a 7-column input space will never overfit it.

• OOF generation is the only expensive step.  For tuning (Phase 4), OOF is
  pre-computed ONCE with default base learners; Optuna tunes only meta_C.

Public API  (signatures UNCHANGED from v1)
------------------------------------------
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
import torch

from src.models.custom_mlp import build_custom_mlp, compute_class_weights
from src.models.ft_transformer import (
    build_ftt, train_ftt, predict_ftt, get_device
)

# ── Constants ───────────────────────────────────────────────────────────────
N_FOLDS = 5

# FTT added as 7th base learner (index 6).  Order is load-bearing: _fill_slot
# uses b_idx directly, so BASE_NAMES must never be re-sorted.
BASE_NAMES = ['CB', 'XGB', 'LGB', 'RF', 'LR', 'MLP', 'FTT']

# Lazily resolved once at first use; avoids repeated CUDA/MPS probe overhead
_DEVICE: torch.device | None = None


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = get_device()
    return _DEVICE


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_sklearn_base_learners(task_type: str, n_classes: int,
                                  y_train: np.ndarray) -> dict:
    """
    Returns fresh untrained instances of the 5 sklearn-compatible base learners.
    MLP and FTT are excluded because they cannot be deepcopy'd and require
    input_dim at construction time; both are handled separately in the OOF loop.
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
    """Builds a fresh Custom MLP for one fold.  Never deepcopy'd."""
    return build_custom_mlp(
        input_dim=input_dim,
        task_type=task_type,
        n_classes=n_classes,
    )


def _build_ftt_fresh(input_dim: int, task_type: str, n_classes: int):
    """
    Builds a fresh FT-Transformer for one fold.  Never deepcopy'd.
    Conservative defaults (n_blocks=2, d_token=64, n_heads=4) keep OOF
    runtime tractable while still providing genuine attention-based signal.
    """
    return build_ftt(
        input_dim=input_dim,
        task_type=task_type,
        n_classes=n_classes,
        d_token=64,
        n_heads=4,
        n_blocks=2,
        ffn_d_hidden=128,
        attn_dropout=0.1,
        ffn_dropout=0.1,
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
    Fits the Keras MLP with early stopping and class weighting.
    Fewer epochs (50) and tighter patience (5) vs standalone tournament MLP
    to keep 5-fold OOF time reasonable.
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


def _fit_ftt(model, X_tr: np.ndarray, y_tr: np.ndarray,
             device: torch.device):
    """
    Trains FT-Transformer via train_ftt().  Identical budget to the OOF MLP
    (50 epochs, patience=5) so wall-clock time is comparable.
    """
    return train_ftt(
        model=model,
        X_tr=X_tr,
        y_tr=y_tr,
        device=device,
        epochs=50,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5,
        val_fraction=0.15,
    )


def _predict_proba_sklearn(model, bname: str, task_type: str,
                            X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    return proba[:, 1] if task_type == 'binary' else proba


def _predict_proba_mlp(model, task_type: str, X: np.ndarray) -> np.ndarray:
    preds = model.predict(X, verbose=0)
    return preds.flatten() if task_type == 'binary' else preds


def _predict_proba_ftt(model, task_type: str,
                        X: np.ndarray, device: torch.device) -> np.ndarray:
    return predict_ftt(model, X, device, task_type)


def _oof_matrix_shape(n_rows: int, task_type: str, n_classes: int) -> tuple:
    """
    Binary     → (n_rows,  7)       one probability column per base learner
    Multiclass → (n_rows,  7 * C)   one block of C columns per base learner
    """
    n_bases = len(BASE_NAMES)   # 7
    return (n_rows, n_bases) if task_type == 'binary' else (n_rows, n_bases * n_classes)


def _fill_slot(matrix: np.ndarray, row_idx, b_idx: int,
               proba: np.ndarray, task_type: str, n_classes: int):
    """Writes one base learner's probabilities into the correct matrix columns."""
    if task_type == 'binary':
        matrix[row_idx, b_idx] = proba
    else:
        matrix[row_idx, b_idx * n_classes: (b_idx + 1) * n_classes] = proba


def _cleanup_ftt(model) -> None:
    """
    Deterministically frees FTT GPU/CPU memory after each fold.
    Mirrors K.clear_session() + gc.collect() used for the MLP.
    """
    device = next(model.parameters()).device
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ── Public API ───────────────────────────────────────────────────────────────

def get_oof_feature_names(base_names: list, task_type: str, n_classes: int) -> list:
    """
    Returns human-readable feature names for the meta-learner's input space.
    Binary     : ['CB', 'XGB', 'LGB', 'RF', 'LR', 'MLP', 'FTT']
    Multiclass : ['CB_c0', ..., 'FTT_c{C-1}']
    """
    if task_type == 'binary':
        return list(base_names)
    return [f'{bn}_c{c}' for bn in base_names for c in range(n_classes)]


def precompute_oof(X_train: np.ndarray, y_train: np.ndarray,
                   task_type: str, n_classes: int,
                   input_dim: int) -> tuple:
    """
    Generates OOF predictions for ALL 7 base learners via StratifiedKFold,
    then retrains each base learner on the FULL training set.

    Fold lifecycle for neural models
    ---------------------------------
    MLP (index 5):
        rebuilt via _build_mlp_fresh → trained → predict → K.clear_session() + gc.collect()

    FTT (index 6):
        rebuilt via _build_ftt_fresh → trained on device → predict → _cleanup_ftt()
        _cleanup_ftt calls del + gc.collect() + torch.cuda.empty_cache() if CUDA

    Parameters
    ----------
    X_train, y_train : scaled numpy arrays from the tournament pipeline
    task_type        : 'binary' | 'multiclass'
    n_classes        : number of target classes
    input_dim        : number of input features (needed to build MLP + FTT graphs)

    Returns
    -------
    oof_matrix    : np.ndarray — shape (n_train, 7) or (n_train, 7*C)
    trained_bases : dict  {bname: fitted_model}  trained on FULL X_train
    base_names    : list[str]  — always BASE_NAMES order
    """
    device       = _get_device()
    sklearn_defs = _build_sklearn_base_learners(task_type, n_classes, y_train)
    n_train      = len(y_train)
    oof_matrix   = np.zeros(_oof_matrix_shape(n_train, task_type, n_classes),
                            dtype=np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # ── sklearn base learners (CB, XGB, LGB, RF, LR) — indices 0-4 ───
        for b_idx, (bname, bmodel_template) in enumerate(sklearn_defs.items()):
            fold_model = copy.deepcopy(bmodel_template)
            fold_model = _fit_sklearn(fold_model, bname, task_type, X_tr, y_tr)
            proba      = _predict_proba_sklearn(fold_model, bname, task_type, X_val)
            _fill_slot(oof_matrix, val_idx, b_idx, proba, task_type, n_classes)

        # ── MLP (index 5) — rebuilt fresh, Keras session cleared after ────
        mlp_fold  = _build_mlp_fresh(input_dim, task_type, n_classes)
        mlp_fold  = _fit_mlp(mlp_fold, X_tr, y_tr)
        mlp_proba = _predict_proba_mlp(mlp_fold, task_type, X_val)
        _fill_slot(oof_matrix, val_idx, 5, mlp_proba, task_type, n_classes)
        del mlp_fold
        K.clear_session()
        gc.collect()

        # ── FTT (index 6) — rebuilt fresh, GPU memory freed after ─────────
        ftt_fold  = _build_ftt_fresh(input_dim, task_type, n_classes)
        ftt_fold  = _fit_ftt(ftt_fold, X_tr, y_tr, device)
        ftt_proba = _predict_proba_ftt(ftt_fold, task_type, X_val, device)
        _fill_slot(oof_matrix, val_idx, 6, ftt_proba, task_type, n_classes)
        _cleanup_ftt(ftt_fold)

    # ── Retrain everything on the FULL training set ───────────────────────
    trained_bases: dict = {}

    for bname, bmodel_template in sklearn_defs.items():
        full_model = copy.deepcopy(bmodel_template)
        full_model = _fit_sklearn(full_model, bname, task_type, X_train, y_train)
        trained_bases[bname] = full_model

    # MLP full retrain
    mlp_full = _build_mlp_fresh(input_dim, task_type, n_classes)
    mlp_full = _fit_mlp(mlp_full, X_train, y_train)
    trained_bases['MLP'] = mlp_full
    # Note: K.clear_session() is intentionally NOT called here — the full-data
    # MLP must remain live for get_test_meta_features().

    # FTT full retrain
    ftt_full = _build_ftt_fresh(input_dim, task_type, n_classes)
    ftt_full = _fit_ftt(ftt_full, X_train, y_train, device)
    trained_bases['FTT'] = ftt_full
    # Note: _cleanup_ftt() intentionally NOT called — model must stay live.

    return oof_matrix, trained_bases, BASE_NAMES


def train_meta_learner(oof_matrix: np.ndarray, y_train: np.ndarray,
                       meta_C: float = 1.0) -> LogisticRegression:
    """
    Trains a Logistic Regression meta-learner (Level 1) on the OOF feature matrix.
    Input shape: (n_train, 7) for binary, (n_train, 7*C) for multiclass.
    The 7 columns represent one base learner signal each — MLP and FTT included.
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
    Builds the (n_test, 7) or (n_test, 7*C) test meta-feature matrix using
    base learners trained on the full training set.
    """
    device      = _get_device()
    n_test      = X_test.shape[0]
    test_matrix = np.zeros(_oof_matrix_shape(n_test, task_type, n_classes),
                           dtype=np.float64)

    for b_idx, bname in enumerate(BASE_NAMES):
        model = trained_bases[bname]
        if bname == 'MLP':
            proba = _predict_proba_mlp(model, task_type, X_test)
        elif bname == 'FTT':
            proba = _predict_proba_ftt(model, task_type, X_test, device)
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

    Level 0 base learners : CB, XGB, LGB, RF, LR, MLP, FTT  (7 models)
    Level 1 meta-learner  : Logistic Regression

    Parameters
    ----------
    X_train, X_test       : scaled numpy arrays
    y_train, y_test       : encoded label arrays
    task_type             : 'binary' | 'multiclass'
    n_classes             : number of target classes
    input_dim             : X_train.shape[1] — required to build MLP + FTT graphs
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
