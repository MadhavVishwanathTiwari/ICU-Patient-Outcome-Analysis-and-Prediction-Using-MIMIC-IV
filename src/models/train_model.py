"""
MIMIC-IV Model Training Pipeline
=================================
Trains and evaluates machine learning models for ICU outcome prediction.

Tasks:
1. Mortality prediction (binary classification)
2. Length of stay prediction (regression)
3. LOS category prediction (multi-class classification)

Models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (primary model)

Author: ML Healthcare Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import json
import warnings
import os  # <--- 1. ADD THIS

warnings.filterwarnings('ignore')

# 2. ADD THIS LINE (Forces joblib to skip the failing Windows subprocess command)
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


# ML libraries
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *


class ModelTrainer:
    """
    Trains and evaluates machine learning models for ICU outcome prediction.
    """
    
    def __init__(self, features_path='data/processed/features_engineered.csv'):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        features_path : str
            Path to the engineered features CSV file
        """
        self.features_path = Path(features_path)
        self.data = None
        self.models = {}
        self.results = {}
        self.scaler = None
        
    def load_data(self):
        """Load the engineered features."""
        print("\n" + "=" * 70)
        print("LOADING FEATURE MATRIX")
        print("=" * 70)
        
        self.data = pd.read_csv(self.features_path)
        
        # Clean column names for XGBoost compatibility
        # XGBoost doesn't allow [, ], < in feature names
        print("  -> Cleaning feature names for XGBoost compatibility...")
        self.data.columns = [
            col.replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt')
            for col in self.data.columns
        ]
        
        print(f"[OK] Loaded {len(self.data):,} samples with {len(self.data.columns)} features")
        
        return self.data
    
    # All possible target columns across the pipeline
    ALL_TARGET_COLS = [
        'mortality', 'los_days', 'los_category',
        'icu_readmit_48h', 'icu_readmit_7d', 'discharge_disposition',
        'need_vent_any', 'need_vasopressor_any', 'need_rrt_any',
        'aki_onset', 'ards_onset', 'liver_injury_onset', 'sepsis_onset'
    ]

    def prepare_data(self, task='mortality'):
        """
        Prepare data for modeling.  Accepts any target column name.
        """
        print("\n" + "=" * 70)
        print(f"PREPARING DATA FOR: {task.upper()}")
        print("=" * 70)
        
        id_cols = ['subject_id', 'hadm_id', 'stay_id']
        drop_cols = id_cols + [c for c in self.ALL_TARGET_COLS if c in self.data.columns]
        
        X = self.data.drop(columns=[c for c in drop_cols if c in self.data.columns])
        
        if task == 'los_regression':
            y = self.data['los_days']
        elif task in self.data.columns:
            y = self.data[task]
        else:
            raise ValueError(f"Unknown task / column not found: {task}")
        
        MULTICLASS_TASKS = {'los_category', 'discharge_disposition'}

        print(f"  [OK] Features shape: {X.shape}")
        print(f"  [OK] Target distribution:")
        if task == 'los_regression':
            print(f"       Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        elif task in MULTICLASS_TASKS:
            counts = y.value_counts().sort_index().to_dict()
            print(f"       {counts}")
        else:
            counts = y.value_counts().sort_index().to_dict()
            print(f"       {counts}")
            print(f"       Positive rate: {y.mean()*100:.1f}%")
        
        # Split data: 60% train, 20% validation, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y if task != 'los_regression' else None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=RANDOM_SEED, stratify=y_temp if task != 'los_regression' else None
        )
        
        print(f"\n  [OK] Data split:")
        print(f"       Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"       Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"       Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Scale features
        print(f"\n  -> Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for interpretability
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_val = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        print(f"     [OK] Features scaled")
        
        # Handle class imbalance for mortality
        if task == 'mortality' and HANDLE_IMBALANCE:
            print(f"\n  -> Handling class imbalance with SMOTE...")
            
            # Downcast to save RAM
            X_train = X_train.astype(np.float32)
            
            # Force multi-threading using the NearestNeighbors object
            knn = NearestNeighbors(n_jobs=-1)
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=knn)
            
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"     [OK] Resampled training set: {len(X_train):,} samples")
            print(f"     New distribution: {pd.Series(y_train).value_counts().to_dict()}")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_mortality_models(self):
        """
        Train models for mortality prediction.
        """
        self._train_binary_models('Mortality Prediction', 'mortality', 'mortality')
        return self.results.get('mortality', {})

    def train_los_classification_models(self):
        """
        Train models for LOS category prediction (multi-class).
        """
        self._train_multiclass_models('LOS Category Prediction', 'los_category', 'los_class')
        return self.results.get('los_category', {})
        
    # ------------------------------------------------------------------
    # Generic binary XGBoost trainer (reused for new targets)
    # ------------------------------------------------------------------

    # Targets where SMOTE hurts (insufficient temporal signal; oversampling adds noise)
    SMOTE_SKIP_TARGETS = {'icu_readmit_48h', 'icu_readmit_7d'}

    def _get_model_dict(self, problem_type='binary', scale_pos_weight=1.0):
        """Return model dictionary for binary or multiclass classification."""
        if problem_type == 'binary':
            return {
                'logistic_regression': LogisticRegression(
                    max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced'
                ),
                'random_forest': RandomForestClassifier(
                    **RANDOM_FOREST_PARAMS, class_weight='balanced'
                ),
                'xgboost': xgb.XGBClassifier(
                    **XGBOOST_PARAMS, eval_metric='logloss', scale_pos_weight=scale_pos_weight
                ),
                'catboost': CatBoostClassifier(
                    iterations=1000, random_seed=RANDOM_SEED, verbose=0,
                    eval_metric='Logloss', auto_class_weights='Balanced'
                )
            }
        return {
            'logistic_regression': LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            'xgboost': xgb.XGBClassifier(**XGBOOST_PARAMS, eval_metric='mlogloss'),
            'catboost': CatBoostClassifier(
                iterations=1000, random_seed=RANDOM_SEED, loss_function='MultiClass', verbose=0
            )
        }

    def _save_binary_roc_plot(self, task_name, y_true, proba_map, output_dir='reports/figures/roc'):
        """Save ROC comparison plot for binary tasks."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        for model_name, probs in proba_map.items():
            fpr, tpr, _ = roc_curve(y_true, probs)
            model_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={model_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison - {task_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        out_path = output_dir / f'{task_name}_roc_comparison.png'
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"  [OK] Saved ROC plot: {out_path}")

    def _save_multiclass_roc_plot(self, task_name, y_true, proba_map, output_dir='reports/figures/roc'):
        """Save one-vs-rest micro-average ROC comparison plot for multiclass tasks."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)

        plt.figure(figsize=(8, 6))
        for model_name, probs in proba_map.items():
            fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
            model_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (micro-AUC={model_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multiclass ROC (One-vs-Rest) - {task_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        out_path = output_dir / f'{task_name}_roc_comparison.png'
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"  [OK] Saved ROC plot: {out_path}")

    def _train_binary_models(self, task_name: str, target_col: str, model_prefix: str):
        """Train LR/RF/XGB/CatBoost models for a binary target."""
        if target_col not in self.data.columns:
            print(f"  [SKIP] {target_col} not in data")
            return
        if self.data[target_col].nunique() < 2:
            print(f"  [SKIP] {target_col} has only one class")
            return

        print(f"\n{'=' * 70}")
        print(f"TRAINING: {task_name}")
        print(f"{'=' * 70}")

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(task=target_col)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
        pos_rate = n_pos / (n_neg + n_pos)

        use_smote = (pos_rate < 0.10 and HANDLE_IMBALANCE and target_col not in self.SMOTE_SKIP_TARGETS)
        if use_smote:
            print(f"  -> Positive rate {pos_rate*100:.1f}% - applying SMOTE + scale_pos_weight={spw}")
            X_train = X_train.astype(np.float32)
            knn = NearestNeighbors(n_jobs=-1)
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=knn)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
            print(f"     [OK] After SMOTE: {len(X_train):,} samples, spw={spw}")
        else:
            print(f"  -> scale_pos_weight={spw}")

        models = self._get_model_dict(problem_type='binary', scale_pos_weight=spw)
        val_results = {}
        test_results = {}
        val_proba_map = {}

        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n[{i}/4] Training {model_name}...")
            if model_name == 'catboost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
            else:
                model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]
            val_proba_map[model_name] = val_proba
            val_results[model_name] = {
                'accuracy': float(accuracy_score(y_val, val_pred)),
                'precision': float(precision_score(y_val, val_pred, zero_division=0)),
                'recall': float(recall_score(y_val, val_pred, zero_division=0)),
                'f1': float(f1_score(y_val, val_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_val, val_proba))
            }

            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1]
            test_results[model_name] = {
                'accuracy': float(accuracy_score(y_test, test_pred)),
                'precision': float(precision_score(y_test, test_pred, zero_division=0)),
                'recall': float(recall_score(y_test, test_pred, zero_division=0)),
                'f1': float(f1_score(y_test, test_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, test_proba))
            }

            model_key = f'{model_prefix}_{model_name}'
            self.models[model_key] = model
            print(f"   [OK] {model_name} - VAL ROC-AUC: {val_results[model_name]['roc_auc']:.4f}")

        self._save_binary_roc_plot(task_name=target_col, y_true=y_val, proba_map=val_proba_map)
        best_model_name = max(val_results, key=lambda k: val_results[k]['roc_auc'])
        self.results[target_col] = {
            'validation': val_results,
            'test': test_results[best_model_name],
            'best_model': best_model_name
        }
        print(f"\n  [BEST MODEL] {best_model_name.upper()} with ROC-AUC: {val_results[best_model_name]['roc_auc']:.4f}")

    def _train_multiclass_models(self, task_name: str, target_col: str, model_prefix: str):
        """Train LR/RF/XGB/CatBoost models for a multiclass target."""
        if target_col not in self.data.columns:
            print(f"  [SKIP] {target_col} not in data")
            return

        print(f"\n{'=' * 70}")
        print(f"TRAINING: {task_name}")
        print(f"{'=' * 70}")

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(task=target_col)
        models = self._get_model_dict(problem_type='multiclass')
        val_results = {}
        test_results = {}
        val_proba_map = {}
        classes = np.unique(y_train)

        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n[{i}/4] Training {model_name}...")
            if model_name == 'catboost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
                val_pred = model.predict(X_val).flatten()
                test_pred = model.predict(X_test).flatten()
            else:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)

            val_proba = model.predict_proba(X_val)
            test_proba = model.predict_proba(X_test)
            val_proba_map[model_name] = val_proba

            val_results[model_name] = {
                'accuracy': float(accuracy_score(y_val, val_pred)),
                'precision_macro': float(precision_score(y_val, val_pred, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_val, val_pred, average='macro', zero_division=0)),
                'f1_macro': float(f1_score(y_val, val_pred, average='macro', zero_division=0)),
                'roc_auc_ovr_macro': float(roc_auc_score(y_val, val_proba, multi_class='ovr', average='macro', labels=classes))
            }
            test_results[model_name] = {
                'accuracy': float(accuracy_score(y_test, test_pred)),
                'precision_macro': float(precision_score(y_test, test_pred, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_test, test_pred, average='macro', zero_division=0)),
                'f1_macro': float(f1_score(y_test, test_pred, average='macro', zero_division=0)),
                'roc_auc_ovr_macro': float(roc_auc_score(y_test, test_proba, multi_class='ovr', average='macro', labels=classes))
            }
            self.models[f'{model_prefix}_{model_name}'] = model
            print(f"   [OK] {model_name} - VAL ROC-AUC(OVR-macro): {val_results[model_name]['roc_auc_ovr_macro']:.4f}")

        self._save_multiclass_roc_plot(task_name=target_col, y_true=y_val, proba_map=val_proba_map)
        best_model_name = max(val_results, key=lambda k: val_results[k]['roc_auc_ovr_macro'])
        self.results[target_col] = {
            'validation': val_results,
            'test': test_results[best_model_name],
            'best_model': best_model_name
        }
        print(f"\n  [BEST MODEL] {best_model_name.upper()} with ROC-AUC(OVR-macro): {val_results[best_model_name]['roc_auc_ovr_macro']:.4f}")

    def _train_binary_xgb(self, task_name: str, target_col: str):
        """Train a single XGBoost binary classifier for *target_col*.

        Imbalance handling strategy:
          - scale_pos_weight = n_neg/n_pos for all binary targets
          - SMOTE applied when positive rate < 10%, EXCEPT for readmission
            targets where it was shown to degrade performance
        """
        if target_col not in self.data.columns:
            print(f"  [SKIP] {target_col} not in data")
            return
        if self.data[target_col].nunique() < 2:
            print(f"  [SKIP] {target_col} has only one class")
            return

        print(f"\n{'=' * 70}")
        print(f"TRAINING: {task_name}")
        print(f"{'=' * 70}")

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(task=target_col)

        # Compute class weight ratio for XGBoost
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
        pos_rate = n_pos / (n_neg + n_pos)

        # SMOTE for severely imbalanced targets, skipped where it degrades signal
        use_smote = (pos_rate < 0.10
                     and HANDLE_IMBALANCE
                     and target_col not in self.SMOTE_SKIP_TARGETS)
        if use_smote:
            print(f"  -> Positive rate {pos_rate*100:.1f}% - applying SMOTE + scale_pos_weight={spw}")
            
            # Downcast to save RAM
            X_train = X_train.astype(np.float32)
            
            # Force multi-threading
            knn = NearestNeighbors(n_jobs=-1)
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=knn)
            
            X_train, y_train = smote.fit_resample(X_train, y_train)
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            spw = round(n_neg / n_pos, 2)
            print(f"     [OK] After SMOTE: {len(X_train):,} samples, spw={spw}")
        else:
            if target_col in self.SMOTE_SKIP_TARGETS:
                print(f"  -> Positive rate {pos_rate*100:.1f}% - SMOTE skipped (readmission); scale_pos_weight={spw}")
            else:
                print(f"  -> scale_pos_weight={spw}")

        model = xgb.XGBClassifier(**XGBOOST_PARAMS, eval_metric='logloss',
                                   scale_pos_weight=spw)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]

        val_metrics = {
            'accuracy': float(accuracy_score(y_val, val_pred)),
            'precision': float(precision_score(y_val, val_pred, zero_division=0)),
            'recall': float(recall_score(y_val, val_pred, zero_division=0)),
            'f1': float(f1_score(y_val, val_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_val, val_proba))
        }
        print(f"  [VAL] ROC-AUC: {val_metrics['roc_auc']:.4f}  F1: {val_metrics['f1']:.4f}")

        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = {
            'accuracy': float(accuracy_score(y_test, test_pred)),
            'precision': float(precision_score(y_test, test_pred, zero_division=0)),
            'recall': float(recall_score(y_test, test_pred, zero_division=0)),
            'f1': float(f1_score(y_test, test_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, test_proba))
        }
        print(f"  [TEST] ROC-AUC: {test_metrics['roc_auc']:.4f}  F1: {test_metrics['f1']:.4f}")

        model_key = target_col + '_xgb'
        self.models[model_key] = model
        self.results[target_col] = {
            'validation': val_metrics,
            'test': test_metrics,
            'best_model': 'xgboost'
        }

    def train_readmission_models(self):
        self._train_binary_models('ICU Readmission 48h', 'icu_readmit_48h', 'icu_readmit_48h')
        self._train_binary_models('ICU Readmission 7d', 'icu_readmit_7d', 'icu_readmit_7d')

    def train_discharge_disposition_model(self):
        self._train_multiclass_models(
            'Discharge Disposition (3-class)', 'discharge_disposition', 'discharge_disposition'
        )

    def train_organ_support_models(self):
        self._train_binary_models('Ventilation Need', 'need_vent_any', 'need_vent_any')
        self._train_binary_models('Vasopressor Need', 'need_vasopressor_any', 'need_vasopressor_any')
        self._train_binary_models('RRT Need', 'need_rrt_any', 'need_rrt_any')

    def train_disease_onset_models(self):
        self._train_binary_models('AKI Onset', 'aki_onset', 'aki_onset')
        self._train_binary_models('ARDS Onset', 'ards_onset', 'ards_onset')
        self._train_binary_models('Liver Injury Onset', 'liver_injury_onset', 'liver_injury_onset')
        self._train_binary_models('Sepsis Onset', 'sepsis_onset', 'sepsis_onset')

    # ------------------------------------------------------------------

    def train_all_models(self):
        """Train models for all prediction tasks."""
        print("\n" + "=" * 70)
        print("MIMIC-IV MODEL TRAINING PIPELINE")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_data()
        
        # Core targets
        self.train_mortality_models()
        self.train_los_classification_models()
        
        # Extended targets
        self.train_readmission_models()
        self.train_discharge_disposition_model()
        self.train_organ_support_models()
        self.train_disease_onset_models()
        
        print("\n" + "=" * 70)
        print("MODEL TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Trained {len(self.models)} models")
        task_list = list(self.results.keys())
        print(f"  Tasks completed: {len(task_list)} ({', '.join(task_list)})")
        
        return self.results
    
    def save_models(self, output_dir='models'):
        """
        Save trained models and results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n>> Saving models to: {output_dir}")
        
        # Save each model
        for model_name, model in self.models.items():
            model_path = output_dir / f'{model_name}.pkl'
            joblib.dump(model, model_path)
            print(f"   [OK] Saved {model_name}")
        
        # Save scaler
        scaler_path = output_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"   [OK] Saved scaler")
        
        # Save results
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"   [OK] Saved results")
        
        print(f"\n   [OK] All models saved to: {output_dir.absolute()}")


def main():
    """
    Main execution function.
    """
    # Initialize trainer
    trainer = ModelTrainer(features_path='data/processed/features_engineered.csv')
    
    # Train all models
    results = trainer.train_all_models()
    
    # Save models
    trainer.save_models()
    
    print("\n*** Model training complete! Ready for dashboard development.")


if __name__ == "__main__":
    main()
