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
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

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
        
        # Handle class imbalance for mortality only (SMOTE used in _train_binary_xgb for others)
        if task == 'mortality' and HANDLE_IMBALANCE:
            print(f"\n  -> Handling class imbalance with SMOTE...")
            smote = SMOTE(random_state=RANDOM_SEED)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"     [OK] Resampled training set: {len(X_train):,} samples")
            print(f"     New distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_mortality_models(self):
        """
        Train models for mortality prediction.
        """
        print("\n" + "=" * 70)
        print("TRAINING MORTALITY PREDICTION MODELS")
        print("=" * 70)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(task='mortality')
        
        results = {}
        
        # 1. Logistic Regression (Baseline)
        print("\n[1/3] Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
        lr_model.fit(X_train, y_train)
        
        lr_pred = lr_model.predict(X_val)
        lr_proba = lr_model.predict_proba(X_val)[:, 1]
        
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_val, lr_pred),
            'precision': precision_score(y_val, lr_pred),
            'recall': recall_score(y_val, lr_pred),
            'f1': f1_score(y_val, lr_pred),
            'roc_auc': roc_auc_score(y_val, lr_proba)
        }
        self.models['mortality_lr'] = lr_model
        
        print(f"   [OK] Logistic Regression - ROC-AUC: {results['logistic_regression']['roc_auc']:.4f}")
        
        # 2. Random Forest
        print("\n[2/3] Training Random Forest...")
        rf_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_val)
        rf_proba = rf_model.predict_proba(X_val)[:, 1]
        
        results['random_forest'] = {
            'accuracy': accuracy_score(y_val, rf_pred),
            'precision': precision_score(y_val, rf_pred),
            'recall': recall_score(y_val, rf_pred),
            'f1': f1_score(y_val, rf_pred),
            'roc_auc': roc_auc_score(y_val, rf_proba)
        }
        self.models['mortality_rf'] = rf_model
        
        print(f"   [OK] Random Forest - ROC-AUC: {results['random_forest']['roc_auc']:.4f}")
        
        # 3. XGBoost
        print("\n[3/3] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        
        xgb_pred = xgb_model.predict(X_val)
        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
        
        results['xgboost'] = {
            'accuracy': accuracy_score(y_val, xgb_pred),
            'precision': precision_score(y_val, xgb_pred),
            'recall': recall_score(y_val, xgb_pred),
            'f1': f1_score(y_val, xgb_pred),
            'roc_auc': roc_auc_score(y_val, xgb_proba)
        }
        self.models['mortality_xgb'] = xgb_model
        
        print(f"   [OK] XGBoost - ROC-AUC: {results['xgboost']['roc_auc']:.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
        print(f"\n  [BEST MODEL] {best_model_name.upper()} with ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        
        # Evaluate best model on test set
        print(f"\n  -> Evaluating best model on test set...")
        if best_model_name == 'logistic_regression':
            best_model = lr_model
        elif best_model_name == 'random_forest':
            best_model = rf_model
        else:
            best_model = xgb_model
        
        test_pred = best_model.predict(X_test)
        test_proba = best_model.predict_proba(X_test)[:, 1]
        
        test_results = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred),
            'roc_auc': roc_auc_score(y_test, test_proba)
        }
        
        print(f"     [TEST SET RESULTS]")
        print(f"     Accuracy:  {test_results['accuracy']:.4f}")
        print(f"     Precision: {test_results['precision']:.4f}")
        print(f"     Recall:    {test_results['recall']:.4f}")
        print(f"     F1-Score:  {test_results['f1']:.4f}")
        print(f"     ROC-AUC:   {test_results['roc_auc']:.4f}")
        
        self.results['mortality'] = {
            'validation': results,
            'test': test_results,
            'best_model': best_model_name
        }
        
        return results
    
    def train_los_classification_models(self):
        """
        Train models for LOS category prediction (multi-class).
        """
        print("\n" + "=" * 70)
        print("TRAINING LENGTH OF STAY CLASSIFICATION MODELS")
        print("=" * 70)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(task='los_category')
        
        results = {}
        
        # 1. Logistic Regression
        print("\n[1/3] Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
        lr_model.fit(X_train, y_train)
        
        lr_pred = lr_model.predict(X_val)
        
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_val, lr_pred),
            'precision_macro': precision_score(y_val, lr_pred, average='macro'),
            'recall_macro': recall_score(y_val, lr_pred, average='macro'),
            'f1_macro': f1_score(y_val, lr_pred, average='macro')
        }
        self.models['los_class_lr'] = lr_model
        
        print(f"   [OK] Logistic Regression - Accuracy: {results['logistic_regression']['accuracy']:.4f}")
        
        # 2. Random Forest
        print("\n[2/3] Training Random Forest...")
        rf_model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_val)
        
        results['random_forest'] = {
            'accuracy': accuracy_score(y_val, rf_pred),
            'precision_macro': precision_score(y_val, rf_pred, average='macro'),
            'recall_macro': recall_score(y_val, rf_pred, average='macro'),
            'f1_macro': f1_score(y_val, rf_pred, average='macro')
        }
        self.models['los_class_rf'] = rf_model
        
        print(f"   [OK] Random Forest - Accuracy: {results['random_forest']['accuracy']:.4f}")
        
        # 3. XGBoost
        print("\n[3/3] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS, eval_metric='mlogloss')
        xgb_model.fit(X_train, y_train)
        
        xgb_pred = xgb_model.predict(X_val)
        
        results['xgboost'] = {
            'accuracy': accuracy_score(y_val, xgb_pred),
            'precision_macro': precision_score(y_val, xgb_pred, average='macro'),
            'recall_macro': recall_score(y_val, xgb_pred, average='macro'),
            'f1_macro': f1_score(y_val, xgb_pred, average='macro')
        }
        self.models['los_class_xgb'] = xgb_model
        
        print(f"   [OK] XGBoost - Accuracy: {results['xgboost']['accuracy']:.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        print(f"\n  [BEST MODEL] {best_model_name.upper()} with Accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        # Evaluate on test set
        if best_model_name == 'logistic_regression':
            best_model = lr_model
        elif best_model_name == 'random_forest':
            best_model = rf_model
        else:
            best_model = xgb_model
        
        test_pred = best_model.predict(X_test)
        
        test_results = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision_macro': precision_score(y_test, test_pred, average='macro'),
            'recall_macro': recall_score(y_test, test_pred, average='macro'),
            'f1_macro': f1_score(y_test, test_pred, average='macro')
        }
        
        print(f"\n     [TEST SET RESULTS]")
        print(f"     Accuracy:  {test_results['accuracy']:.4f}")
        print(f"     Precision: {test_results['precision_macro']:.4f}")
        print(f"     Recall:    {test_results['recall_macro']:.4f}")
        print(f"     F1-Score:  {test_results['f1_macro']:.4f}")
        
        self.results['los_classification'] = {
            'validation': results,
            'test': test_results,
            'best_model': best_model_name
        }
        
        return results
    
    # ------------------------------------------------------------------
    # Generic binary XGBoost trainer (reused for new targets)
    # ------------------------------------------------------------------

    # Targets where SMOTE hurts (insufficient temporal signal; oversampling adds noise)
    SMOTE_SKIP_TARGETS = {'icu_readmit_48h', 'icu_readmit_7d'}

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
            smote = SMOTE(random_state=RANDOM_SEED)
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
        self._train_binary_xgb('ICU Readmission 48h', 'icu_readmit_48h')
        self._train_binary_xgb('ICU Readmission 7d', 'icu_readmit_7d')

    def train_discharge_disposition_model(self):
        """3-class XGBoost for discharge disposition."""
        target = 'discharge_disposition'
        if target not in self.data.columns:
            return
        print(f"\n{'=' * 70}")
        print("TRAINING: Discharge Disposition (3-class)")
        print(f"{'=' * 70}")

        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(task=target)
        model = xgb.XGBClassifier(**XGBOOST_PARAMS, eval_metric='mlogloss',
                                   num_class=3, objective='multi:softprob')
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_metrics = {
            'accuracy': float(accuracy_score(y_val, val_pred)),
            'f1_macro': float(f1_score(y_val, val_pred, average='macro', zero_division=0))
        }
        test_pred = model.predict(X_test)
        test_metrics = {
            'accuracy': float(accuracy_score(y_test, test_pred)),
            'f1_macro': float(f1_score(y_test, test_pred, average='macro', zero_division=0))
        }
        print(f"  [VAL]  Accuracy: {val_metrics['accuracy']:.4f}  F1-macro: {val_metrics['f1_macro']:.4f}")
        print(f"  [TEST] Accuracy: {test_metrics['accuracy']:.4f}  F1-macro: {test_metrics['f1_macro']:.4f}")

        self.models['discharge_disposition_xgb'] = model
        self.results['discharge_disposition'] = {
            'validation': val_metrics, 'test': test_metrics, 'best_model': 'xgboost'
        }

    def train_organ_support_models(self):
        self._train_binary_xgb('Ventilation Need', 'need_vent_any')
        self._train_binary_xgb('Vasopressor Need', 'need_vasopressor_any')
        self._train_binary_xgb('RRT Need', 'need_rrt_any')

    def train_disease_onset_models(self):
        self._train_binary_xgb('AKI Onset', 'aki_onset')
        self._train_binary_xgb('ARDS Onset', 'ards_onset')
        self._train_binary_xgb('Liver Injury Onset', 'liver_injury_onset')
        self._train_binary_xgb('Sepsis Onset', 'sepsis_onset')

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
