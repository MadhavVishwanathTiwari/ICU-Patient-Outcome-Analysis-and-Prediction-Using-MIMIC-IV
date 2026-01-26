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
    
    def prepare_data(self, task='mortality'):
        """
        Prepare data for modeling.
        
        Parameters:
        -----------
        task : str
            Prediction task: 'mortality', 'los_regression', or 'los_classification'
        
        Returns:
        --------
        tuple : (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "=" * 70)
        print(f"PREPARING DATA FOR: {task.upper()}")
        print("=" * 70)
        
        # Separate features and targets
        id_cols = ['subject_id', 'hadm_id', 'stay_id']
        target_cols = ['mortality', 'los_days', 'los_category']
        
        X = self.data.drop(columns=id_cols + target_cols)
        
        # Select target based on task
        if task == 'mortality':
            y = self.data['mortality']
        elif task == 'los_regression':
            y = self.data['los_days']
        elif task == 'los_classification':
            y = self.data['los_category']
        else:
            raise ValueError(f"Unknown task: {task}")
        
        print(f"  [OK] Features shape: {X.shape}")
        print(f"  [OK] Target distribution:")
        if task in ['mortality', 'los_classification']:
            print(f"       {y.value_counts().to_dict()}")
        else:
            print(f"       Mean: {y.mean():.2f}, Std: {y.std():.2f}")
        
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
        
        # Handle class imbalance for classification tasks
        if task in ['mortality', 'los_classification'] and HANDLE_IMBALANCE:
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
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(task='los_classification')
        
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
    
    def train_all_models(self):
        """
        Train models for all prediction tasks.
        """
        print("\n" + "=" * 70)
        print("MIMIC-IV MODEL TRAINING PIPELINE")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        self.load_data()
        
        # Train mortality models
        self.train_mortality_models()
        
        # Train LOS classification models
        self.train_los_classification_models()
        
        print("\n" + "=" * 70)
        print("MODEL TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Trained {len(self.models)} models")
        print(f"  Tasks completed: 2 (Mortality, LOS Classification)")
        
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
