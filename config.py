"""
Configuration file for MIMIC-IV ICU Outcome Prediction Project
===============================================================
Centralized configuration for paths, parameters, and settings.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT

# Raw data paths
MIMIC_IV_ROOT = DATA_ROOT / 'mimic-iv-2.2' / 'mimic-iv-2.2'
MIMIC_IV_HOSP = MIMIC_IV_ROOT / 'hosp'
MIMIC_IV_ICU = MIMIC_IV_ROOT / 'icu'
MIMIC_IV_ED = DATA_ROOT / 'mimic-iv-ed-2.2' / 'mimic-iv-ed-2.2' / 'ed'

# Processed data paths
DATA_INTERIM = DATA_ROOT / 'data' / 'interim'
DATA_PROCESSED = DATA_ROOT / 'data' / 'processed'

# Model paths
MODELS_DIR = PROJECT_ROOT / 'models'

# Report paths
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# ============================================================================
# COHORT PARAMETERS
# ============================================================================
MIN_AGE = 18  # Minimum age for inclusion (adults only)
MAX_AGE = 120  # Maximum reasonable age

# ============================================================================
# TARGET VARIABLE DEFINITIONS
# ============================================================================
# Length of Stay categories (in days)
LOS_SHORT_THRESHOLD = 3    # < 3 days = short stay
LOS_MEDIUM_THRESHOLD = 7   # 3-7 days = medium stay
                           # > 7 days = long stay

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Cross-validation
CV_FOLDS = 5

# Class imbalance handling
HANDLE_IMBALANCE = True
IMBALANCE_METHOD = 'smote'  # Options: 'smote', 'adasyn', 'class_weight'

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# Features to extract from diagnosis codes
TOP_N_DIAGNOSES = 50

# Features to extract from procedures
TOP_N_PROCEDURES = 30

# Temporal windows for lab values (in hours before ICU admission)
LAB_WINDOWS = [6, 12, 24, 48]

# ============================================================================
# MODEL HYPERPARAMETERS (Default)
# ============================================================================
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================
DASHBOARD_TITLE = "ICU Patient Outcome Prediction Dashboard"
DASHBOARD_PORT = 8501

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
