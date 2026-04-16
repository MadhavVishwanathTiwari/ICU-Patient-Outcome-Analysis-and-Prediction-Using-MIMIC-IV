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
# FEATURE ENGINEERING V2 (Clinically grounded pool)
# ============================================================================
FEATURE_WINDOW_HOURS_V2 = 24

# ICU charted vital itemids (MIMIC-IV meta-vision + carevue common IDs)
VITAL_ITEMIDS_V2 = {
    'heart_rate': [220045, 211],
    'resp_rate': [220210, 618, 224690, 615],
    'map': [220052, 456, 220181, 52, 6702, 443, 225312],
    'spo2': [220277, 646],
    # Includes Celsius and Fahrenheit itemids; code normalizes Fahrenheit to Celsius.
    'temperature_c': [223762, 676, 223761, 678, 679],
}

# Urine output (ICU outputevents) itemids
URINE_OUTPUT_ITEMIDS_V2 = [226559, 226560, 226561, 226584, 226563, 226564]

# Key labs used for first-24h physiological feature extraction
LAB_ITEMIDS_V2 = {
    'creatinine': [50912],
    'bun': [51006],
    'lactate': [50813],
    'bicarbonate': [50882, 50803],
    'anion_gap': [50868],
    'wbc': [51300, 51301, 51755],
    'hemoglobin': [51222, 50811],
    'platelets': [51265],
    'inr': [51237, 51274],
    'sodium': [50983, 50824],
    'potassium': [50971, 50822],
    'glucose': [50931, 50809],
    'albumin': [50862, 53085],
    'bilirubin': [50885],
    'alt': [50861],
    'ast': [50878],
}

# Curated comorbidity code prefixes (ICD-9/10 mixed, hadm-level context only)
COMORBIDITY_PREFIXES_V2 = {
    'ckd': ['585', 'N18'],
    'copd': ['491', '492', '496', 'J44'],
    'diabetes': ['250', 'E10', 'E11', 'E13'],
    'chf': ['428', 'I50'],
    'chronic_liver_disease': ['571', 'K70', 'K74', 'K76'],
    'malignancy': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
                   '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
                   '160', '161', '162', '163', '164', '165', '170', '171', '172', '174',
                   '179', '180', '181', '182', '183', '184', '185', '186', '187', '188',
                   '189', '190', '191', '192', '193', '194', '195', '196', '197', '198',
                   '199', 'C'],
    'hypertension': ['401', '402', '403', '404', '405', 'I10', 'I11', 'I12', 'I13'],
}

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
    'n_jobs': 2
}

# ============================================================================
# CLINICAL ITEMID CONSTANTS (MIMIC-IV)
# ============================================================================
# Vasopressors (inputevents)
VASOPRESSOR_ITEMIDS = [221906, 221289, 221662, 222315, 221749]

# Ventilation (chartevents)
VENT_ITEMIDS = [223848, 223849, 720]

# Renal Replacement Therapy (procedureevents)
RRT_ITEMIDS = [225802, 225803, 225805]

# SOFA component labs (labevents)
CREATININE_ITEMID = 50912
PLATELETS_ITEMID  = 51265
BILIRUBIN_ITEMID  = 50885
PAO2_ITEMID       = 50821
ALT_ITEMID        = 50861
AST_ITEMID        = 50878

# SOFA component vitals (chartevents)
MAP_ITEMID  = 220052
GCS_ITEMID  = 220739
FIO2_ITEMID = 223835

# Upper limits of normal for liver injury detection
ULN_ALT = 40       # U/L
ULN_AST = 40       # U/L
ULN_BILIRUBIN = 1.2  # mg/dL

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
