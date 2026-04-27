# ICU Patient Outcome Analysis and Prediction Using MIMIC-IV

> **A tournament-based machine learning framework for simultaneous prediction of 12 clinical outcomes in the Intensive Care Unit.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![MIMIC-IV](https://img.shields.io/badge/Data-MIMIC--IV%20v2.2-green)](https://physionet.org/content/mimiciv/2.2/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This project builds an end-to-end clinical machine learning pipeline on **MIMIC-IV v2.2** to predict 12 clinically meaningful ICU outcomes from the first 24 hours of admission data. A 240-model tournament (5 algorithms × 4 feature matrices × 12 targets) identifies the best model-feature combination per target, followed by Bayesian hyperparameter tuning via Optuna and SHAP-based interpretability analysis.

The full pipeline culminates in an interactive **Streamlit dashboard** for real-time patient-level risk scoring.

---

## Predicted Outcomes (12 Targets)

| Category | Target | Type |
|---|---|---|
| Survival | In-hospital Mortality | Binary |
| Acute Complications | AKI Onset (KDIGO) | Binary |
| | ARDS Onset (Berlin) | Binary |
| | Acute Liver Injury (3× ULN) | Binary |
| | Sepsis Onset (Sepsis-3) | Binary |
| Organ Support | Need for Mechanical Ventilation | Binary |
| | Need for Vasopressors | Binary |
| | Need for Renal Replacement Therapy | Binary |
| Readmission | ICU Readmission within 48h | Binary |
| | ICU Readmission within 7d | Binary |
| Stay & Disposition | Length of Stay Category | Multiclass |
| | Discharge Disposition | Multiclass |

---

## Key Results (Post-Tuning AUROC)

| Target | Best Model | Matrix | Tuned AUROC |
|---|---|---|---|
| Mortality | CatBoost | LASSO | **0.8966** |
| AKI Onset | Custom MLP | LASSO | **0.8191** |
| Sepsis Onset | Random Forest | MI | **0.7841** |
| ARDS Onset | CatBoost | MI | **0.9373** |
| Liver Injury | CatBoost | IG | **0.9290** |
| ICU Readmit (48h) | Logistic Regression | ANOVA | 0.5919 |
| ICU Readmit (7d) | Logistic Regression | ANOVA | 0.6031 |
| LOS Category | Custom MLP | LASSO | **0.7681** |
| Discharge Disposition | XGBoost | LASSO | **0.8239** |
| Need for Ventilation | CatBoost | MI | **0.8960** |
| Need for Vasopressors | CatBoost | MI | **0.8855** |
| Need for RRT | Custom MLP | IG | **0.9444** |

---

## Project Structure

```
MinorProject/
├── src/
│   ├── app.py                          # Streamlit dashboard
│   ├── data/
│   │   └── make_cohort.py              # ICU cohort extraction
│   ├── features/
│   │   ├── build_features_v2.py        # 24-hour feature engineering
│   │   ├── clinical_targets.py         # AKI / ARDS / liver / sepsis labels
│   │   ├── select_features_v2.py       # IG / ANOVA / MI / LASSO selection
│   │   ├── leakage_rules.py            # Outcome leakage prevention
│   │   ├── sofa_calculator.py          # SOFA score for Sepsis-3
│   │   └── shap_analysis.py            # SHAP interpretability
│   └── models/
│       ├── custom_mlp.py               # Keras MLP architecture
│       ├── run_full_tournament.py      # 240-model tournament
│       ├── tune_winners.py             # Optuna hyperparameter tuning
│       ├── save_tuned_models.py        # Serialize models to disk
│       └── train_model_v2.py           # V2 training entrypoint
├── data/
│   ├── interim/
│   └── processed/
│       └── tournament/                 # X_ig/anova/mi/lasso_union.csv
├── models/
│   └── tuned/                          # .pkl / .keras model artifacts
├── results/
│   ├── tournament_scores.json          # All 240 baseline AUROC scores
│   ├── best_hyperparams.json           # Winning hyperparameters per target
│   ├── tuning_summary.json             # Baseline vs tuned AUROC comparison
│   ├── roc_curves/                     # Pre-tuning ROC PNGs
│   ├── roc_curves_tuned/               # Post-tuning ROC PNGs
│   └── shap/                           # SHAP beeswarm plots
├── config.py                           # All itemids, paths, constants
├── requirements.txt
└── RUN_DASHBOARD.bat                   # Windows launcher for Streamlit
```

---

## Pipeline Architecture

```
MIMIC-IV v2.2
     │
     ▼
make_cohort.py          ──► Adult ICU stays, first admission
     │
     ▼
build_features_v2.py    ──► 24h vitals, labs, comorbidities, treatment flags
     │
     ▼
clinical_targets.py     ──► 12 binary/multiclass outcome labels
 + sofa_calculator.py       (KDIGO / Berlin / Sepsis-3 / 3×ULN)
     │
     ▼
select_features_v2.py   ──► Spearman + VIF purge → IG / ANOVA / MI / LASSO matrices
     │
     ▼
run_full_tournament.py  ──► 240-model evaluation (5 models × 4 matrices × 12 targets)
     │
     ▼
tune_winners.py         ──► Optuna Bayesian optimisation per target winner
     │
     ▼
save_tuned_models.py    ──► Serialise to models/tuned/ (.pkl / .keras)
     │
     ├── shap_analysis.py    ──► SHAP beeswarm plots per target
     │
     └── app.py              ──► Streamlit real-time dashboard
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MadhavVishwanathTiwari/ICU-Patient-Outcome-Analysis-and-Prediction-Using-MIMIC-IV.git
cd ICU-Patient-Outcome-Analysis-and-Prediction-Using-MIMIC-IV
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.9+ is recommended. TensorFlow 2.x is required for the Custom MLP.

### 3. Obtain MIMIC-IV data

This project requires **MIMIC-IV v2.2** and **MIMIC-IV-ED v2.2**, available from PhysioNet under a data use agreement:

- https://physionet.org/content/mimiciv/2.2/
- https://physionet.org/content/mimic-iv-ed/2.2/

Place the extracted data as follows:

```
MinorProject/
├── mimic-iv-2.2/mimic-iv-2.2/
│   ├── hosp/       # labevents, prescriptions, diagnoses, etc.
│   └── icu/        # chartevents, inputevents, outputevents, etc.
└── mimic-iv-ed-2.2/mimic-iv-ed-2.2/ed/
```

The paths are configured in `config.py` (`MIMIC_IV_HOSP`, `MIMIC_IV_ICU`, `MIMIC_IV_ED`).

---

## Running the Pipeline

Run each stage in order:

```bash
# 1. Build cohort
python src/data/make_cohort.py

# 2. Engineer features (first 24h)
python src/features/build_features_v2.py

# 3. Compute clinical targets
#    (called internally by build_features_v2.py via clinical_targets.py)

# 4. Feature selection — generates 4 matrices in data/processed/tournament/
python src/features/select_features_v2.py

# 5. Run the full 240-model tournament
python src/models/run_full_tournament.py

# 6. Hyperparameter tuning (Optuna, runs per target)
python src/models/tune_winners.py

# 7. Save tuned models to disk
python src/models/save_tuned_models.py

# 8. SHAP analysis
python src/features/shap_analysis.py

# 9. Launch dashboard
streamlit run src/app.py
# OR on Windows:
RUN_DASHBOARD.bat
```

> **Memory note:** MIMIC-IV labevents and chartevents are large (~20GB uncompressed). All reads are chunked (1.5M–2M rows per chunk). Minimum 16GB RAM recommended; 32GB preferred.

---

## Feature Engineering

Features are extracted from the **first 24 hours** of ICU admission only, ensuring predictions are available at the point of admission.

| Feature Group | Details |
|---|---|
| **Vital signs** | Heart rate, respiratory rate, MAP, SpO2, temperature (min/max/mean/SD) |
| **Laboratory** | 17 categories including creatinine, lactate, bilirubin, WBC, platelets, INR, ALT, AST, albumin, BUN, glucose, electrolytes |
| **Urine output** | Cumulative 24h total (ml) |
| **Treatment flags** | Ventilation, vasopressor, RRT use in first 24h (leakage-controlled) |
| **Comorbidities** | 7 ICD-coded flags: CKD, COPD, diabetes, CHF, chronic liver disease, malignancy, hypertension |

---

## Clinical Target Definitions

| Target | Standard | Criterion |
|---|---|---|
| AKI | KDIGO | Creatinine ≥+0.3 mg/dL in 48h or ≥1.5× baseline in 7d |
| ARDS | Berlin | PaO2/FiO2 ≤ 300 at any point during stay |
| Liver Injury | 3× ULN | ALT or AST or bilirubin > 3× upper limit of normal |
| Sepsis | Sepsis-3 | Suspected infection + SOFA delta ≥ 2 (baseline: hrs −6 to +6; peak: hrs +6 to +72) |

---

## Leakage Prevention

The `leakage_rules.py` module enforces an explicit policy:

> When predicting **need_vent_any**, **need_vasopressor_any**, or **need_rrt_any**, the first-24h treatment flags (`ventilation_24h_flag`, `vasopressor_24h_flag`, `rrt_24h_flag`) are automatically dropped from the feature matrix — these are near-direct proxies for the outcomes being predicted.

---

## Models

Five classifiers were evaluated in the tournament:

| Model | Notes |
|---|---|
| **CatBoost** | Gradient boosting; handles imbalance via `auto_class_weights='Balanced'` |
| **XGBoost** | Gradient boosting; logloss / mlogloss evaluation |
| **Random Forest** | 100 estimators; `class_weight='balanced'` |
| **Logistic Regression** | L1/L2/ElasticNet via SAGA solver; balanced weights |
| **Custom MLP** | Keras Sequential (128→64→32); Dropout; EarlyStopping; binary/multiclass output |

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
catboost
tensorflow>=2.10
optuna
shap
streamlit
statsmodels
joblib
matplotlib
```

See `requirements.txt` for pinned versions.

---

## Data Availability

MIMIC-IV is **not included** in this repository. It must be obtained independently:

1. Complete the CITI training program
2. Sign the PhysioNet data use agreement
3. Download from: https://physionet.org/content/mimiciv/2.2/

---

## Citation

If you use this code or methodology in your work, please cite:

```
@misc{tiwari2025icu,
  title   = {Multi-Target Clinical Outcome Prediction in the ICU Using MIMIC-IV:
             A Tournament-Based Machine Learning Framework},
  author  = {Madhav Vishwanath Tiwari et al.},
  year    = {2025},
  url     = {https://github.com/MadhavVishwanathTiwari/ICU-Patient-Outcome-Analysis-and-Prediction-Using-MIMIC-IV}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) — Johnson et al., Scientific Data 2023
- [Optuna](https://optuna.org/) — Akiba et al., KDD 2019
- [SHAP](https://github.com/slundberg/shap) — Lundberg & Lee, NeurIPS 2017
- [CatBoost](https://catboost.ai/) — Dorogush et al., 2018