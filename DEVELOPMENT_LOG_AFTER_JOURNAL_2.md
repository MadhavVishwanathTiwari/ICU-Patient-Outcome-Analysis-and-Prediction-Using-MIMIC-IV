# Development Log: After Journal Entry 2

**Scope:** All development following the preprocessing phase (Journal Entry 2)  
**Reference:** Journal Entry 2 covered data loading, merging, age calculation, filtering, and initial target engineering (mortality, los_days, los_category).

---

## Summary of Additions

| Category | Before | After |
|----------|--------|-------|
| Cohort targets | 3 (mortality, los_days, los_category) | 6 (+ readmit 48h/7d, discharge_disposition) |
| Feature targets | 0 | 7 (vent, vaso, rrt, aki, ards, liver, sepsis) |
| Trained models | 6 | 16 |
| Dashboard prediction panels | 2 (mortality, LOS) | 8 (all targets) |

---

## 1. Cohort Extensions (`make_cohort.py`)

| Change | Description |
|--------|-------------|
| `_add_readmission_labels()` | Compute `icu_readmit_48h` and `icu_readmit_7d` from sequential ICU stays per patient. Hours between `outtime` and next `intime`; readmit if 0–48h or 0–168h. Set 0 for deceased patients; fill NaN (last stay) with 0. |
| `_add_discharge_disposition()` | Map to 0=Home, 1=Facility, 2=Death. Death from `hospital_expire_flag`; Home from discharge_location containing "HOME"; else Facility. |
| `engineer_targets()` | Extended to call readmission and discharge disposition helpers. |
| `select_columns()` | Added new target columns to output. |

---

## 2. Configuration (`config.py`)

| Addition | Purpose |
|----------|---------|
| `VASOPRESSOR_ITEMIDS` | [221906, 221289, 221662, 222315, 221749] |
| `VENT_ITEMIDS` | [223848, 223849, 720] |
| `RRT_ITEMIDS` | [225802, 225803, 225805] |
| `CREATININE_ITEMID`, `PLATELETS_ITEMID`, `BILIRUBIN_ITEMID`, `PAO2_ITEMID`, `ALT_ITEMID`, `AST_ITEMID` | Lab itemids for SOFA & clinical targets |
| `MAP_ITEMID`, `GCS_ITEMID`, `FIO2_ITEMID` | Chart itemids |
| `ULN_ALT`, `ULN_AST`, `ULN_BILIRUBIN` | Upper limits of normal for liver injury |

---

## 3. Feature Engineering (`build_features.py`)

| Addition | Description |
|----------|-------------|
| `load_cohort()` | Now includes all new target columns in feature matrix. |
| `extract_ventilation_targets()` | Chunked read of chartevents; flag stays with vent itemids. |
| `extract_vasopressor_targets()` | Flag stays with vasopressor infusions (inputevents). |
| `extract_rrt_targets()` | Flag from procedureevents + procedures_icd dialysis codes. |
| `extract_clinical_targets()` | Call AKI, ARDS, liver, sepsis modules. |
| `build_feature_matrix()` | Runs all new extractors. |

---

## 4. New Modules

### `src/features/clinical_targets.py`
| Function | Definition |
|----------|-------------|
| `_load_labevents_for_items()` | Chunked load of labevents; filter by itemids and cohort. |
| `compute_aki_labels()` | KDIGO creatinine: baseline first 6h; flag if +0.3 in 48h or 1.5× in 7d. |
| `compute_ards_labels()` | P/F ratio ≤300 with FiO2 from chartevents (chunked). |
| `compute_liver_injury_labels()` | AST/ALT/bili >3× ULN. |
| `compute_sepsis_labels()` | Sepsis-3: suspected infection (culture + abx) + SOFA delta ≥2. |
| `_suspected_infection()` | Microbiology + prescriptions overlap. |

### `src/features/sofa_calculator.py`
| Function | Purpose |
|----------|---------|
| `compute_sofa_deltas()` | Chunked lab/chart load; compute SOFA; return stays with delta ≥2. |
| `_window_sofa()` | 6-component SOFA (resp, coag, liver, cv, cns, renal) for a time window. |

---

## 5. Memory Optimizations (Low-RAM / 8GB)

| Location | Change |
|----------|--------|
| `build_features.extract_ventilation_targets()` | Chunked chartevents (2M rows); filter by vent itemids and cohort stays. |
| `clinical_targets._load_labevents_for_items()` | Chunked labevents (1.5M rows); merge with cohort per chunk. |
| `clinical_targets.compute_ards_labels()` | Chunked chartevents for FiO2. |
| `sofa_calculator.compute_sofa_deltas()` | Chunked labevents and chartevents. |

---

## 6. Model Training (`train_model.py`)

### New Trainers
| Function | Targets |
|----------|---------|
| `_train_binary_xgb()` | Generic binary XGBoost with imbalance handling. |
| `train_readmission_models()` | icu_readmit_48h, icu_readmit_7d |
| `train_discharge_disposition_model()` | 3-class XGBoost (Home/Facility/Death) |
| `train_organ_support_models()` | need_vent_any, need_vasopressor_any, need_rrt_any |
| `train_disease_onset_models()` | aki_onset, ards_onset, liver_injury_onset, sepsis_onset |

### Imbalance Handling
| Logic | Applied to |
|-------|------------|
| `scale_pos_weight = n_neg/n_pos` | All binary XGBoost models |
| SMOTE | Mortality; RRT (pos rate <10%). |
| SMOTE skipped | Readmission (icu_readmit_48h, icu_readmit_7d) – SMOTE was degrading F1. |

### Logging / Fixes
| Change | Reason |
|--------|--------|
| `los_classification` → `los_category` | Column name fix. |
| No "Positive rate" for multi-class | Correct for los_category, discharge_disposition. |
| Em-dash → hyphen | Avoid Windows console encoding issues. |
| `prepare_data()` | Supports any target column from ALL_TARGET_COLS. |

---

## 7. Dashboard (`app.py`)

| Addition | Details |
|----------|---------|
| `load_models()` | Loads all 16 model pkls and optional extended models. |
| Prediction panels | Mortality gauge; LOS bar; readmission gauges (48h, 7d); discharge disposition bar; organ support (vent, vaso, rrt); disease onset (AKI, ARDS, liver, sepsis). |
| Actual Outcomes | Extended to show all targets: mortality, LOS, readmit 48h/7d, discharge, vent, vaso, rrt, AKI, ARDS, sepsis. |
| `st.progress()` | Wrap with `float()` to avoid Streamlit float32 error. |
| Dashboard metrics | Second row for readmit 7d, vent, AKI, sepsis rates. |
| Model performance | Section for extended targets. |
| About page | Updated task list and description. |

---

## 8. Bug Fixes & Minor Changes

| Issue | Fix |
|-------|-----|
| `ValueError: Unknown task los_classification` | Use `los_category` in prepare_data and train_los_classification_models. |
| `StreamlitAPIException: Progress Value has invalid type: float32` | Use `float()` in `st.progress()`. |
| `>3� ULN` console corruption | Use `>3x ULN` (ASCII). |
| Corrupted em-dash in logs | Use ASCII hyphen in SMOTE-related prints. |

---

## 9. Challenges and Resolutions

| Challenge | Resolution |
|-----------|------------|
| MemoryError loading chartevents (~313M rows) | Chunked loading with per-chunk filtering. |
| ICU readmission F1 near zero | `scale_pos_weight`; SMOTE disabled for readmission. |
| SMOTE hurting readmission ROC/F1 | Added `SMOTE_SKIP_TARGETS` for readmission. |
| Misleading "Positive rate" for multi-class | Show value_counts only; skip for los_category, discharge_disposition. |

---

## 10. Final Metrics (Latest Training Run)

| Target | Test ROC-AUC / Acc | Test F1 |
|--------|--------------------|---------|
| Mortality | 0.90 | 0.51 |
| LOS (3-class) | 0.75 | 0.57 |
| ICU Readmit 48h | 0.60 | 0.13 |
| ICU Readmit 7d | 0.68 | 0.24 |
| Discharge Disposition | 0.69 | 0.64 |
| Ventilation | 0.92 | 0.84 |
| Vasopressor | 0.86 | 0.69 |
| RRT | 0.93 | 0.41 |
| AKI | 0.78 | 0.48 |
| ARDS | 0.89 | 0.78 |
| Liver Injury | 0.81 | 0.51 |
| Sepsis | 0.76 | 0.48 |

---

**Document Created:** 2026-03-01  
**Last Updated:** 2026-03-01
