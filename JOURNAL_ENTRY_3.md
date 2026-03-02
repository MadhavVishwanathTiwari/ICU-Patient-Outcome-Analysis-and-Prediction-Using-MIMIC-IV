# Journal Entry 3 – Feature Engineering, Extended Targets & Model Training

---

## Index Entry


| Meeting No. | Date       | Purpose                                                | Page No. |
| ----------- | ---------- | ------------------------------------------------------ | -------- |
| 3           | 2026-03-01 | Feature Engineering, Extended Targets & Model Training | 3        |


---

## Weekly Log Entry – Page 3

### Week No.: 3

### Meeting No.: 3

### Date & Time of Meeting: March 1, 2026, [TIME]

### Prepared By: [MEMBER_NAME] (Roll No: [ROLL_NO])

---

### Member(s) Present:

1. [Member 1 Name] (Roll No: [___])
2. [Member 2 Name] (Roll No: [___])
3. [Member 3 Name] (Roll No: [___])
4. [Member 4 Name] (Roll No: [___])

### Member(s) Absent:

- None

---

### Tasks Decided in Previous Meeting (from Meeting 2):

1. Begin feature engineering (vital signs, lab results)
2. Implement data split for training/validation/testing
3. Explore baseline ML models

---

### Work Completed During the Week

Work was divided among four members as follows.

---

#### **Member 1 – Cohort Extensions & Configuration**

**Scope:** Extend cohort targets and centralize itemids and thresholds.

**Tasks Accomplished:**

1. **Cohort readmission labels** (`make_cohort.py`):
  - Implemented `_add_readmission_labels()` to compute ICU readmission within 48h and 7d
  - Used sequential stays per `subject_id`; hours between `outtime` and next `intime`; flags for 0–48h and 0–168h
  - Set readmission = 0 for deceased; filled NaN for last stay with 0
2. **Discharge disposition** (`make_cohort.py`):
  - Implemented `_add_discharge_disposition()` mapping: 0 = Home, 1 = Facility, 2 = Death
  - Logic: Death from `hospital_expire_flag`; Home if "HOME" in `discharge_location`; else Facility
3. **Cohort pipeline updates** (`make_cohort.py`):
  - Extended `engineer_targets()` to call readmission and discharge helpers
  - Extended `select_columns()` to include new target columns
4. **Configuration** (`config.py`):
  - Added vasopressor itemids (inputevents)
  - Added ventilation itemids (chartevents)
  - Added RRT itemids (procedureevents)
  - Added lab/chart itemids for SOFA and clinical targets (creatinine, platelets, bilirubin, PaO₂, FiO₂, MAP, GCS, ALT, AST)
  - Added upper limits of normal (ULN) for liver injury

**Deliverables:** `make_cohort.py` changes, `config.py` additions (~150 lines)

---

#### **Member 2 – Organ Support Feature Extraction**

**Scope:** Extract ventilation, vasopressor, and RRT targets from MIMIC-IV event tables.

**Tasks Accomplished:**

1. `**load_cohort()`** (`build_features.py`):
  - Updated to load all new target columns into the feature matrix
2. **Ventilation targets** (`build_features.py`):
  - Implemented `extract_ventilation_targets()` using chartevents
  - Used chunked loading (2M rows per chunk) to manage memory on 8 GB RAM
  - Filtered by vent itemids and cohort `stay_id`s; stored as `need_vent_any`
3. **Vasopressor targets** (`build_features.py`):
  - Implemented `extract_vasopressor_targets()` from inputevents
  - Flagged stays with vasopressor infusions (amount > 0) as `need_vasopressor_any`
4. **RRT targets** (`build_features.py`):
  - Implemented `extract_rrt_targets()` from procedureevents and procedures_icd
  - Included dialysis ICD codes (e.g. 5A1D) and procedure itemids for RRT as `need_rrt_any`
5. **Pipeline integration** (`build_features.py`):
  - Wired all organ support extractors into `build_feature_matrix()`

**Deliverables:** `build_features.py` organ support extraction (~120 lines)

---

#### **Member 3 – Clinical Onset Targets (AKI, ARDS, Liver, Sepsis)**

**Scope:** Implement clinical onset definitions and SOFA-based sepsis detection.

**Tasks Accomplished:**

1. `**clinical_targets.py` (new file):**
  - `_load_labevents_for_items()` – chunked lab load; filter by itemids and cohort
  - `compute_aki_labels()` – KDIGO creatinine: baseline first 6h; flag if +0.3 in 48h or 1.5× in 7d
  - `compute_ards_labels()` – Berlin: P/F ratio ≤300 with FiO₂ from chartevents (chunked)
  - `compute_liver_injury_labels()` – AST/ALT/bilirubin >3× ULN
  - `compute_sepsis_labels()` – Sepsis-3: suspected infection + SOFA delta ≥2
  - `_suspected_infection()` – microbiology cultures + antibiotic prescriptions in peri-admission window (-24h to +72h)
2. `**sofa_calculator.py` (new file):**
  - `compute_sofa_deltas()` – chunked lab/chart load; SOFA baseline (first 12h) vs peak (6–72h); return stays with delta ≥2
  - `_window_sofa()` – 6-component SOFA (resp, coag, liver, cv, cns, renal) per time window
3. **Integration** (`build_features.py`):
  - Implemented `extract_clinical_targets()` to call the four onset calculators; invoked from `build_feature_matrix()`
4. **Memory handling:**
  - Chunked loading for labevents and chartevents across both modules to support 8 GB RAM

**Deliverables:** `clinical_targets.py`, `sofa_calculator.py`, `extract_clinical_targets()` (~450 lines)

---

#### **Member 4 – Model Training & Dashboard**

**Scope:** Train models for all targets, handle imbalance, and extend the dashboard.

**Tasks Accomplished:**

1. **Model training** (`train_model.py`):
  - Implemented `_train_binary_xgb()` – generic binary XGBoost with `scale_pos_weight`
  - Added trainers for: readmission (48h, 7d), discharge disposition (3-class), organ support (vent, vaso, rrt), disease onset (AKI, ARDS, liver, sepsis)
  - Imbalance: `scale_pos_weight` for all binary targets; SMOTE for very imbalanced (pos <10%) except readmission
  - Fixed `los_classification` → `los_category`; corrected multi-class logging; fixed console encoding
2. **Dashboard** (`app.py`):
  - Extended `load_models()` to load 12 prediction models (mortality_xgb, los_class_xgb, 10 extended XGB) plus scaler
  - New panels: readmission gauges (48h, 7d), discharge disposition bar chart, organ support bars, disease onset cards
  - Extended Actual Outcomes for all new targets
  - Fixed `st.progress()` float32 error with `float()`
  - Added second metrics row; extended model performance page; updated About page
3. **Pipeline completion:**
  - All 12 tasks trained; 16 models saved; pipeline validated end-to-end

**Deliverables:** `train_model.py` extensions (~~200 lines), `app.py` extensions (~~150 lines)

---

### Preprocessing Pipeline (Summary)

A consolidated overview of all data preprocessing steps, split by member responsibility:

---

#### **Member 1 – Cohort & Core Features** (foundation)

| Step | Description |
|------|-------------|
| **Cohort load** | Load `patients.csv.gz`, `admissions.csv.gz`, `icustays.csv.gz`; parse dates (dod, admittime, dischtime, intime, outtime) |
| **Age** | Compute age at admission: `anchor_age + (admission_year - anchor_year)` |
| **Merge** | Join patients → admissions → icustays on subject_id, hadm_id |
| **Deduplicate** | Remove duplicate rows by `stay_id` (or subject_id+hadm_id+intime); keep first |
| **Filter** | Adults only (age ≥ 18); drop rows with missing `los` or `hospital_expire_flag` |
| **Targets** | Engineer mortality, los_days, los_category, icu_readmit_48h, icu_readmit_7d, discharge_disposition; fill NaN readmission with 0 for last stay |
| **Configuration** | Centralize itemids (vent, vaso, RRT, labs) and ULN thresholds in `config.py` |
| **Diagnoses** | Filter to cohort; top-N ICD codes as binary indicators |
| **Procedures** | Filter to cohort; top-N procedure codes as binary indicators |
| **Labs** | First 24h window; aggregate (min, max, mean, std, count) per stay for top lab items; left merge (NaN if no value) |
| **Demographics** | Age (numeric), gender (binary M), ethnicity (grouped), insurance/admission/ICU (one-hot); fillna → Unknown/UNKNOWN |
| **Temporal** | Extract hour, day_of_week, month, season from admittime |

---

#### **Member 2 – Organ Support Targets**

| Step | Description |
|------|-------------|
| **Ventilation** | Extract from chartevents; chunked load; filter by vent itemids and cohort stay_ids |
| **Vasopressor** | Extract from inputevents; flag stays with amount &gt; 0 |
| **RRT** | Extract from procedureevents + procedures_icd (dialysis ICD 5A1D); merge into feature matrix |

---

#### **Member 3 – Clinical Onset Targets**

| Step | Description |
|------|-------------|
| **AKI** | KDIGO creatinine: baseline first 6h; +0.3 in 48h or 1.5× in 7d |
| **ARDS** | Berlin: P/F ratio ≤300; FiO₂ from chartevents (chunked) |
| **Liver injury** | AST/ALT/bilirubin &gt; 3× ULN |
| **Sepsis** | Sepsis-3: suspected infection + SOFA delta ≥2; microbiology + antibiotics, sofa_calculator.py |

---

#### **Member 4 – Model Preparation**

| Step | Description |
|------|-------------|
| **Load** | Read engineered features CSV |
| **Column names** | Sanitize for XGBoost: remove `[`, `]`, `<`, `>` |
| **Split** | 60% train / 20% validation / 20% test; stratified for classification |
| **Scaling** | StandardScaler fit on train, transform train/val/test |
| **Imbalance** | `scale_pos_weight` for all binary XGB; SMOTE when pos &lt; 10% (except readmission targets) |

---

### Joint Work

- **Integration:** Coordinated cohort → features → training → dashboard flow
- **Testing:** Full run on 8 GB machine; chunked loading validated; all models saved
- **Documentation:** `DEVELOPMENT_LOG_AFTER_JOURNAL_2.md` created

---

### Issues or Challenges

1. **Memory:** Loading chartevents (~313M rows) caused MemoryError on 8 GB RAM
  - **Resolution:** Chunked loading (1.5M–2M rows per chunk) with per-chunk filtering
2. **Readmission F1 near zero:** SMOTE worsened ROC and F1
  - **Resolution:** Skipped SMOTE for readmission; used `scale_pos_weight` only
3. **Streamlit:** `st.progress()` failed with NumPy float32
  - **Resolution:** Wrapped values with `float()`
4. **Console encoding:** Unicode (×, —) corrupted on Windows
  - **Resolution:** Replaced with ASCII equivalents (3x, hyphen)

---

### Key Results Achieved


| Metric                      | Value                             |
| --------------------------- | --------------------------------- |
| Cohort targets              | 6 (was 3)                         |
| Feature matrix columns      | 128 (112 features + 16 target/ID) |
| Trained models              | 16                                |
| Pipeline runtime (features) | ~30–60 min on 8 GB RAM            |


**Test performance (all targets):**


| Target               | ROC-AUC / Acc | F1    |
| -------------------- | ------------- | ----- |
| Mortality            | 0.90          | 0.51  |
| LOS category         | 0.75 (Acc)    | 0.57  |
| ICU Readmit 48h      | 0.60          | 0.13  |
| ICU Readmit 7d       | 0.68          | 0.24  |
| Discharge disposition| 0.69 (Acc)    | 0.64  |
| Ventilation          | 0.92          | 0.84  |
| Vasopressor          | 0.86          | 0.69  |
| RRT                  | 0.93          | 0.41  |
| AKI onset            | 0.78          | 0.48  |
| ARDS onset           | 0.89          | 0.78  |
| Liver injury onset   | 0.81          | 0.51  |
| Sepsis onset         | 0.76          | 0.48  |


---

### Supervisor’s Remarks and Signature

**Comments:**  
[Space for supervisor feedback]

**Signature:** _________________________  
**Name:** [SUPERVISOR_NAME]  
**Date:** _____________

---

### Tasks for Next Meeting

1. Consider ensemble or threshold tuning for low-performers (e.g. readmission 48h)
2. Add feature importance / SHAP for model interpretability
3. Optional: external validation or cross-validation strateg
4. Prepare final report and presentation

---

*End of Journal Entry 3*