# Journal 3 – Step-by-Step: What We Did

This document explains, in order, the steps the team took during Journal Entry 3 for **Feature Engineering, Extended Targets, and Model Training**. It follows the pipeline from raw MIMIC-IV data to trained models and the dashboard.

---

## Overview of the Pipeline

We built a single end-to-end flow:

1. **Build the cohort** – Define who is in the study and compute their outcome labels.
2. **Engineer features** – Turn raw events (labs, vitals, diagnoses, etc.) into a feature matrix.
3. **Add extended targets** – Compute organ-support and clinical-onset labels (ventilation, AKI, sepsis, etc.).
4. **Train models** – Fit XGBoost (and baseline) models for each target.
5. **Run the dashboard** – Load models and show predictions and performance.

Below, each phase is broken down into concrete steps.

---

## Phase 1: Cohort Building (Foundation)

The cohort is the list of ICU stays we study. Each row is one ICU stay (`stay_id`). We only keep adults and ensure we have the basic outcomes we need.

### Step 1.1 – Load raw tables

We read three MIMIC-IV files:

- **patients** – Demographics and anchor age/year.
- **admissions** – Hospital admissions (admittime, dischtime, discharge_location, etc.).
- **icustays** – ICU stays (intime, outtime, los, first_careunit, etc.).

Dates are parsed so we can compute time windows and lengths of stay.

### Step 1.2 – Compute age at admission

MIMIC-IV uses `anchor_age` and `anchor_year` for privacy. We compute:

**Age at admission = anchor_age + (admission_year − anchor_year)**

so every stay has a numeric age for filtering and as a feature.

### Step 1.3 – Merge into one cohort table

We join **patients → admissions → icustays** on `subject_id` and `hadm_id`. After this we have one row per ICU stay with patient and admission info attached.

### Step 1.4 – Deduplicate

Sometimes joins create duplicate rows (e.g. multiple rows per `stay_id`). We keep **one row per ICU stay** by dropping duplicates on `stay_id` (or on `subject_id` + `hadm_id` + `intime` if `stay_id` is missing).

### Step 1.5 – Apply filters

We keep only:

- **Adults:** age ≥ 18.
- **Complete data:** no missing `los` or `hospital_expire_flag`.

Rows that fail these are dropped.

### Step 1.6 – Engineer core targets

For each stay we define the outcomes we want to predict:

| Target | How it’s defined |
|--------|-------------------|
| **mortality** | In-hospital death from `hospital_expire_flag` (0/1). |
| **los_days** | Length of stay in days from `los`. |
| **los_category** | Short (&lt;3 d), Medium (3–7 d), Long (&gt;7 d) from `los_days`. |
| **icu_readmit_48h** | 1 if the same patient has another ICU stay starting within 48 hours of this stay’s `outtime`. |
| **icu_readmit_7d** | Same idea, but within 7 days (168 hours). |
| **discharge_disposition** | 0 = Home, 1 = Facility, 2 = Death (from `discharge_location` and `hospital_expire_flag`). |

For readmission we look at **sequential stays per patient**: time from this stay’s `outtime` to the next stay’s `intime`. If that gap is in 0–48 h we set `icu_readmit_48h = 1`; if in 0–168 h we set `icu_readmit_7d = 1`. Deceased patients are set to 0; last stay per patient (no “next” stay) is filled with 0.

### Step 1.7 – Centralize configuration

We put all MIMIC item IDs and thresholds in **config.py** so the rest of the code doesn’t hardcode them. This includes:

- Ventilation item IDs (chartevents).
- Vasopressor item IDs (inputevents).
- RRT procedure item IDs and dialysis ICD codes (e.g. 5A1D).
- Lab and chart item IDs for SOFA and clinical targets (creatinine, platelets, bilirubin, PaO₂, FiO₂, MAP, GCS, ALT, AST).
- Upper limits of normal (ULN) for liver injury (ALT, AST, bilirubin).

The cohort is saved as **cohort_labeled.csv** with identifiers, demographics/admission columns, and these six targets.

---

## Phase 2: Feature Matrix – Core Features

We start from the cohort and add **features** (inputs for the models). The result is one row per `stay_id` with many columns: IDs, targets, and feature columns.

### Step 2.1 – Load cohort and initialize feature table

We read **cohort_labeled.csv** and build an initial feature table that has `subject_id`, `hadm_id`, `stay_id`, and all available target columns. Every other feature we add is merged onto this table by `stay_id` or `hadm_id`.

### Step 2.2 – Diagnosis features (top-N ICD codes)

We load **diagnoses_icd**, restrict to admissions in our cohort, and find the **top-N most common ICD codes**. For each of those codes we add a **binary column**: 1 if that admission had that diagnosis, 0 otherwise. So we get “top diagnoses” as binary indicators per stay.

### Step 2.3 – Procedure features (top-N procedure codes)

Same idea as diagnoses: we use **procedures_icd**, restrict to cohort, take the top-N procedure codes, and add **binary columns** (1/0) per code per stay.

### Step 2.4 – Lab features (first 24 hours)

We load **labevents**, keep only events in the **first 24 hours** of each ICU stay, and restrict to a set of important lab item IDs. For each (stay, lab item) we compute **min, max, mean, std, count** of the values. These become numeric columns (e.g. `creatinine_max`, `creatinine_mean`). Stays with no value for a lab get NaN; we merge with **left** join so we keep all stays.

### Step 2.5 – Demographics and admission

We pull from the cohort:

- **Age** – numeric.
- **Gender** – binary (e.g. male = 1).
- **Ethnicity** – grouped into a few categories and encoded (e.g. one-hot or binary flags).
- **Insurance, admission type, first care unit** – one-hot encoded.

Missing categorical values are filled with “Unknown” or “UNKNOWN” so the model always has a valid value.

### Step 2.6 – Temporal features

From **admittime** we derive:

- Hour of day, day of week, month.
- Flags such as night admission, weekend admission.
- Season (winter/spring/summer/fall).

These capture time-of-admission patterns that may relate to outcomes.

After Phase 2 we have a **core feature matrix**: one row per stay, with IDs, the six cohort targets, diagnosis/procedure/lab/demographic/temporal features, and (later) the extended targets.

---

## Phase 3: Extended Targets – Organ Support

We add **binary targets** that indicate whether the patient needed certain organ support during the stay. These come from event tables, not from the cohort CSV.

### Step 3.1 – Ventilation

We use **chartevents** and the ventilation item IDs from config. The file is very large (~313M rows), so we read it in **chunks** (e.g. 2M rows per chunk). For each chunk we keep only rows with those vent item IDs and with `stay_id` in our cohort. Any stay that appears in such rows is considered to have received ventilation. We add a column **need_vent_any** (0/1) to the feature matrix.

### Step 3.2 – Vasopressors

We use **inputevents** and the vasopressor item IDs. We flag stays where there is at least one row with one of those item IDs and **amount &gt; 0**. We add **need_vasopressor_any** (0/1).

### Step 3.3 – Renal replacement therapy (RRT)

We combine two sources:

- **procedureevents** – RRT procedure item IDs from config.
- **procedures_icd** – dialysis ICD codes (e.g. 5A1D) for cohort admissions.

A stay is positive if it appears in either source. We add **need_rrt_any** (0/1).

All three are merged into the same feature matrix so every stay has vent/vaso/RRT labels.

---

## Phase 4: Extended Targets – Clinical Onset

We add four more **binary targets** that indicate whether the patient developed a specific condition during the stay (AKI, ARDS, liver injury, sepsis). These use standard clinical definitions and require lab/chart data in time windows.

### Step 4.1 – AKI (acute kidney injury)

We use **KDIGO creatinine criteria**:

- **Baseline creatinine:** first value in the **first 6 hours** of the stay.
- **Onset:** we flag AKI if, after the first 6 hours, creatinine either **rises by ≥0.3 mg/dL within 48 hours** or **reaches ≥1.5× baseline within 7 days**.

We load creatinine from **labevents** (chunked if needed), align by `stay_id` and time from `intime`, compute baseline and subsequent values per stay, and set **aki_onset** (0/1).

### Step 4.2 – ARDS (acute respiratory distress syndrome)

We use the **Berlin definition**: P/F ratio (PaO₂ / FiO₂) ≤ 300.

- **PaO₂** comes from **labevents** (one lab item ID).
- **FiO₂** comes from **chartevents** (chunked); we use values in a time window and optionally merge to the nearest PaO₂ time (e.g. within ±2 h).

For each stay we compute P/F at relevant times; if any value is ≤ 300 we set **ards_onset** = 1.

### Step 4.3 – Liver injury

We load **AST, ALT, and bilirubin** from labevents (chunked). We use the **upper limits of normal (ULN)** from config. If **any** of the three exceeds **3× its ULN** during the stay, we set **liver_injury_onset** = 1.

### Step 4.4 – Sepsis (Sepsis-3)

Sepsis-3 is defined as **suspected infection** plus **increase in SOFA by ≥ 2**.

- **Suspected infection:** We use **microbiologyevents** (cultures) and **prescriptions** (antibiotics). We define a time window (e.g. −24 h to +72 h from ICU `intime`). A stay is “suspected infection” if it has both a culture and an antibiotic in that window.
- **SOFA delta ≥ 2:** We have a **SOFA calculator** that uses six components (respiration, coagulation, liver, cardiovascular, CNS, renal). We load labevents and chartevents in chunks, compute a **baseline SOFA** (e.g. first 12 h) and a **peak SOFA** (e.g. 6–72 h). If (peak − baseline) ≥ 2, that stay qualifies for the SOFA part.

A stay gets **sepsis_onset** = 1 only if it meets **both** suspected infection and SOFA delta ≥ 2.

All four onset targets are merged into the feature matrix. The result is saved as **features_engineered.csv** (or similar), with 128 columns (IDs, 6 cohort targets, 4 organ-support targets, 4 clinical-onset targets, and the rest features).

---

## Phase 5: Model Training

We train separate models for each target. The feature matrix is split, scaled, and then used to fit XGBoost (and optionally baseline) models.

### Step 5.1 – Load feature matrix and prepare columns

We read the engineered feature CSV. Column names are **sanitized** for XGBoost: characters like `[`, `]`, `<`, `>` are removed so they don’t cause errors.

### Step 5.2 – Train/validation/test split

We split the data:

- **60% train** – used to fit the model.
- **20% validation** – used for tuning or early stopping if needed.
- **20% test** – held out for final performance only.

For **classification** targets we use **stratified** splits so the positive rate is similar in train, validation, and test.

### Step 5.3 – Scaling

We fit a **StandardScaler** on the **training** set only, then **transform** train, validation, and test. So all numeric features have zero mean and unit variance on the training distribution, and we avoid data leakage.

### Step 5.4 – Handle class imbalance

Many targets are imbalanced (e.g. few readmissions, few deaths). We use:

- **scale_pos_weight** in XGBoost for all binary targets so the algorithm penalizes errors on the minority class more.
- **SMOTE** (synthetic oversampling) only when the positive class is very small (e.g. &lt;10%), and **not** for readmission targets (we found SMOTE hurt readmission F1, so we use only scale_pos_weight there).

### Step 5.5 – Train one model per target

We train:

- **Binary targets:** XGBoost via a shared `_train_binary_xgb()` (mortality, readmission 48h/7d, vent, vaso, RRT, AKI, ARDS, liver, sepsis).
- **Multi-class:** e.g. **los_category** (3 classes), **discharge_disposition** (3 classes) with an appropriate XGBoost objective.

We also trained baseline models (e.g. logistic regression, random forest) for comparison. In total we saved **16 models** (e.g. 12 XGB plus 4 baseline) and the **scaler**, so the dashboard can load them and run predictions.

---

## Phase 6: Dashboard

We extended the Streamlit app so it loads all new models and shows the new targets.

### Step 6.1 – Load models and scaler

The app loads the 12 prediction models (mortality, LOS category, readmission 48h/7d, discharge disposition, vent, vaso, RRT, AKI, ARDS, liver, sepsis) plus the scaler from disk.

### Step 6.2 – New panels for new targets

We added:

- **Readmission** – e.g. gauges or metrics for 48h and 7d readmission risk.
- **Discharge disposition** – e.g. bar chart of Home / Facility / Death probabilities.
- **Organ support** – bars or cards for ventilation, vasopressor, RRT.
- **Disease onset** – cards for AKI, ARDS, liver injury, sepsis.

We also extended **Actual Outcomes** so the user can see true values for all these targets when available.

### Step 6.3 – Fixes

We fixed:

- **st.progress()** – Streamlit expected a Python float; we wrapped NumPy float32 values with `float()`.
- **About and model performance** – Updated to reflect the 12 tasks and new metrics.

---

## Summary: Order of Steps

| Order | Phase | What we did |
|-------|--------|-------------|
| 1 | Cohort | Load patients/admissions/icustays → merge → deduplicate → filter (adults, complete) → add 6 targets → save cohort_labeled.csv |
| 2 | Config | Centralize item IDs and ULN in config.py |
| 3 | Features | Load cohort → add diagnosis, procedure, lab, demographic, temporal features |
| 4 | Organ support | Add vent, vasopressor, RRT from chartevents/inputevents/procedureevents (chunked where needed) |
| 5 | Clinical onset | Add AKI (KDIGO), ARDS (Berlin), liver (3× ULN), sepsis (Sepsis-3 + SOFA delta) |
| 6 | Save | Write features_engineered.csv (128 columns) |
| 7 | Train | Load CSV → sanitize columns → split 60/20/20 → scale → handle imbalance → train 12 XGB (+ baselines) → save 16 models + scaler |
| 8 | Dashboard | Load models → add panels for all targets → show predictions and actual outcomes |

---

## Challenges We Hit (and How We Fixed Them)

1. **Memory (chartevents too large)** – We switched to **chunked loading** (1.5M–2M rows per chunk) and filtered each chunk to cohort stay_ids and relevant item IDs so we never load the full table.
2. **Readmission F1 very low with SMOTE** – We **turned off SMOTE** for readmission and used only **scale_pos_weight** for XGBoost.
3. **Streamlit progress bar** – We **wrapped** progress values in `float()` to avoid NumPy float32.
4. **Windows console encoding** – We **replaced** Unicode symbols (×, —) with ASCII (e.g. “3x”, “-”) so logs display correctly.

---

*This document summarizes the steps described in Journal Entry 3 (Feature Engineering, Extended Targets & Model Training).*
