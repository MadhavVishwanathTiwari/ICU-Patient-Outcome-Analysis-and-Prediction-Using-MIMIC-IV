# Glossary: Understanding Journal Entry 3

*A learner's guide to every technical topic and keyword in the journal. Assumes basic statistics knowledge (mean, standard deviation).*

---

## How to Use This Document

Read entries in order, or jump to a term when you encounter it in the journal. Each entry explains the concept and how it appears in this project. New terms used within an explanation are **bolded**.

---

## Part 1: Machine Learning Fundamentals

### **Machine Learning (ML)**

Using computers to find patterns in data and make predictions without being explicitly programmed for each case. In this project: we train models to predict patient outcomes (e.g., mortality, need for ventilation) from their data.

**Where in Journal 3:** The whole entry is about building and training ML models.

---

### **Target / Target Variable / Label**

The outcome we want to predict. It’s the “answer” for each row. Examples: mortality (did the patient die?), need_vent_any (did they need a ventilator?).

**Where in Journal 3:** “Cohort targets,” “12 tasks trained,” “extended targets.”

---

### **Feature / Predictor / Input Variable**

A piece of information we use to predict the target. Examples: age, lab values, diagnosis codes. Features are the columns (other than IDs and the target) in our data.

**Where in Journal 3:** “Feature matrix,” “112 features,” “feature engineering.”

---

### **Feature Matrix**

A table where each row is a patient/stay and each column is a feature or target. The shape (rows × columns) describes its size.

**Where in Journal 3:** “128 (112 features + 16 target/ID)” — 128 columns total.

---

### **Feature Engineering**

Creating new features from raw data to help the model. Examples: turning discharge text into numbers (Home=0, Facility=1, Death=2), aggregating labs (min, max, mean) over 24 hours.

**Where in Journal 3:** Title; “Feature Engineering, Extended Targets.”

---

### **Binary Classification**

Predicting one of two classes: yes/no, 0/1. Examples: mortality, need for ventilation, sepsis onset.

**Where in Journal 3:** “binary targets,” “_train_binary_xgb.”

---

### **Multi-class Classification**

Predicting one of more than two classes. Examples: LOS category (short/medium/long), discharge disposition (Home/Facility/Death).

**Where in Journal 3:** “3-class XGBoost,” “discharge disposition (3-class).”

---

### **Training / Learning**

Using data with known targets so the model can learn patterns. The model adjusts its internal parameters to match the training examples.

**Where in Journal 3:** “Model training,” “train_all_models.”

---

### **Training Set / Validation Set / Test Set**

- **Training:** Data used to train the model.
- **Validation:** Data used to compare models and tune settings; not used for training.
- **Test:** Data used only at the end to report final performance; never seen during training.

Splitting avoids **overfitting** (doing well on training data but poorly on new data).

**Where in Journal 3:** “60% train / 20% validation / 20% test,” “Data split.”

---

### **Stratified Split**

Splitting data so that each subset has about the same proportion of each class. Important when one class is rare (e.g., death).

**Where in Journal 3:** “stratified for classification.”

---

### **Scaling / StandardScaler**

Scaling numerical features to have mean 0 and standard deviation 1. Helps many algorithms (e.g., logistic regression) that depend on feature ranges. We fit the scaler on the training set and apply the same transformation to validation and test sets.

**Where in Journal 3:** “StandardScaler fit on train, transform train/val/test.”

---

### **Class Imbalance**

When one class is much more common than the other (e.g., few deaths vs many survivors). Models can favor the majority class and perform poorly on the minority class.

**Where in Journal 3:** “handle imbalance,” “scale_pos_weight,” “SMOTE.”

---

### **SMOTE (Synthetic Minority Over-sampling Technique)**

A method to fix class imbalance by generating synthetic examples of the minority class. Creates new points between existing minority points. Used when the minority class is small (<10%) but not for readmission (where it hurt performance).

**Where in Journal 3:** “SMOTE when pos < 10% except readmission,” “SMOTE worsened ROC and F1” for readmission.

---

### **scale_pos_weight**

An XGBoost parameter that increases the weight of positive (minority) examples so the model pays more attention to them. Ratio often set as (number of negatives) / (number of positives).

**Where in Journal 3:** “scale_pos_weight for all binary targets,” “SMOTE skipped; used scale_pos_weight only.”

---

### **XGBoost (Extreme Gradient Boosting)**

A strong model for classification and regression. Builds many small “weak” models (trees) that vote together. Handles mixed features and imbalance well.

**Where in Journal 3:** “XGBoost,” “_train_binary_xgb,” “10 extended XGB” models.

---

### **Logistic Regression**

A simpler classifier that predicts probabilities using a linear combination of features. Often used as a baseline.

**Where in Journal 3:** “Logistic Regression (baseline),” “mortality_lr,” “los_class_lr.”

---

### **Random Forest**

An ensemble of many decision trees; each tree votes, and the majority wins. Good for tabular data and resistant to overfitting.

**Where in Journal 3:** “Random Forest,” “mortality_rf,” “los_class_rf.”

---

### **Pipeline**

A fixed sequence of steps: cohort → features → training → dashboard. Each step feeds the next.

**Where in Journal 3:** “Pipeline completion,” “cohort → features → training → dashboard flow.”

---

### **Model / Algorithm**

A mathematical way to relate features to the target. “Model” often refers to a trained instance (with learned parameters) saved for later use.

**Where in Journal 3:** “16 models saved,” “load 12 prediction models.”

---

### **Random Seed**

A number that makes random operations reproducible. Same seed → same splits, same results. Here: 42.

**Where in Journal 3:** Implicit in training configuration; mentioned in config.

---

### **Hyperparameters**

Settings we choose before training (e.g., tree depth, learning rate). They affect model behavior but are not learned from data.

**Where in Journal 3:** In config: n_estimators, max_depth, learning_rate, etc.

---

## Part 2: Evaluation Metrics

### **Accuracy**

Fraction of predictions that are correct. Good for balanced classes; misleading when classes are imbalanced.

**Where in Journal 3:** Test performance table, LOS category, discharge disposition.

---

### **Precision**

Of all predicted positives, how many were actually positive? High precision = few false alarms.

Formula: True Positives / (True Positives + False Positives)

**Where in Journal 3:** In results; combined with F1, ROC-AUC in the table.

---

### **Recall (Sensitivity)**

Of all actual positives, how many did we find? High recall = few missed cases.

Formula: True Positives / (True Positives + False Negatives)

**Where in Journal 3:** In results; combined with F1, ROC-AUC.

---

### **F1-Score / F1**

Harmonic mean of precision and recall. Balances precision and recall; used when both matter (e.g., rare events).

**Where in Journal 3:** “Test performance” table; “F1 near zero” for readmission.

---

### **ROC-AUC (Area Under the ROC Curve)**

Measures how well the model separates positive and negative classes at different thresholds. 1.0 = perfect; 0.5 = random guessing. Useful for imbalanced data and comparing models.

**Where in Journal 3:** Test performance table for binary targets.

---

### **F1-macro**

For multi-class problems, average F1 across all classes, treating each class equally. Used for LOS category and discharge disposition.

**Where in Journal 3:** LOS category and discharge disposition in results.

---

## Part 3: Data & Preprocessing

### **Cohort**

A defined group of patients used for analysis. Here: adult ICU stays with the required data.

**Where in Journal 3:** “Cohort building,” “cohort targets,” “filter to cohort.”

---

### **Merge / Join**

Combining tables on common columns (e.g., subject_id, hadm_id). Like “VLOOKUP” in spreadsheets.

**Where in Journal 3:** “Merge patients → admissions → icustays.”

---

### **Deduplicate**

Removing duplicate rows. Here: one row per ICU stay (stay_id).

**Where in Journal 3:** “Deduplicate,” “remove duplicate rows by stay_id.”

---

### **NaN (Not a Number) / Missing Value**

Placeholder for missing data. Often handled by filling with a default (e.g., 0) or dropping rows.

**Where in Journal 3:** “fill NaN readmission with 0,” “left merge (NaN if no value).”

---

### **fillna**

Filling missing values with a chosen value. Example: ethnicity.fillna('UNKNOWN').

**Where in Journal 3:** “fillna → Unknown/UNKNOWN.”

---

### **Left Merge**

A join that keeps all rows from the left table and adds matching rows from the right. Rows with no match get NaN in new columns.

**Where in Journal 3:** “left merge (NaN if no value)” for lab features.

---

### **Aggregation**

Summarizing many values into one (e.g., min, max, mean, std, count). Used for labs per stay over 24 hours.

**Where in Journal 3:** “aggregate (min, max, mean, std, count) per stay.”

---

### **One-hot Encoding**

Turning a categorical variable into multiple binary columns, one per category. Example: Insurance A=1,0,0; B=0,1,0; C=0,0,1.

**Where in Journal 3:** “insurance/admission/ICU (one-hot).”

---

### **Binary Indicator**

A 0/1 column indicating presence or absence of something. Example: diag_J10.0_influenza = 1 if patient had that ICD code.

**Where in Journal 3:** “top-N ICD codes as binary indicators.”

---

### **Chunked Loading**

Reading a large file in parts (e.g., 2M rows at a time) instead of loading it all at once. Reduces memory use.

**Where in Journal 3:** “chunked loading (1.5M–2M rows per chunk),” “MemoryError on 8 GB RAM.”

---

### **MemoryError**

Error when the program tries to use more RAM than available. Fixed here by chunked loading.

**Where in Journal 3:** “Loading chartevents (~313M rows) caused MemoryError.”

---

## Part 4: Clinical Concepts & Data

### **MIMIC-IV**

Medical Information Mart for Intensive Care, version IV — a public ICU dataset with patients, admissions, stays, labs, and events.

**Where in Journal 3:** Data source for cohort and features.

---

### **ICU (Intensive Care Unit)**

Hospital unit for very sick patients who need close monitoring and support.

**Where in Journal 3:** “ICU stays,” “ICU readmission.”

---

### **ICU Stay / stay_id**

A single continuous period in the ICU. One patient can have several stays (e.g., readmission).

**Where in Journal 3:** “one row per ICU stay,” “stay_id.”

---

### **Readmission**

Return to the ICU after discharge. Here: within 48 hours or 7 days.

**Where in Journal 3:** “icu_readmit_48h,” “icu_readmit_7d,” “ICU readmission.”

---

### **Discharge Disposition**

Where the patient went after leaving the hospital: Home (0), Facility (1), Death (2).

**Where in Journal 3:** “discharge_disposition,” “Home / Facility / Death.”

---

### **Mortality**

Death during the hospital stay. Binary (0 = survived, 1 = died).

**Where in Journal 3:** “mortality,” “hospital_expire_flag.”

---

### **LOS (Length of Stay)**

How long a patient stayed in the ICU, usually in days. Categorized here as: Short (<3d), Medium (3–7d), Long (>7d).

**Where in Journal 3:** “los_days,” “los_category.”

---

### **Ventilation / Mechanical Ventilation**

Use of a machine to support breathing (a ventilator).

**Where in Journal 3:** “need_vent_any,” “ventilation targets,” “chartevents.”

---

### **Vasopressor**

Drug that raises blood pressure (e.g., norepinephrine). Indicates severe circulatory issues.

**Where in Journal 3:** “need_vasopressor_any,” “vasopressor infusions,” “inputevents.”

---

### **RRT (Renal Replacement Therapy)**

Treatment when kidneys fail (e.g., dialysis). Includes hemodialysis and similar procedures.

**Where in Journal 3:** “need_rrt_any,” “procedureevents,” “dialysis ICD 5A1D.”

---

### **AKI (Acute Kidney Injury)**

Rapid decline in kidney function. Detected here using **KDIGO** (creatinine) criteria.

**Where in Journal 3:** “aki_onset,” “KDIGO creatinine.”

---

### **KDIGO (Kidney Disease: Improving Global Outcomes)**

International guidelines for AKI. Uses creatinine rise: e.g., ≥0.3 mg/dL in 48h or ≥1.5× baseline in 7 days.

**Where in Journal 3:** “KDIGO creatinine: baseline first 6h; +0.3 in 48h or 1.5× in 7d.”

---

### **ARDS (Acute Respiratory Distress Syndrome)**

Severe lung failure with low oxygen. Defined here using the **Berlin definition**.

**Where in Journal 3:** “ards_onset,” “Berlin: P/F ratio ≤300.”

---

### **Berlin Definition (ARDS)**

ARDS criteria that include P/F ratio (PaO₂/FiO₂) ≤300 with positive end-expiratory pressure (PEEP) ≥5.

**Where in Journal 3:** “Berlin: P/F ratio ≤300.”

---

### **P/F Ratio (PaO₂/FiO₂)**

Partial pressure of oxygen in blood (PaO₂) divided by fraction of inspired oxygen (FiO₂). Lower ratio = worse gas exchange.

**Where in Journal 3:** “P/F ratio ≤300.”

---

### **FiO₂ (Fraction of Inspired Oxygen)**

Fraction of oxygen in inspired air (0.21–1.0). Higher values mean more oxygen support.

**Where in Journal 3:** “FiO₂ from chartevents.”

---

### **PaO₂ (Partial Pressure of Oxygen)**

Oxygen pressure in arterial blood. Reflects how well the lungs are oxygenating.

**Where in Journal 3:** Lab item for SOFA and P/F ratio.

---

### **Sepsis**

Life-threatening organ dysfunction caused by an infection. Defined here as **Sepsis-3**.

**Where in Journal 3:** “sepsis_onset,” “Sepsis-3.”

---

### **Sepsis-3**

Definition: suspected infection plus acute rise in **SOFA** score ≥2 points.

**Where in Journal 3:** “Sepsis-3: suspected infection + SOFA delta ≥2.”

---

### **SOFA (Sequential Organ Failure Assessment)**

Score measuring failure of six organs: respiratory, coagulation, liver, cardiovascular, central nervous system, renal. Each 0–4; total 0–24. Higher = worse.

**Where in Journal 3:** “SOFA baseline (first 12h) vs peak (6–72h),” “6-component SOFA.”

---

### **SOFA Delta**

Change in SOFA score over time. Delta ≥2 indicates new organ dysfunction (central to Sepsis-3).

**Where in Journal 3:** “SOFA delta ≥2.”

---

### **Suspected Infection**

Here: microbiology culture positive or antibiotics given within a time window around admission (-24h to +72h).

**Where in Journal 3:** “microbiology cultures + antibiotic prescriptions in peri-admission window.”

---

### **Liver Injury**

Here: AST, ALT, or bilirubin above 3× the **ULN** (upper limit of normal).

**Where in Journal 3:** “AST/ALT/bilirubin >3× ULN.”

---

### **ULN (Upper Limit of Normal)**

Maximum “normal” value for a lab (e.g., ALT 40 U/L). Above ULN suggests abnormality.

**Where in Journal 3:** “ULN for liver injury,” config thresholds.

---

### **Creatinine**

Blood marker of kidney function. Rising creatinine suggests AKI.

**Where in Journal 3:** “KDIGO creatinine,” lab itemid.

---

### **Bilirubin**

Marker of liver function. High values indicate liver injury.

**Where in Journal 3:** “bilirubin,” SOFA liver component.

---

### **ALT / AST**

Enzymes indicating liver damage (alanine aminotransferase, aspartate aminotransferase).

**Where in Journal 3:** “AST/ALT/bilirubin,” “ALT_ITEMID,” “AST_ITEMID.”

---

### **Platelets**

Blood cells involved in clotting. Low counts affect the coagulation part of SOFA.

**Where in Journal 3:** “PLATELETS_ITEMID,” SOFA component.

---

### **MAP (Mean Arterial Pressure)**

Average blood pressure. Used in the cardiovascular part of SOFA.

**Where in Journal 3:** “MAP_ITEMID,” SOFA vitals.

---

### **GCS (Glasgow Coma Scale)**

Score (3–15) for level of consciousness. Used in the central nervous system part of SOFA.

**Where in Journal 3:** “GCS_ITEMID,” SOFA component.

---

## Part 5: MIMIC-IV Data Structure

### **itemid**

Numeric ID for a measurement or event in MIMIC (e.g., specific lab, vital, or procedure). Different tables use different itemids.

**Where in Journal 3:** “vent itemids,” “vasopressor itemids,” “lab itemids.”

---

### **chartevents**

Table of nurse-charted measurements (vitals, FiO₂, etc.) during an ICU stay.

**Where in Journal 3:** “chartevents,” “FiO₂ from chartevents,” ventilation.

---

### **inputevents**

Table of medications and fluids given (e.g., vasopressor infusions).

**Where in Journal 3:** “inputevents,” vasopressor targets.

---

### **labevents**

Table of lab results (creatinine, bilirubin, etc.).

**Where in Journal 3:** “labevents,” “chunked lab load.”

---

### **procedureevents**

Table of procedures performed (e.g., dialysis, RRT items).

**Where in Journal 3:** “procedureevents,” RRT extraction.

---

### **procedures_icd**

Table of procedure diagnosis codes (ICD). Used here for dialysis (5A1D).

**Where in Journal 3:** “procedures_icd,” “dialysis ICD 5A1D.”

---

### **ICD Code (International Classification of Diseases)**

Standard codes for diagnoses and procedures. ICD-9 and ICD-10 are different versions.

**Where in Journal 3:** “ICD codes,” “top-N ICD codes,” “5A1D.”

---

### **subject_id, hadm_id, stay_id**

- **subject_id:** Patient identifier  
- **hadm_id:** Hospital admission identifier  
- **stay_id:** ICU stay identifier

**Where in Journal 3:** “merge on subject_id, hadm_id,” “cohort stay_ids.”

---

### **intime, outtime**

ICU admission and discharge timestamps for a stay.

**Where in Journal 3:** “hours between outtime and next intime,” “first 12h,” “6–72h window.”

---

## Part 6: Software & Technical

### **Configuration / config.py**

Central file for paths, constants, and hyperparameters (itemids, ULN, model settings). Easier to change without editing many files.

**Where in Journal 3:** “config.py,” “centralize itemids.”

---

### **CSV (Comma-Separated Values)**

Simple text format for tabular data. Common for exports and imports.

**Where in Journal 3:** “cohort_labeled.csv,” “features_engineered.csv.”

---

### **Streamlit**

Python library for building web dashboards. Used here for the ICU prediction dashboard.

**Where in Journal 3:** “Dashboard (app.py),” “st.progress().”

---

### **float32**

32-bit floating-point type. Some libraries (e.g., Streamlit) expect a Python `float`, not NumPy `float32`, so we convert with `float()`.

**Where in Journal 3:** “st.progress() failed with NumPy float32.”

---

### **Sanitize (column names)**

Cleaning column names (e.g., removing `[`, `]`, `<`, `>`) so they work with tools like XGBoost.

**Where in Journal 3:** “Sanitize for XGBoost: remove `[`, `]`, `<`, `>`.”

---

### **Encoding (character)**

How text is stored (e.g., UTF-8). Unicode symbols (×, —) can fail on some Windows consoles, so we use ASCII (3x, hyphen).

**Where in Journal 3:** “Console encoding: Unicode corrupted on Windows.”

---

## Part 7: Concepts Mentioned for Future Work

### **Ensemble**

Combining multiple models (e.g., averaging predictions) to improve performance.

**Where in Journal 3:** “Consider ensemble” in Tasks for Next Meeting.

---

### **Threshold Tuning**

Changing the cutoff above which we predict “positive” (e.g., 0.5 → 0.4) to improve precision or recall.

**Where in Journal 3:** “threshold tuning for low-performers.”

---

### **Feature Importance / SHAP**

Ways to explain which features the model uses most. **SHAP** (SHapley Additive exPlanations) assigns each feature a contribution to each prediction.

**Where in Journal 3:** “Add feature importance / SHAP for model interpretability.”

---

### **Cross-validation**

Splitting data into multiple folds, training on some and evaluating on others in rotation. Gives more stable performance estimates.

**Where in Journal 3:** “Optional: external validation or cross-validation.”

---

## Quick Reference: Metrics at a Glance


| Metric    | What it measures                              |
| --------- | --------------------------------------------- |
| Accuracy  | Fraction correct                              |
| Precision | Of predicted positives, how many correct?     |
| Recall    | Of actual positives, how many did we find?    |
| F1        | Balance of precision and recall               |
| ROC-AUC   | Ability to separate classes (higher = better) |


---

## Reading Order Suggestion

1. Part 1 (ML fundamentals)
2. Part 2 (Metrics)
3. Part 3 (Data & preprocessing)
4. Part 4 (Clinical) — skim if needed
5. Part 5 (MIMIC-IV structure)
6. Part 6 (Software)

Then read Journal Entry 3 with this glossary beside you.

---

*End of Glossary*