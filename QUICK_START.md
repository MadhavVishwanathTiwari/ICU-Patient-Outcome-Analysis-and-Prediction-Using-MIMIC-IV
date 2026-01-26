# Quick Start Guide
## ICU Patient Outcome Analysis Using MIMIC-IV

---

## ✅ What's Done (Step 1)

1. **Project Structure Created** - Standard data science layout
2. **Dependencies Defined** - `requirements.txt` ready
3. **Cohort Generated** - 73,181 ICU stays with labels
4. **Configuration Set** - Centralized settings in `config.py`
5. **Documentation Complete** - README, PROJECT_STATUS, this guide

---

## 📊 Generated Cohort Summary

**File**: `data/processed/cohort_labeled.csv` (18.5 MB)

| Metric | Value |
|--------|-------|
| ICU Stays | 73,181 |
| Unique Patients | 50,920 |
| Features | 19 columns |
| Mortality Rate | 11.38% |
| Mean LOS | 3.45 days |

**Target Variables**:
- `mortality` - Binary (0/1)
- `los_days` - Continuous (days)
- `los_category` - Class (0: <3d, 1: 3-7d, 2: >7d)

---

## 🚀 Commands Cheat Sheet

### Verify Setup
```powershell
python verify_setup.py
```

### Generate Cohort (already done)
```powershell
python src/data/make_cohort.py
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Start Jupyter Notebook
```powershell
jupyter notebook notebooks/01_cohort_exploration.ipynb
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/data/make_cohort.py` | Cohort generation pipeline |
| `config.py` | Project configuration |
| `verify_setup.py` | Setup checker |
| `requirements.txt` | Python dependencies |
| `data/processed/cohort_labeled.csv` | **Master cohort (OUTPUT)** |
| `notebooks/01_cohort_exploration.ipynb` | Initial analysis notebook |

---

## 🎯 Next Steps

### Immediate (Step 2)
```powershell
# 1. Install remaining packages
pip install -r requirements.txt

# 2. Explore the cohort
jupyter notebook notebooks/01_cohort_exploration.ipynb
```

### Upcoming (Steps 3-5)
1. **Feature Engineering** - Extract clinical features from:
   - Lab events (`labevents.csv.gz`)
   - Diagnoses (`diagnoses_icd.csv.gz`)
   - Procedures (`procedures_icd.csv.gz`)
   - Chart events (`chartevents.csv.gz`)

2. **Model Training** - Build and evaluate:
   - Baseline: Logistic Regression
   - Advanced: XGBoost, Random Forest
   - Ensemble methods

3. **Dashboard** - Streamlit app for predictions

---

## 💡 Quick Tips

- **All data is already compressed** - No need to unzip `.gz` files
- **Windows compatible** - All scripts work with PowerShell
- **Version control ready** - `.gitignore` excludes raw data
- **Reproducible** - Set `RANDOM_SEED = 42` in `config.py`

---

## 📖 Documentation

- **Project Overview**: `README.md`
- **Detailed Status**: `PROJECT_STATUS.md`
- **This Guide**: `QUICK_START.md`

---

## 🔍 Sample Code

### Load the Cohort
```python
import pandas as pd

# Load cohort
cohort = pd.read_csv('data/processed/cohort_labeled.csv')

print(f"Shape: {cohort.shape}")
print(f"Mortality rate: {cohort['mortality'].mean():.2%}")
```

### Basic Statistics
```python
# Target distribution
print(cohort['los_category'].value_counts())

# Demographics
print(f"Age range: {cohort['age'].min()}-{cohort['age'].max()}")
print(f"Gender: {cohort['gender'].value_counts()}")
```

---

**Ready to proceed!** 🎉

Use this guide as your reference throughout the project.
