# ICU Patient Outcome Analysis and Prediction Using MIMIC-IV

A comprehensive machine learning project for predicting patient outcomes in Intensive Care Units using the MIMIC-IV v2.2 dataset.

## 📋 Project Overview

This project analyzes ICU patient data to predict:
- **Mortality**: In-hospital mortality risk
- **Length of Stay**: ICU duration prediction
- **LOS Category**: Short (<3 days), Medium (3-7 days), or Long (>7 days) stay classification

## 🗂️ Project Structure

```
MinorProject/
├── data/
│   ├── interim/           # Intermediate processed data
│   └── processed/         # Final processed datasets
├── mimic-iv-2.2/          # Raw MIMIC-IV core data (not tracked)
├── mimic-iv-ed-2.2/       # Raw MIMIC-IV ED data (not tracked)
├── models/                # Trained model artifacts
├── notebooks/             # Jupyter notebooks for exploration
├── reports/
│   └── figures/           # Generated visualizations
├── src/
│   ├── data/              # Data processing scripts
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and evaluation
│   └── visualization/     # Plotting utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- MIMIC-IV v2.2 dataset (requires PhysioNet credentialing)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd D:\College\MinorProject
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

Ensure your MIMIC-IV data is organized as:
```
mimic-iv-2.2/mimic-iv-2.2/
├── hosp/
│   ├── patients.csv.gz
│   ├── admissions.csv.gz
│   └── ...
└── icu/
    ├── icustays.csv.gz
    └── ...
```

## 📊 Usage

### Step 1: Generate Master Cohort

Create the labeled cohort from raw data:

```bash
python src/data/make_cohort.py
```

This will:
- Load patients, admissions, and ICU stays data
- Filter for adult patients (age ≥ 18)
- Engineer target variables (mortality, LOS)
- Save to `data/processed/cohort_labeled.csv`

**Expected output**: A CSV file with ~50,000-70,000 ICU stays

### Step 2: Feature Engineering

(To be implemented)

```bash
python src/features/build_features.py
```

### Step 3: Model Training

(To be implemented)

```bash
python src/models/train_model.py
```

### Step 4: Dashboard

(To be implemented)

```bash
streamlit run src/app.py
```

## 📈 Target Variables

1. **mortality** (Binary Classification)
   - 0: Patient survived
   - 1: In-hospital death

2. **los_days** (Regression)
   - Continuous value representing ICU length of stay in days

3. **los_category** (Multi-class Classification)
   - 0: Short stay (<3 days)
   - 1: Medium stay (3-7 days)
   - 2: Long stay (>7 days)

## 🧪 Model Pipeline (Planned)

1. **Baseline Models**: Logistic Regression, Decision Trees
2. **Advanced Models**: Random Forest, XGBoost, LightGBM
3. **Deep Learning**: Neural Networks (if time permits)
4. **Ensemble Methods**: Stacking, Voting classifiers

## 📝 Data Privacy

This project uses the MIMIC-IV dataset, which contains de-identified patient data. All dates are shifted, and ages >89 are aggregated to protect patient privacy.

**Important**: Do not share or publish any raw data files.

## 👥 Authors

- Healthcare ML Team
- Minor Project - 2026

## 📄 License

This project follows the MIMIC-IV data use agreement. The MIMIC-IV dataset is licensed under PhysioNet Credentialed Health Data License 1.5.0.

## 🙏 Acknowledgments

- MIT Laboratory for Computational Physiology
- PhysioNet
- MIMIC-IV dataset creators and contributors
