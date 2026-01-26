"""
MIMIC-IV Feature Engineering Pipeline
======================================
Extracts and engineers clinical features from MIMIC-IV raw data.

Features Extracted:
- Diagnosis codes (ICD-9/ICD-10) - top N most common
- Procedure codes - top N most common
- Lab values - aggregated statistics from first 24/48 hours
- Vital signs - from chartevents
- Demographics - encoded categorical variables
- Temporal features - time-based patterns

Author: ML Healthcare Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import *


class FeatureEngineer:
    """
    Extracts and engineers clinical features from MIMIC-IV data.
    """
    
    def __init__(self, cohort_path='data/processed/cohort_labeled.csv'):
        """
        Initialize the feature engineer.
        
        Parameters:
        -----------
        cohort_path : str
            Path to the master cohort CSV file
        """
        self.cohort_path = Path(cohort_path)
        self.cohort = None
        self.features = None
        
        # Data paths
        self.hosp_dir = MIMIC_IV_HOSP
        self.icu_dir = MIMIC_IV_ICU
        
    def load_cohort(self):
        """Load the master cohort."""
        print("\n" + "=" * 70)
        print("LOADING MASTER COHORT")
        print("=" * 70)
        
        self.cohort = pd.read_csv(
            self.cohort_path,
            parse_dates=['admittime', 'dischtime', 'intime', 'outtime']
        )
        print(f"[OK] Loaded {len(self.cohort):,} ICU stays")
        
        # Initialize features dataframe with identifiers and targets
        self.features = self.cohort[[
            'subject_id', 'hadm_id', 'stay_id',
            'mortality', 'los_days', 'los_category'
        ]].copy()
        
        return self.cohort
    
    def extract_diagnosis_features(self, top_n=50):
        """
        Extract diagnosis code features (ICD-9 and ICD-10).
        
        Creates binary features for the top N most common diagnoses.
        
        Parameters:
        -----------
        top_n : int
            Number of top diagnoses to extract as features
        """
        print("\n" + "=" * 70)
        print(f"EXTRACTING DIAGNOSIS FEATURES (Top {top_n})")
        print("=" * 70)
        
        # Load diagnoses
        print("  -> Loading diagnoses_icd.csv.gz...")
        diagnoses = pd.read_csv(
            self.hosp_dir / 'diagnoses_icd.csv.gz',
            compression='gzip'
        )
        print(f"    [OK] Loaded {len(diagnoses):,} diagnosis records")
        
        # Merge with cohort to get only relevant admissions
        cohort_hadm = self.cohort[['hadm_id']].drop_duplicates()
        diagnoses = diagnoses.merge(cohort_hadm, on='hadm_id', how='inner')
        print(f"    [OK] Filtered to {len(diagnoses):,} diagnoses for cohort")
        
        # Find top N most common diagnoses
        top_diagnoses = diagnoses['icd_code'].value_counts().head(top_n).index.tolist()
        print(f"    [OK] Identified top {top_n} diagnoses")
        
        # Load diagnosis descriptions for interpretability
        print("  -> Loading diagnosis descriptions...")
        d_diagnoses = pd.read_csv(
            self.hosp_dir / 'd_icd_diagnoses.csv.gz',
            compression='gzip'
        )
        
        # Create binary features for top diagnoses
        print("  -> Creating binary diagnosis features...")
        diag_features = pd.DataFrame()
        diag_features['hadm_id'] = self.cohort['hadm_id']
        
        for idx, icd_code in enumerate(top_diagnoses, 1):
            if idx % 10 == 0:
                print(f"    Processing diagnosis {idx}/{top_n}...")
            
            # Get description for feature name
            desc = d_diagnoses[d_diagnoses['icd_code'] == icd_code]['long_title'].values
            if len(desc) > 0:
                # Create clean feature name
                desc_text = desc[0][:30].replace(' ', '_').replace(',', '').replace('/', '_')
                feature_name = f'diag_{icd_code}_{desc_text}'
            else:
                feature_name = f'diag_{icd_code}'
            
            # Create binary indicator
            hadm_with_diag = diagnoses[diagnoses['icd_code'] == icd_code]['hadm_id'].unique()
            diag_features[feature_name] = diag_features['hadm_id'].isin(hadm_with_diag).astype(int)
        
        # Merge with main features
        diag_features = diag_features.drop('hadm_id', axis=1)
        self.features = pd.concat([self.features, diag_features], axis=1)
        
        print(f"    [OK] Created {len(diag_features.columns)} diagnosis features")
        
        return self.features
    
    def extract_procedure_features(self, top_n=30):
        """
        Extract procedure code features.
        
        Creates binary features for the top N most common procedures.
        
        Parameters:
        -----------
        top_n : int
            Number of top procedures to extract as features
        """
        print("\n" + "=" * 70)
        print(f"EXTRACTING PROCEDURE FEATURES (Top {top_n})")
        print("=" * 70)
        
        # Load procedures
        print("  -> Loading procedures_icd.csv.gz...")
        procedures = pd.read_csv(
            self.hosp_dir / 'procedures_icd.csv.gz',
            compression='gzip'
        )
        print(f"    [OK] Loaded {len(procedures):,} procedure records")
        
        # Merge with cohort
        cohort_hadm = self.cohort[['hadm_id']].drop_duplicates()
        procedures = procedures.merge(cohort_hadm, on='hadm_id', how='inner')
        print(f"    [OK] Filtered to {len(procedures):,} procedures for cohort")
        
        # Find top N most common procedures
        top_procedures = procedures['icd_code'].value_counts().head(top_n).index.tolist()
        print(f"    [OK] Identified top {top_n} procedures")
        
        # Load procedure descriptions
        print("  -> Loading procedure descriptions...")
        d_procedures = pd.read_csv(
            self.hosp_dir / 'd_icd_procedures.csv.gz',
            compression='gzip'
        )
        
        # Create binary features
        print("  -> Creating binary procedure features...")
        proc_features = pd.DataFrame()
        proc_features['hadm_id'] = self.cohort['hadm_id']
        
        for idx, icd_code in enumerate(top_procedures, 1):
            if idx % 10 == 0:
                print(f"    Processing procedure {idx}/{top_n}...")
            
            # Get description
            desc = d_procedures[d_procedures['icd_code'] == icd_code]['long_title'].values
            if len(desc) > 0:
                desc_text = desc[0][:30].replace(' ', '_').replace(',', '').replace('/', '_')
                feature_name = f'proc_{icd_code}_{desc_text}'
            else:
                feature_name = f'proc_{icd_code}'
            
            # Create binary indicator
            hadm_with_proc = procedures[procedures['icd_code'] == icd_code]['hadm_id'].unique()
            proc_features[feature_name] = proc_features['hadm_id'].isin(hadm_with_proc).astype(int)
        
        # Merge with main features
        proc_features = proc_features.drop('hadm_id', axis=1)
        self.features = pd.concat([self.features, proc_features], axis=1)
        
        print(f"    [OK] Created {len(proc_features.columns)} procedure features")
        
        return self.features
    
    def extract_lab_features(self, window_hours=24):
        """
        Extract lab value features from the first N hours of ICU stay.
        
        Computes min, max, mean, std for key lab measurements.
        
        Parameters:
        -----------
        window_hours : int
            Hours from ICU admission to consider for lab values
        """
        print("\n" + "=" * 70)
        print(f"EXTRACTING LAB FEATURES (First {window_hours}h window)")
        print("=" * 70)
        
        print("  -> Loading labevents.csv.gz...")
        print("     [WARNING] This file is very large (~3GB). Loading may take 2-5 minutes...")
        
        # Load lab events (this is a large file)
        try:
            labevents = pd.read_csv(
                self.hosp_dir / 'labevents.csv.gz',
                compression='gzip',
                parse_dates=['charttime'],
                usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']
            )
            print(f"    [OK] Loaded {len(labevents):,} lab events")
        except Exception as e:
            print(f"    [WARNING] Could not load labevents: {e}")
            print(f"    [INFO] Skipping lab features for now (file too large)")
            return self.features
        
        # Merge with cohort to get ICU admission times and filter
        cohort_subset = self.cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
        labevents = labevents.merge(cohort_subset, on=['subject_id', 'hadm_id'], how='inner')
        print(f"    [OK] Filtered to {len(labevents):,} lab events for cohort")
        
        # Filter to first N hours of ICU stay
        labevents['hours_from_icu_admit'] = (
            (labevents['charttime'] - labevents['intime']).dt.total_seconds() / 3600
        )
        labevents = labevents[
            (labevents['hours_from_icu_admit'] >= 0) & 
            (labevents['hours_from_icu_admit'] <= window_hours)
        ]
        print(f"    [OK] Filtered to first {window_hours} hours: {len(labevents):,} events")
        
        # Load lab item descriptions
        print("  -> Loading lab item descriptions...")
        d_labitems = pd.read_csv(
            self.hosp_dir / 'd_labitems.csv.gz',
            compression='gzip'
        )
        
        # Select important lab items (most common and clinically relevant)
        important_labs = labevents['itemid'].value_counts().head(20).index.tolist()
        labevents = labevents[labevents['itemid'].isin(important_labs)]
        print(f"    [OK] Selected {len(important_labs)} most common lab items")
        
        # Aggregate lab values
        print("  -> Computing aggregated statistics...")
        lab_features = []
        
        for itemid in important_labs:
            lab_name = d_labitems[d_labitems['itemid'] == itemid]['label'].values
            if len(lab_name) > 0:
                lab_name = lab_name[0].replace(' ', '_').replace(',', '')[:20]
            else:
                lab_name = f'lab_{itemid}'
            
            item_data = labevents[labevents['itemid'] == itemid]
            
            # Compute statistics per stay
            agg = item_data.groupby('stay_id')['valuenum'].agg([
                ('min', 'min'),
                ('max', 'max'),
                ('mean', 'mean'),
                ('std', 'std'),
                ('count', 'count')
            ]).reset_index()
            
            agg.columns = ['stay_id'] + [f'{lab_name}_{stat}' for stat in ['min', 'max', 'mean', 'std', 'count']]
            lab_features.append(agg)
        
        # Merge all lab features
        lab_df = self.features[['stay_id']].copy()
        for lab_agg in lab_features:
            lab_df = lab_df.merge(lab_agg, on='stay_id', how='left')
        
        lab_df = lab_df.drop('stay_id', axis=1)
        self.features = pd.concat([self.features, lab_df], axis=1)
        
        print(f"    [OK] Created {len(lab_df.columns)} lab features")
        
        return self.features
    
    def extract_demographic_features(self):
        """
        Extract and encode demographic features.
        
        Creates one-hot encoded features for categorical variables.
        """
        print("\n" + "=" * 70)
        print("EXTRACTING DEMOGRAPHIC FEATURES")
        print("=" * 70)
        
        # Get demographic columns from cohort
        demo_cols = ['age', 'gender', 'ethnicity', 'insurance', 'admission_type', 
                     'first_careunit', 'marital_status']
        
        available_cols = [col for col in demo_cols if col in self.cohort.columns]
        demo_df = self.cohort[available_cols].copy()
        
        print(f"  -> Processing {len(available_cols)} demographic variables...")
        
        # Age as is (already numeric)
        if 'age' in demo_df.columns:
            self.features['age'] = demo_df['age']
        
        # Gender - binary encoding
        if 'gender' in demo_df.columns:
            self.features['gender_M'] = (demo_df['gender'] == 'M').astype(int)
        
        # Ethnicity - group into major categories
        if 'ethnicity' in demo_df.columns:
            print("  -> Grouping ethnicity into major categories...")
            ethnicity_grouped = demo_df['ethnicity'].fillna('UNKNOWN')
            
            # Simplified categories
            self.features['ethnicity_WHITE'] = ethnicity_grouped.str.contains('WHITE', case=False, na=False).astype(int)
            self.features['ethnicity_BLACK'] = ethnicity_grouped.str.contains('BLACK', case=False, na=False).astype(int)
            self.features['ethnicity_HISPANIC'] = ethnicity_grouped.str.contains('HISPANIC', case=False, na=False).astype(int)
            self.features['ethnicity_ASIAN'] = ethnicity_grouped.str.contains('ASIAN', case=False, na=False).astype(int)
            self.features['ethnicity_OTHER'] = (
                ~ethnicity_grouped.str.contains('WHITE|BLACK|HISPANIC|ASIAN', case=False, na=False)
            ).astype(int)
        
        # Insurance - one-hot encoding
        if 'insurance' in demo_df.columns:
            insurance_dummies = pd.get_dummies(
                demo_df['insurance'].fillna('Unknown'), 
                prefix='insurance'
            )
            self.features = pd.concat([self.features, insurance_dummies], axis=1)
        
        # Admission type
        if 'admission_type' in demo_df.columns:
            admission_dummies = pd.get_dummies(
                demo_df['admission_type'].fillna('Unknown'),
                prefix='admission'
            )
            self.features = pd.concat([self.features, admission_dummies], axis=1)
        
        # First care unit (ICU type)
        if 'first_careunit' in demo_df.columns:
            icu_dummies = pd.get_dummies(
                demo_df['first_careunit'].fillna('Unknown'),
                prefix='icu'
            )
            self.features = pd.concat([self.features, icu_dummies], axis=1)
        
        print(f"    [OK] Created demographic features")
        print(f"    Total feature count: {len(self.features.columns)}")
        
        return self.features
    
    def extract_temporal_features(self):
        """
        Extract temporal features from admission times.
        
        Creates features based on time patterns (hour, day of week, month, season).
        """
        print("\n" + "=" * 70)
        print("EXTRACTING TEMPORAL FEATURES")
        print("=" * 70)
        
        # Extract from admission time
        if 'admittime' in self.cohort.columns:
            admittime = pd.to_datetime(self.cohort['admittime'])
            
            # Hour of day
            self.features['admit_hour'] = admittime.dt.hour
            self.features['admit_is_night'] = ((admittime.dt.hour >= 22) | (admittime.dt.hour <= 6)).astype(int)
            
            # Day of week (0=Monday, 6=Sunday)
            self.features['admit_day_of_week'] = admittime.dt.dayofweek
            self.features['admit_is_weekend'] = (admittime.dt.dayofweek >= 5).astype(int)
            
            # Month
            self.features['admit_month'] = admittime.dt.month
            
            # Season
            month = admittime.dt.month
            self.features['admit_season_winter'] = month.isin([12, 1, 2]).astype(int)
            self.features['admit_season_spring'] = month.isin([3, 4, 5]).astype(int)
            self.features['admit_season_summer'] = month.isin([6, 7, 8]).astype(int)
            self.features['admit_season_fall'] = month.isin([9, 10, 11]).astype(int)
            
            print(f"    [OK] Created temporal features")
        
        return self.features
    
    def build_feature_matrix(self):
        """
        Main pipeline to build the complete feature matrix.
        """
        print("\n" + "=" * 70)
        print("MIMIC-IV FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load cohort
        self.load_cohort()
        
        # Extract features
        self.extract_diagnosis_features(top_n=TOP_N_DIAGNOSES)
        self.extract_procedure_features(top_n=TOP_N_PROCEDURES)
        self.extract_demographic_features()
        self.extract_temporal_features()
        
        # Note: Lab features are optional due to large file size
        # Uncomment if you have enough memory:
        # self.extract_lab_features(window_hours=24)
        
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 70)
        print(f"  Final feature matrix shape: {self.features.shape}")
        print(f"  Total features: {len(self.features.columns) - 6} (excluding IDs and targets)")
        print(f"  ICU stays: {len(self.features):,}")
        
        return self.features
    
    def save_features(self, output_path='data/processed/features_engineered.csv'):
        """
        Save the feature matrix to CSV.
        
        Parameters:
        -----------
        output_path : str
            Path to save the features
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n>> Saving features to: {output_path}")
        self.features.to_csv(output_path, index=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   [OK] File size: {file_size_mb:.2f} MB")
        print(f"   [OK] Location: {output_path.absolute()}")


def main():
    """
    Main execution function.
    """
    # Initialize feature engineer
    engineer = FeatureEngineer(cohort_path='data/processed/cohort_labeled.csv')
    
    # Build features
    features = engineer.build_feature_matrix()
    
    # Save features
    engineer.save_features()
    
    print("\n*** Feature engineering complete! Ready for model training.")


if __name__ == "__main__":
    main()
