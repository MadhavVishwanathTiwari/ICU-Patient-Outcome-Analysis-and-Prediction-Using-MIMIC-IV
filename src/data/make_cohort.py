"""
MIMIC-IV Cohort Generation Script
==================================
Creates the master cohort (ground truth) dataframe from MIMIC-IV raw data.

This script:
- Reads patients, admissions, and ICU stays data
- Filters for adult patients (age >= 18)
- Engineers target variables for outcome prediction
- Saves the labeled cohort for downstream analysis

Author: ML Healthcare Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CohortBuilder:
    """
    Builds the master cohort from MIMIC-IV raw data files.
    """
    
    def __init__(self, data_root='mimic-iv-2.2/mimic-iv-2.2'):
        """
        Initialize the cohort builder.
        
        Parameters:
        -----------
        data_root : str
            Root directory containing MIMIC-IV data folders
        """
        self.data_root = Path(data_root)
        self.hosp_dir = self.data_root / 'hosp'
        self.icu_dir = self.data_root / 'icu'
        
    def load_data(self):
        """
        Load required datasets from compressed CSV files.
        
        Returns:
        --------
        tuple : (patients_df, admissions_df, icustays_df)
        """
        print(">> Loading datasets...")
        
        # Load patients
        print("  -> Loading patients.csv.gz...")
        patients = pd.read_csv(
            self.hosp_dir / 'patients.csv.gz',
            compression='gzip',
            parse_dates=['dod']
        )
        print(f"    [OK] Loaded {len(patients):,} patients")
        
        # Load admissions
        print("  -> Loading admissions.csv.gz...")
        admissions = pd.read_csv(
            self.hosp_dir / 'admissions.csv.gz',
            compression='gzip',
            parse_dates=['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        )
        print(f"    [OK] Loaded {len(admissions):,} hospital admissions")
        
        # Load ICU stays
        print("  -> Loading icustays.csv.gz...")
        icustays = pd.read_csv(
            self.icu_dir / 'icustays.csv.gz',
            compression='gzip',
            parse_dates=['intime', 'outtime']
        )
        print(f"    [OK] Loaded {len(icustays):,} ICU stays")
        
        return patients, admissions, icustays
    
    def calculate_age(self, patients, admissions):
        """
        Calculate patient age at admission.
        
        MIMIC-IV uses anchor_year and anchor_age for privacy.
        Age at admission = anchor_age + (admission_year - anchor_year)
        
        Parameters:
        -----------
        patients : pd.DataFrame
            Patients dataframe with anchor_age and anchor_year
        admissions : pd.DataFrame
            Admissions dataframe with admittime
            
        Returns:
        --------
        pd.DataFrame : Admissions with 'age' column added
        """
        print("\n>> Calculating patient ages...")
        
        # Merge to get anchor info
        admissions_with_age = admissions.merge(
            patients[['subject_id', 'anchor_age', 'anchor_year']],
            on='subject_id',
            how='left'
        )
        
        # Extract admission year
        admissions_with_age['admission_year'] = admissions_with_age['admittime'].dt.year
        
        # Calculate age at admission
        admissions_with_age['age'] = (
            admissions_with_age['anchor_age'] + 
            (admissions_with_age['admission_year'] - admissions_with_age['anchor_year'])
        )
        
        print(f"  [OK] Age range: {admissions_with_age['age'].min():.0f} - {admissions_with_age['age'].max():.0f} years")
        
        return admissions_with_age
    
    def merge_datasets(self, patients, admissions, icustays):
        """
        Merge patients, admissions, and ICU stays into a single cohort.
        
        Parameters:
        -----------
        patients : pd.DataFrame
        admissions : pd.DataFrame
        icustays : pd.DataFrame
        
        Returns:
        --------
        pd.DataFrame : Merged cohort dataframe
        """
        print("\n>> Merging datasets...")
        
        # Calculate ages
        admissions = self.calculate_age(patients, admissions)
        
        # Merge admissions with ICU stays
        cohort = icustays.merge(
            admissions,
            on=['subject_id', 'hadm_id'],
            how='inner'
        )
        print(f"  [OK] After merging admissions + ICU stays: {len(cohort):,} records")
        
        # Add patient demographics
        cohort = cohort.merge(
            patients[['subject_id', 'gender', 'dod']],
            on='subject_id',
            how='left'
        )
        print(f"  [OK] After adding patient info: {len(cohort):,} records")
        
        return cohort
    
    def apply_filters(self, cohort):
        """
        Apply inclusion/exclusion criteria.
        
        Parameters:
        -----------
        cohort : pd.DataFrame
            
        Returns:
        --------
        pd.DataFrame : Filtered cohort
        """
        print("\n>> Applying filters...")
        
        initial_count = len(cohort)
        
        # Filter 1: Adults only (age >= 18)
        cohort = cohort[cohort['age'] >= 18].copy()
        print(f"  [OK] Age >= 18: {len(cohort):,} records (removed {initial_count - len(cohort):,})")
        
        # Filter 2: Remove records with missing critical data
        before = len(cohort)
        cohort = cohort.dropna(subset=['los', 'hospital_expire_flag'])
        if before > len(cohort):
            print(f"  [OK] Removed {before - len(cohort):,} records with missing LOS or mortality flag")
        
        return cohort
    
    def engineer_targets(self, cohort):
        """
        Engineer target variables for prediction tasks.
        
        Target Variables:
        -----------------
        1. mortality (binary): In-hospital mortality
        2. los_days (continuous): ICU length of stay in days
        3. los_category (categorical): Short (<3d), Medium (3-7d), Long (>7d)
        
        Parameters:
        -----------
        cohort : pd.DataFrame
        
        Returns:
        --------
        pd.DataFrame : Cohort with target variables
        """
        print("\n>> Engineering target variables...")
        
        # Target 1: Mortality (binary)
        cohort['mortality'] = cohort['hospital_expire_flag'].astype(int)
        mortality_rate = cohort['mortality'].mean() * 100
        print(f"  [OK] Mortality rate: {mortality_rate:.2f}%")
        
        # Target 2: Length of Stay in Days (continuous)
        # MIMIC-IV 'los' is in days (not hours as in MIMIC-III)
        cohort['los_days'] = cohort['los'].astype(float)
        print(f"  [OK] LOS range: {cohort['los_days'].min():.2f} - {cohort['los_days'].max():.2f} days")
        print(f"  [OK] Mean LOS: {cohort['los_days'].mean():.2f} days (median: {cohort['los_days'].median():.2f})")
        
        # Target 3: Length of Stay Category (categorical)
        cohort['los_category'] = pd.cut(
            cohort['los_days'],
            bins=[0, 3, 7, float('inf')],
            labels=[0, 1, 2],
            right=False
        ).astype(int)
        
        los_dist = cohort['los_category'].value_counts().sort_index()
        print(f"  [OK] LOS distribution:")
        print(f"      - Short (<3 days): {los_dist[0]:,} ({los_dist[0]/len(cohort)*100:.1f}%)")
        print(f"      - Medium (3-7 days): {los_dist[1]:,} ({los_dist[1]/len(cohort)*100:.1f}%)")
        print(f"      - Long (>7 days): {los_dist[2]:,} ({los_dist[2]/len(cohort)*100:.1f}%)")
        
        return cohort
    
    def select_columns(self, cohort):
        """
        Select and organize final columns for the cohort.
        
        Parameters:
        -----------
        cohort : pd.DataFrame
        
        Returns:
        --------
        pd.DataFrame : Cohort with selected columns
        """
        # Core identifiers
        id_cols = ['subject_id', 'hadm_id', 'stay_id']
        
        # Demographics
        demo_cols = ['gender', 'age', 'ethnicity', 'marital_status', 'insurance']
        
        # Admission details
        admission_cols = [
            'admission_type', 'admission_location', 'discharge_location',
            'first_careunit', 'last_careunit'
        ]
        
        # Temporal features
        time_cols = ['admittime', 'dischtime', 'intime', 'outtime']
        
        # Target variables
        target_cols = ['mortality', 'los_days', 'los_category']
        
        # Select available columns
        all_cols = id_cols + demo_cols + admission_cols + time_cols + target_cols
        available_cols = [col for col in all_cols if col in cohort.columns]
        
        return cohort[available_cols]
    
    def build_cohort(self):
        """
        Main pipeline to build the cohort.
        
        Returns:
        --------
        pd.DataFrame : Final labeled cohort
        """
        print("=" * 70)
        print("MIMIC-IV COHORT GENERATION")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Load data
        patients, admissions, icustays = self.load_data()
        
        # Merge datasets
        cohort = self.merge_datasets(patients, admissions, icustays)
        
        # Apply filters
        cohort = self.apply_filters(cohort)
        
        # Engineer targets
        cohort = self.engineer_targets(cohort)
        
        # Select final columns
        cohort = self.select_columns(cohort)
        
        print("\n" + "=" * 70)
        print(f"SUCCESS: COHORT GENERATION COMPLETE")
        print(f"   Final cohort size: {len(cohort):,} ICU stays")
        print(f"   Number of unique patients: {cohort['subject_id'].nunique():,}")
        print(f"   Features: {len(cohort.columns)} columns")
        print("=" * 70)
        
        return cohort
    
    def save_cohort(self, cohort, output_path='data/processed/cohort_labeled.csv'):
        """
        Save the cohort to CSV.
        
        Parameters:
        -----------
        cohort : pd.DataFrame
        output_path : str
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n>> Saving cohort to: {output_path}")
        cohort.to_csv(output_path, index=False)
        
        # Print file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   [OK] File size: {file_size_mb:.2f} MB")
        print(f"   [OK] Location: {output_path.absolute()}")


def main():
    """
    Main execution function.
    """
    # Initialize builder
    builder = CohortBuilder(data_root='mimic-iv-2.2/mimic-iv-2.2')
    
    # Build cohort
    cohort = builder.build_cohort()
    
    # Save cohort
    builder.save_cohort(cohort)
    
    print("\n*** All done! You can now use 'data/processed/cohort_labeled.csv' for analysis.")
    

if __name__ == "__main__":
    main()
