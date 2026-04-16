"""
MIMIC-IV Feature Engineering Pipeline (v2)
=========================================
Builds a clinically grounded predictor matrix using first-24h physiology,
demographics, and curated comorbidity context features.

Key design choices:
- No top-N frequency diagnosis/procedure feature families.
- Predictor extraction restricted to first 24h from ICU admission.
- Existing target columns remain compatible with downstream training code.
"""

from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    MIMIC_IV_HOSP,
    MIMIC_IV_ICU,
    FEATURE_WINDOW_HOURS_V2,
    VITAL_ITEMIDS_V2,
    LAB_ITEMIDS_V2,
    URINE_OUTPUT_ITEMIDS_V2,
    COMORBIDITY_PREFIXES_V2,
    VENT_ITEMIDS,
    VASOPRESSOR_ITEMIDS,
    RRT_ITEMIDS,
)


class FeatureEngineerV2:
    """Clinically grounded feature engineering for ICU prediction tasks."""

    TARGET_COLS = [
        'mortality', 'los_days', 'los_category',
        'icu_readmit_48h', 'icu_readmit_7d', 'discharge_disposition',
        'need_vent_any', 'need_vasopressor_any', 'need_rrt_any',
        'aki_onset', 'ards_onset', 'liver_injury_onset', 'sepsis_onset',
    ]

    def __init__(self, cohort_path='data/processed/cohort_labeled.csv', window_hours=FEATURE_WINDOW_HOURS_V2):
        self.cohort_path = Path(cohort_path)
        self.window_hours = int(window_hours)
        self.cohort = None
        self.features = None
        self.hosp_dir = MIMIC_IV_HOSP
        self.icu_dir = MIMIC_IV_ICU

    def load_cohort(self):
        print("\n" + "=" * 70)
        print("LOADING MASTER COHORT (V2)")
        print("=" * 70)
        self.cohort = pd.read_csv(
            self.cohort_path,
            parse_dates=['admittime', 'dischtime', 'intime', 'outtime']
        )
        base_cols = ['subject_id', 'hadm_id', 'stay_id']
        target_cols = [c for c in self.TARGET_COLS if c in self.cohort.columns]
        self.features = self.cohort[base_cols + target_cols].copy()
        print(f"[OK] Loaded {len(self.cohort):,} stays with {len(target_cols)} pre-existing targets")
        return self.cohort

    def _time_filter(self, df, time_col, intime_col='intime'):
        hours = (df[time_col] - df[intime_col]).dt.total_seconds() / 3600
        return df[(hours >= 0) & (hours <= self.window_hours)].copy()

    def extract_demographic_features(self):
        print("\n" + "=" * 70)
        print("EXTRACTING DEMOGRAPHIC FEATURES (V2)")
        print("=" * 70)
        cols = ['age', 'gender', 'ethnicity', 'insurance', 'admission_type', 'first_careunit']
        available = [c for c in cols if c in self.cohort.columns]
        demo = self.cohort[available].copy()

        if 'age' in demo.columns:
            self.features['age'] = demo['age']
        if 'gender' in demo.columns:
            self.features['gender_M'] = (demo['gender'] == 'M').astype(int)
        if 'ethnicity' in demo.columns:
            ethnicity = demo['ethnicity'].fillna('UNKNOWN')
            self.features['ethnicity_WHITE'] = ethnicity.str.contains('WHITE', case=False, na=False).astype(int)
            self.features['ethnicity_BLACK'] = ethnicity.str.contains('BLACK', case=False, na=False).astype(int)
            self.features['ethnicity_HISPANIC'] = ethnicity.str.contains('HISPANIC', case=False, na=False).astype(int)
            self.features['ethnicity_ASIAN'] = ethnicity.str.contains('ASIAN', case=False, na=False).astype(int)
            self.features['ethnicity_OTHER'] = (
                ~ethnicity.str.contains('WHITE|BLACK|HISPANIC|ASIAN', case=False, na=False)
            ).astype(int)
        if 'insurance' in demo.columns:
            self.features = pd.concat(
                [self.features, pd.get_dummies(demo['insurance'].fillna('Unknown'), prefix='insurance')],
                axis=1
            )
        if 'admission_type' in demo.columns:
            self.features = pd.concat(
                [self.features, pd.get_dummies(demo['admission_type'].fillna('Unknown'), prefix='admission')],
                axis=1
            )
        if 'first_careunit' in demo.columns:
            self.features = pd.concat(
                [self.features, pd.get_dummies(demo['first_careunit'].fillna('Unknown'), prefix='icu')],
                axis=1
            )
        print(f"[OK] Demographic feature count now: {self.features.shape[1]}")
        return self.features

    def extract_temporal_features(self):
        """
        Retain legacy temporal admission features that showed useful IG signal
        while remaining leakage-safe (known at admission).
        """
        print("\n" + "=" * 70)
        print("EXTRACTING TEMPORAL FEATURES (IG-RETAINED)")
        print("=" * 70)
        if 'admittime' not in self.cohort.columns:
            print("[WARNING] admittime not found, skipping temporal features")
            return self.features

        admittime = pd.to_datetime(self.cohort['admittime'])
        self.features['admit_hour'] = admittime.dt.hour
        self.features['admit_is_night'] = (
            (admittime.dt.hour >= 22) | (admittime.dt.hour <= 6)
        ).astype(int)
        self.features['admit_day_of_week'] = admittime.dt.dayofweek
        self.features['admit_is_weekend'] = (admittime.dt.dayofweek >= 5).astype(int)
        self.features['admit_month'] = admittime.dt.month
        print("[OK] Added admit_hour, admit_is_night, admit_day_of_week, admit_is_weekend, admit_month")
        return self.features

    def extract_comorbidity_features(self):
        print("\n" + "=" * 70)
        print("EXTRACTING CURATED COMORBIDITY FEATURES (V2)")
        print("=" * 70)
        diagnoses = pd.read_csv(
            self.hosp_dir / 'diagnoses_icd.csv.gz',
            compression='gzip',
            usecols=['hadm_id', 'icd_code']
        )
        hadm_set = set(self.cohort['hadm_id'].unique())
        diagnoses = diagnoses[diagnoses['hadm_id'].isin(hadm_set)].copy()
        diagnoses['icd_code'] = diagnoses['icd_code'].astype(str).str.upper().str.strip()

        comorb = self.cohort[['hadm_id']].copy()
        for name, prefixes in COMORBIDITY_PREFIXES_V2.items():
            mask = diagnoses['icd_code'].str.startswith(tuple(prefixes), na=False)
            hadm_pos = set(diagnoses.loc[mask, 'hadm_id'].unique())
            comorb[f'comorb_{name}'] = comorb['hadm_id'].isin(hadm_pos).astype(int)

        self.features = pd.concat([self.features, comorb.drop(columns=['hadm_id'])], axis=1)
        print(f"[OK] Added {len(COMORBIDITY_PREFIXES_V2)} curated comorbidity flags")
        return self.features

    def extract_vital_features(self):
        print("\n" + "=" * 70)
        print(f"EXTRACTING VITAL FEATURES (FIRST {self.window_hours}H)")
        print("=" * 70)
        cohort_keys = self.cohort[['stay_id', 'intime']].drop_duplicates()
        stay_set = set(cohort_keys['stay_id'].astype('int64'))
        item_map = {}
        for vital_name, itemids in VITAL_ITEMIDS_V2.items():
            for itemid in itemids:
                item_map[itemid] = vital_name
        vital_itemids = list(item_map.keys())

        chunks = []
        for i, chunk in enumerate(pd.read_csv(
            self.icu_dir / 'chartevents.csv.gz',
            compression='gzip',
            usecols=['stay_id', 'itemid', 'charttime', 'valuenum'],
            parse_dates=['charttime'],
            chunksize=2_000_000
        )):
            if (i + 1) % 50 == 0:
                print(f"  -> chartevents chunk {i+1}")
            chunk = chunk[
                chunk['stay_id'].isin(stay_set)
                & chunk['itemid'].isin(vital_itemids)
                & chunk['valuenum'].notna()
            ]
            if chunk.empty:
                continue
            chunk = chunk.merge(cohort_keys, on='stay_id', how='inner')
            chunk = self._time_filter(chunk, 'charttime')
            if chunk.empty:
                continue
            chunk['vital_name'] = chunk['itemid'].map(item_map)
            # Normalize Fahrenheit charting rows into Celsius for a single temperature feature family.
            temp_mask = (chunk['vital_name'] == 'temperature_c') & (chunk['valuenum'] > 80)
            if temp_mask.any():
                chunk.loc[temp_mask, 'valuenum'] = (chunk.loc[temp_mask, 'valuenum'] - 32.0) * (5.0 / 9.0)
            chunks.append(chunk[['stay_id', 'vital_name', 'valuenum']])

        if not chunks:
            print("[WARNING] No vital records found in extraction window")
            return self.features

        vitals = pd.concat(chunks, ignore_index=True)
        agg = (
            vitals.groupby(['stay_id', 'vital_name'])['valuenum']
            .agg(['min', 'max', 'mean'])
            .reset_index()
        )
        wide = agg.pivot(index='stay_id', columns='vital_name')
        wide.columns = [f'{vital}_{stat}_24h' for stat, vital in wide.columns]
        wide = wide.reset_index()

        merged = self.features[['stay_id']].merge(wide, on='stay_id', how='left').drop(columns=['stay_id'])
        self.features = pd.concat([self.features, merged], axis=1)
        print(f"[OK] Added {merged.shape[1]} vital aggregate features")
        return self.features

    def extract_lab_features(self):
        print("\n" + "=" * 70)
        print(f"EXTRACTING LAB FEATURES (FIRST {self.window_hours}H)")
        print("=" * 70)
        cohort_keys = self.cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].drop_duplicates()
        lab_itemids = sorted({item for ids in LAB_ITEMIDS_V2.values() for item in ids})
        item_to_name = {}
        for name, ids in LAB_ITEMIDS_V2.items():
            for item in ids:
                item_to_name[item] = name

        chunks = []
        for i, chunk in enumerate(pd.read_csv(
            self.hosp_dir / 'labevents.csv.gz',
            compression='gzip',
            usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
            parse_dates=['charttime'],
            chunksize=1_500_000
        )):
            if (i + 1) % 25 == 0:
                print(f"  -> labevents chunk {i+1}")
            chunk = chunk[chunk['itemid'].isin(lab_itemids) & chunk['valuenum'].notna()]
            if chunk.empty:
                continue
            chunk = chunk.merge(cohort_keys, on=['subject_id', 'hadm_id'], how='inner')
            if chunk.empty:
                continue
            chunk = self._time_filter(chunk, 'charttime')
            if chunk.empty:
                continue
            chunk['lab_name'] = chunk['itemid'].map(item_to_name)
            chunks.append(chunk[['stay_id', 'lab_name', 'valuenum']])

        if not chunks:
            print("[WARNING] No lab records found in extraction window")
            return self.features

        labs = pd.concat(chunks, ignore_index=True)
        agg = (
            labs.groupby(['stay_id', 'lab_name'])['valuenum']
            .agg(['min', 'max'])
            .reset_index()
        )
        wide = agg.pivot(index='stay_id', columns='lab_name')
        wide.columns = [f'{lab}_{stat}_24h' for stat, lab in wide.columns]
        wide = wide.reset_index()

        merged = self.features[['stay_id']].merge(wide, on='stay_id', how='left').drop(columns=['stay_id'])
        self.features = pd.concat([self.features, merged], axis=1)
        print(f"[OK] Added {merged.shape[1]} lab aggregate features")
        return self.features

    def extract_urine_output_feature(self):
        print("\n" + "=" * 70)
        print(f"EXTRACTING URINE OUTPUT FEATURE (FIRST {self.window_hours}H)")
        print("=" * 70)
        cohort_keys = self.cohort[['stay_id', 'intime']].drop_duplicates()
        stay_set = set(cohort_keys['stay_id'].astype('int64'))

        try:
            outputevents = pd.read_csv(
                self.icu_dir / 'outputevents.csv.gz',
                compression='gzip',
                usecols=['stay_id', 'itemid', 'charttime', 'value'],
                parse_dates=['charttime']
            )
        except Exception as exc:
            print(f"[WARNING] Could not load outputevents.csv.gz: {exc}")
            self.features['urine_output_total_24h'] = np.nan
            return self.features

        outputevents = outputevents[
            outputevents['stay_id'].isin(stay_set)
            & outputevents['itemid'].isin(URINE_OUTPUT_ITEMIDS_V2)
            & outputevents['value'].notna()
        ]
        if outputevents.empty:
            self.features['urine_output_total_24h'] = np.nan
            return self.features

        outputevents = outputevents.merge(cohort_keys, on='stay_id', how='inner')
        outputevents = self._time_filter(outputevents, 'charttime')
        urine_totals = outputevents.groupby('stay_id')['value'].sum().reset_index()
        urine_totals = urine_totals.rename(columns={'value': 'urine_output_total_24h'})

        merged = self.features[['stay_id']].merge(urine_totals, on='stay_id', how='left')
        self.features['urine_output_total_24h'] = merged['urine_output_total_24h']
        print("[OK] Added urine_output_total_24h")
        return self.features

    def extract_treatment_features_24h(self):
        print("\n" + "=" * 70)
        print(f"EXTRACTING FIRST {self.window_hours}H TREATMENT FLAGS")
        print("=" * 70)
        cohort_keys = self.cohort[['stay_id', 'intime']].drop_duplicates()
        stay_set = set(cohort_keys['stay_id'].astype('int64'))

        intime_map = dict(zip(cohort_keys['stay_id'], cohort_keys['intime']))

        # Ventilation exposure in first 24h (chunked to avoid high memory usage)
        vent_flags = pd.Series(False, index=self.features.index)
        try:
            vent_stays = set()
            for i, chunk in enumerate(pd.read_csv(
                self.icu_dir / 'chartevents.csv.gz',
                compression='gzip',
                usecols=['stay_id', 'itemid', 'charttime'],
                parse_dates=['charttime'],
                chunksize=2_000_000
            )):
                if (i + 1) % 50 == 0:
                    print(f"  -> treatment vent chunk {i+1}")
                chunk = chunk[
                    chunk['stay_id'].isin(stay_set)
                    & chunk['itemid'].isin(VENT_ITEMIDS)
                ]
                if chunk.empty:
                    continue
                chunk['intime'] = chunk['stay_id'].map(intime_map)
                chunk = self._time_filter(chunk, 'charttime')
                if chunk.empty:
                    continue
                vent_stays.update(chunk['stay_id'].unique().tolist())
            vent_flags = self.features['stay_id'].isin(vent_stays)
        except Exception as exc:
            print(f"[WARNING] Vent 24h predictor extraction failed: {exc}")
        self.features['ventilation_24h_flag'] = vent_flags.astype(int)

        # Vasopressor exposure in first 24h (chunked)
        vaso_flags = pd.Series(False, index=self.features.index)
        try:
            vaso_stays = set()
            for i, chunk in enumerate(pd.read_csv(
                self.icu_dir / 'inputevents.csv.gz',
                compression='gzip',
                usecols=['stay_id', 'itemid', 'starttime', 'amount'],
                parse_dates=['starttime'],
                chunksize=1_000_000
            )):
                if (i + 1) % 20 == 0:
                    print(f"  -> treatment vaso chunk {i+1}")
                chunk = chunk[
                    chunk['stay_id'].isin(stay_set)
                    & chunk['itemid'].isin(VASOPRESSOR_ITEMIDS)
                    & (chunk['amount'] > 0)
                ]
                if chunk.empty:
                    continue
                chunk['intime'] = chunk['stay_id'].map(intime_map)
                chunk = self._time_filter(chunk, 'starttime')
                if chunk.empty:
                    continue
                vaso_stays.update(chunk['stay_id'].unique().tolist())
            vaso_flags = self.features['stay_id'].isin(vaso_stays)
        except Exception as exc:
            print(f"[WARNING] Vasopressor 24h predictor extraction failed: {exc}")
        self.features['vasopressor_24h_flag'] = vaso_flags.astype(int)

        # RRT exposure in first 24h (chunked)
        rrt_flags = pd.Series(False, index=self.features.index)
        try:
            rrt_stays = set()
            for i, chunk in enumerate(pd.read_csv(
                self.icu_dir / 'procedureevents.csv.gz',
                compression='gzip',
                usecols=['stay_id', 'itemid', 'starttime'],
                parse_dates=['starttime'],
                chunksize=750_000
            )):
                if (i + 1) % 20 == 0:
                    print(f"  -> treatment rrt chunk {i+1}")
                chunk = chunk[
                    chunk['stay_id'].isin(stay_set)
                    & chunk['itemid'].isin(RRT_ITEMIDS)
                ]
                if chunk.empty:
                    continue
                chunk['intime'] = chunk['stay_id'].map(intime_map)
                chunk = self._time_filter(chunk, 'starttime')
                if chunk.empty:
                    continue
                rrt_stays.update(chunk['stay_id'].unique().tolist())
            rrt_flags = self.features['stay_id'].isin(rrt_stays)
        except Exception as exc:
            print(f"[WARNING] RRT 24h predictor extraction failed: {exc}")
        self.features['rrt_24h_flag'] = rrt_flags.astype(int)

        print("[OK] Added ventilation_24h_flag, vasopressor_24h_flag, rrt_24h_flag")
        return self.features

    def extract_sofa_proxy_features(self):
        print("\n" + "=" * 70)
        print("EXTRACTING SOFA-PROXY FEATURES")
        print("=" * 70)
        proxy_map = {
            'sofa_proxy_renal': 'creatinine_max_24h',
            'sofa_proxy_coag': 'platelets_min_24h',
            'sofa_proxy_liver': 'bilirubin_max_24h',
            'sofa_proxy_cv': 'map_min_24h',
            'sofa_proxy_resp': 'spo2_min_24h',
        }
        for proxy_col, source_col in proxy_map.items():
            self.features[proxy_col] = self.features[source_col] if source_col in self.features.columns else np.nan
        print(f"[OK] Added {len(proxy_map)} SOFA-proxy columns")
        return self.features

    def extract_ventilation_targets(self):
        print("\n" + "=" * 70)
        print("EXTRACTING VENTILATION TARGETS")
        print("=" * 70)
        vent_stays = set()
        cohort_stays = set(self.features['stay_id'].dropna().astype('int64'))
        try:
            for chunk in pd.read_csv(
                self.icu_dir / 'chartevents.csv.gz', compression='gzip',
                usecols=['stay_id', 'itemid'], dtype={'stay_id': 'int64', 'itemid': 'int32'},
                chunksize=2_000_000
            ):
                subset = chunk[
                    (chunk['itemid'].isin(VENT_ITEMIDS)) &
                    (chunk['stay_id'].isin(cohort_stays))
                ]
                vent_stays.update(subset['stay_id'].unique().tolist())
            self.features['need_vent_any'] = self.features['stay_id'].isin(vent_stays).astype(int)
        except Exception as exc:
            print(f"[WARNING] Could not load chartevents for vent target: {exc}")
            self.features['need_vent_any'] = 0
        print(f"[OK] need_vent_any prevalence: {self.features['need_vent_any'].mean()*100:.1f}%")
        return self.features

    def extract_vasopressor_targets(self):
        print("\n" + "=" * 70)
        print("EXTRACTING VASOPRESSOR TARGETS")
        print("=" * 70)
        try:
            inputevents = pd.read_csv(
                self.icu_dir / 'inputevents.csv.gz', compression='gzip',
                usecols=['stay_id', 'itemid', 'amount'], dtype={'itemid': 'int32'}
            )
            vaso_stays = inputevents[
                (inputevents['itemid'].isin(VASOPRESSOR_ITEMIDS)) &
                (inputevents['amount'] > 0)
            ]['stay_id'].unique()
            self.features['need_vasopressor_any'] = self.features['stay_id'].isin(vaso_stays).astype(int)
        except Exception as exc:
            print(f"[WARNING] Could not load inputevents for vasopressor target: {exc}")
            self.features['need_vasopressor_any'] = 0
        print(f"[OK] need_vasopressor_any prevalence: {self.features['need_vasopressor_any'].mean()*100:.1f}%")
        return self.features

    def extract_rrt_targets(self):
        print("\n" + "=" * 70)
        print("EXTRACTING RRT TARGETS")
        print("=" * 70)
        rrt_stays = set()
        try:
            proc_events = pd.read_csv(
                self.icu_dir / 'procedureevents.csv.gz', compression='gzip',
                usecols=['stay_id', 'itemid'], dtype={'itemid': 'int32'}
            )
            rrt_stays.update(proc_events[proc_events['itemid'].isin(RRT_ITEMIDS)]['stay_id'].unique())
        except Exception:
            pass
        try:
            proc_icd = pd.read_csv(self.hosp_dir / 'procedures_icd.csv.gz', compression='gzip')
            cohort_hadm = self.cohort[['hadm_id', 'stay_id']].drop_duplicates()
            proc_icd = proc_icd.merge(cohort_hadm, on='hadm_id', how='inner')
            dialysis_codes = proc_icd[proc_icd['icd_code'].astype(str).str.contains('5A1D', na=False)]['stay_id'].unique()
            rrt_stays.update(dialysis_codes)
        except Exception:
            pass
        self.features['need_rrt_any'] = self.features['stay_id'].isin(rrt_stays).astype(int)
        print(f"[OK] need_rrt_any prevalence: {self.features['need_rrt_any'].mean()*100:.1f}%")
        return self.features

    def extract_clinical_targets(self):
        print("\n" + "=" * 70)
        print("EXTRACTING CLINICAL ONSET TARGETS")
        print("=" * 70)
        try:
            from src.features.clinical_targets import (
                compute_aki_labels,
                compute_ards_labels,
                compute_liver_injury_labels,
                compute_sepsis_labels,
            )
        except ImportError:
            from features.clinical_targets import (
                compute_aki_labels,
                compute_ards_labels,
                compute_liver_injury_labels,
                compute_sepsis_labels,
            )
        self.features = compute_aki_labels(self.features, self.cohort, self.hosp_dir)
        self.features = compute_ards_labels(self.features, self.cohort, self.hosp_dir, self.icu_dir)
        self.features = compute_liver_injury_labels(self.features, self.cohort, self.hosp_dir)
        self.features = compute_sepsis_labels(self.features, self.cohort, self.hosp_dir, self.icu_dir)
        return self.features

    def run_quality_checks(self):
        print("\n" + "=" * 70)
        print("V2 QUALITY & LEAKAGE CHECKS")
        print("=" * 70)
        id_and_targets = {'subject_id', 'hadm_id', 'stay_id'} | set(self.TARGET_COLS)
        predictor_cols = [c for c in self.features.columns if c not in id_and_targets]
        print(f"[OK] Predictor count: {len(predictor_cols)}")
        print(f"[OK] Predictor target band check (60-90): {'PASS' if 60 <= len(predictor_cols) <= 90 else 'WARN'}")

        missing = self.features[predictor_cols].isna().mean().sort_values(ascending=False).head(10)
        print("\nTop-10 missingness (predictors):")
        for col, val in missing.items():
            print(f"  - {col}: {val*100:.1f}% missing")

        binary_cols = [c for c in predictor_cols if self.features[c].dropna().isin([0, 1]).all()]
        print("\nTop-10 prevalence (binary predictors):")
        for col in binary_cols[:10]:
            prevalence = self.features[col].mean()
            print(f"  - {col}: {prevalence*100:.1f}% positive")

        leaked_family = [c for c in self.features.columns if c.startswith('diag_') or c.startswith('proc_')]
        print(f"\n[OK] Top-N diag/proc family present: {'YES (WARN)' if leaked_family else 'NO'}")
        if leaked_family:
            print(f"  [WARN] Found legacy columns: {leaked_family[:10]}")
        print(f"[OK] Predictor extraction window: first {self.window_hours}h from ICU intime")
        self._report_ig_relevance_retention()
        return self.features

    def _report_ig_relevance_retention(self, ig_path='all_targets_ig_matrix.csv'):
        """
        Optional audit: check which top IG legacy features are retained in v2.
        """
        try:
            ig_df = pd.read_csv(ig_path)
        except Exception:
            return
        ig_cols = [c for c in ig_df.columns if c.endswith('_IG')]
        if not ig_cols:
            return
        ig_df['max_ig'] = ig_df[ig_cols].max(axis=1)
        top_legacy = ig_df.sort_values('max_ig', ascending=False).head(30)['Feature_Name'].tolist()
        retained = [f for f in top_legacy if f in self.features.columns]
        print(f"[OK] Retained {len(retained)}/30 of highest-IG legacy features (non-leaky overlap)")
        if retained:
            print(f"  -> Examples: {retained[:8]}")

    def build_feature_matrix(self):
        print("\n" + "=" * 70)
        print("MIMIC-IV FEATURE ENGINEERING PIPELINE (V2)")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.load_cohort()
        self.extract_demographic_features()
        self.extract_temporal_features()
        self.extract_comorbidity_features()
        self.extract_vital_features()
        self.extract_lab_features()
        self.extract_urine_output_feature()
        self.extract_treatment_features_24h()
        self.extract_sofa_proxy_features()

        # Targets are computed after predictor extraction by design.
        self.extract_ventilation_targets()
        self.extract_vasopressor_targets()
        self.extract_rrt_targets()
        self.extract_clinical_targets()
        self.run_quality_checks()

        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING V2 COMPLETE")
        print("=" * 70)
        print(f"  Final matrix shape: {self.features.shape}")
        print(f"  ICU stays: {len(self.features):,}")
        return self.features

    def save_features(self, output_path='data/processed/features_engineered_v2.csv'):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n>> Saving v2 features to: {output_path}")
        self.features.to_csv(output_path, index=False)
        print(f"[OK] Saved: {output_path.absolute()}")


def main():
    engineer = FeatureEngineerV2(cohort_path='data/processed/cohort_labeled.csv')
    engineer.build_feature_matrix()
    engineer.save_features(output_path='data/processed/features_engineered_v2.csv')
    print("\n*** Feature engineering v2 complete.")


if __name__ == "__main__":
    main()
