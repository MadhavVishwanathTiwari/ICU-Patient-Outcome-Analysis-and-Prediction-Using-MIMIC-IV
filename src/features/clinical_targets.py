"""
Clinical Onset Target Computation
==================================
Computes binary labels for complication onset during ICU stay:
  - AKI  (KDIGO creatinine criteria)
  - ARDS (Berlin definition P/F ratio)
  - Acute Liver Injury (AST / ALT / bilirubin >3× ULN)
  - Sepsis (Sepsis-3: suspected infection + SOFA delta >= 2)

All functions receive the running features DataFrame and return it
with the new target column appended.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    CREATININE_ITEMID, PAO2_ITEMID, FIO2_ITEMID,
    ALT_ITEMID, AST_ITEMID, BILIRUBIN_ITEMID,
    ULN_ALT, ULN_AST, ULN_BILIRUBIN
)


def _load_labevents_for_items(hosp_dir: Path, item_ids: list,
                              cohort: pd.DataFrame) -> pd.DataFrame:
    """Load labevents filtered to specific itemids and cohort (chunked for low-RAM)."""
    cohort_ids = cohort[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime']].drop_duplicates()
    chunks = []
    try:
        chunk_size = 1_500_000
        for i, chunk in enumerate(pd.read_csv(
            hosp_dir / 'labevents.csv.gz', compression='gzip',
            usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
            parse_dates=['charttime'], chunksize=chunk_size
        )):
            if (i + 1) % 20 == 0:
                print(f"    labevents chunk {i+1}...")
            chunk = chunk[chunk['itemid'].isin(item_ids)].dropna(subset=['valuenum'])
            chunk = chunk.merge(cohort_ids, on=['subject_id', 'hadm_id'], how='inner')
            if not chunk.empty:
                chunks.append(chunk)
    except Exception as e:
        print(f"  [WARNING] Could not load labevents: {e}")
        return pd.DataFrame()
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------
# AKI — KDIGO creatinine criteria
# ---------------------------------------------------------------
def compute_aki_labels(features: pd.DataFrame, cohort: pd.DataFrame,
                       hosp_dir: Path) -> pd.DataFrame:
    print("  -> Computing AKI labels (KDIGO creatinine)...")
    labs = _load_labevents_for_items(hosp_dir, [CREATININE_ITEMID], cohort)
    if labs.empty:
        features['aki_onset'] = 0
        return features

    labs['hours'] = (
        (labs['charttime'] - labs['intime']).dt.total_seconds() / 3600
    )
    labs = labs[(labs['hours'] >= 0) & (labs['hours'] <= 168)]

    aki_stay_ids = set()
    for stay_id, grp in labs.groupby('stay_id'):
        grp = grp.sort_values('charttime')
        baseline_vals = grp[grp['hours'] <= 6]['valuenum']
        if baseline_vals.empty:
            continue
        baseline = baseline_vals.iloc[0]
        subsequent = grp[grp['hours'] > 6]['valuenum']
        if subsequent.empty:
            continue
        # KDIGO: absolute increase >= 0.3 in 48h OR 1.5× baseline in 7d
        if (subsequent.max() >= baseline + 0.3) or (subsequent.max() >= baseline * 1.5):
            aki_stay_ids.add(stay_id)

    features['aki_onset'] = features['stay_id'].isin(aki_stay_ids).astype(int)
    print(f"  [OK] AKI rate: {features['aki_onset'].mean()*100:.1f}%")
    return features


# ---------------------------------------------------------------
# ARDS — Berlin definition (P/F ratio <= 300 while on ventilation)
# ---------------------------------------------------------------
def compute_ards_labels(features: pd.DataFrame, cohort: pd.DataFrame,
                        hosp_dir: Path, icu_dir: Path) -> pd.DataFrame:
    print("  -> Computing ARDS labels (P/F ratio)...")
    labs = _load_labevents_for_items(hosp_dir, [PAO2_ITEMID], cohort)
    if labs.empty:
        features['ards_onset'] = 0
        return features

    # Get FiO2 from chartevents (chunked for low-RAM)
    cohort_stays = set(features['stay_id'].astype('int64'))
    fio2_chunks = []
    try:
        for i, chunk in enumerate(pd.read_csv(
            icu_dir / 'chartevents.csv.gz', compression='gzip',
            usecols=['stay_id', 'itemid', 'charttime', 'valuenum'],
            parse_dates=['charttime'], chunksize=2_000_000
        )):
            if (i + 1) % 50 == 0:
                print(f"    chartevents (FiO2) chunk {i+1}...")
            chunk = chunk[(chunk['itemid'] == FIO2_ITEMID) & (chunk['stay_id'].isin(cohort_stays))]
            if not chunk.empty:
                chunk = chunk.rename(columns={'valuenum': 'fio2'})[['stay_id', 'charttime', 'fio2']]
                chunk.loc[chunk['fio2'] > 1, 'fio2'] /= 100
                chunk = chunk[chunk['fio2'] > 0.2]
                fio2_chunks.append(chunk)
        fio2 = pd.concat(fio2_chunks, ignore_index=True) if fio2_chunks else pd.DataFrame()
    except Exception as e:
        print(f"  [WARNING] Could not load chartevents for FiO2: {e}")
        features['ards_onset'] = 0
        return features
    if fio2.empty:
        features['ards_onset'] = 0
        return features

    pao2 = labs[['stay_id', 'charttime', 'valuenum']].rename(columns={'valuenum': 'pao2'})

    # Merge nearest FiO2 within ±2h window per stay using merge_asof
    pao2 = pao2.sort_values('charttime')
    fio2 = fio2.sort_values('charttime')

    ards_stays = set()
    for stay_id in pao2['stay_id'].unique():
        p = pao2[pao2['stay_id'] == stay_id].copy()
        f = fio2[fio2['stay_id'] == stay_id].copy()
        if f.empty or p.empty:
            continue
        merged = pd.merge_asof(
            p.sort_values('charttime'), f[['charttime', 'fio2']].sort_values('charttime'),
            on='charttime', direction='nearest', tolerance=pd.Timedelta('2h')
        )
        merged = merged.dropna(subset=['fio2'])
        if merged.empty:
            continue
        merged['pf_ratio'] = merged['pao2'] / merged['fio2']
        if (merged['pf_ratio'] <= 300).any():
            ards_stays.add(stay_id)

    features['ards_onset'] = features['stay_id'].isin(ards_stays).astype(int)
    print(f"  [OK] ARDS rate: {features['ards_onset'].mean()*100:.1f}%")
    return features


# ---------------------------------------------------------------
# Acute Liver Injury — AST/ALT/bilirubin > 3× ULN
# ---------------------------------------------------------------
def compute_liver_injury_labels(features: pd.DataFrame, cohort: pd.DataFrame,
                                hosp_dir: Path) -> pd.DataFrame:
    print("  -> Computing Liver Injury labels (>3x ULN)...")
    labs = _load_labevents_for_items(
        hosp_dir, [ALT_ITEMID, AST_ITEMID, BILIRUBIN_ITEMID], cohort
    )
    if labs.empty:
        features['liver_injury_onset'] = 0
        return features

    thresholds = {
        ALT_ITEMID: ULN_ALT * 3,
        AST_ITEMID: ULN_AST * 3,
        BILIRUBIN_ITEMID: ULN_BILIRUBIN * 3,
    }
    liver_stays = set()
    for (stay_id, itemid), grp in labs.groupby(['stay_id', 'itemid']):
        if grp['valuenum'].max() >= thresholds.get(itemid, float('inf')):
            liver_stays.add(stay_id)

    features['liver_injury_onset'] = features['stay_id'].isin(liver_stays).astype(int)
    print(f"  [OK] Liver injury rate: {features['liver_injury_onset'].mean()*100:.1f}%")
    return features


# ---------------------------------------------------------------
# Sepsis — Sepsis-3 (suspected infection + SOFA delta >= 2)
# ---------------------------------------------------------------
def compute_sepsis_labels(features: pd.DataFrame, cohort: pd.DataFrame,
                          hosp_dir: Path, icu_dir: Path) -> pd.DataFrame:
    print("  -> Computing Sepsis labels (Sepsis-3)...")
    try:
        from src.features.sofa_calculator import compute_sofa_deltas
    except ImportError:
        try:
            from features.sofa_calculator import compute_sofa_deltas
        except ImportError:
            print("  [WARNING] sofa_calculator not found, skipping sepsis")
            features['sepsis_onset'] = 0
            return features

    # Step 1: suspected infection = culture + antibiotic within 24h
    suspected = _suspected_infection(cohort, hosp_dir)

    # Step 2: SOFA delta >= 2
    sofa_deltas = compute_sofa_deltas(cohort, hosp_dir, icu_dir)

    sepsis_stays = suspected & sofa_deltas
    features['sepsis_onset'] = features['stay_id'].isin(sepsis_stays).astype(int)
    print(f"  [OK] Sepsis rate: {features['sepsis_onset'].mean()*100:.1f}%")
    return features


def _suspected_infection(cohort: pd.DataFrame, hosp_dir: Path) -> set:
    """Return set of stay_ids with culture + antibiotic within 24h."""
    culture_stays = set()
    abx_stays = set()

    # Microbiology cultures
    try:
        micro = pd.read_csv(
            hosp_dir / 'microbiologyevents.csv.gz', compression='gzip',
            usecols=['subject_id', 'hadm_id', 'charttime'],
            parse_dates=['charttime']
        )
        cohort_ids = cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].drop_duplicates()
        micro = micro.merge(cohort_ids, on=['subject_id', 'hadm_id'], how='inner')
        micro['hours'] = (micro['charttime'] - micro['intime']).dt.total_seconds() / 3600
        culture_stays = set(micro[(micro['hours'] >= -24) & (micro['hours'] <= 72)]['stay_id'].unique())
    except Exception:
        pass

    # Antibiotic prescriptions
    try:
        prescriptions = pd.read_csv(
            hosp_dir / 'prescriptions.csv.gz', compression='gzip',
            usecols=['subject_id', 'hadm_id', 'drug', 'starttime'],
            parse_dates=['starttime']
        )
        abx_keywords = ['CILLIN', 'MYCIN', 'FLOXACIN', 'CYCLINE', 'AZOLE',
                         'MEROPENEM', 'VANCOMYCIN', 'CEFTRI', 'PIPERACILLIN',
                         'METRONIDAZOLE', 'CEFEPIME', 'AZITHROMYCIN',
                         'LEVOFLOXACIN', 'CIPROFLOXACIN', 'AMPICILLIN']
        pattern = '|'.join(abx_keywords)
        prescriptions = prescriptions[
            prescriptions['drug'].str.upper().str.contains(pattern, na=False)
        ]
        cohort_ids = cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].drop_duplicates()
        prescriptions = prescriptions.merge(cohort_ids, on=['subject_id', 'hadm_id'], how='inner')
        prescriptions['hours'] = (
            (prescriptions['starttime'] - prescriptions['intime']).dt.total_seconds() / 3600
        )
        abx_stays = set(
            prescriptions[(prescriptions['hours'] >= -24) & (prescriptions['hours'] <= 72)]['stay_id'].unique()
        )
    except Exception:
        pass

    return culture_stays & abx_stays
