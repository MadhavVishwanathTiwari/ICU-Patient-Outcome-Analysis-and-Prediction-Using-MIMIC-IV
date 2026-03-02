"""
SOFA Score Calculator
=====================
Computes the Sequential Organ Failure Assessment (SOFA) score
from MIMIC-IV chartevents and labevents data.

Six components (0-4 each, max total = 24):
  1. Respiration  — PaO2/FiO2 ratio
  2. Coagulation  — Platelets
  3. Liver        — Bilirubin
  4. Cardiovascular — MAP + vasopressors
  5. CNS          — GCS
  6. Renal        — Creatinine

Used by clinical_targets.compute_sepsis_labels() to determine
SOFA delta >= 2 for Sepsis-3 definition.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    PAO2_ITEMID, FIO2_ITEMID, PLATELETS_ITEMID, BILIRUBIN_ITEMID,
    MAP_ITEMID, GCS_ITEMID, CREATININE_ITEMID, VASOPRESSOR_ITEMIDS
)


def _score_resp(pf: float) -> int:
    if pf < 100: return 4
    if pf < 200: return 3
    if pf < 300: return 2
    if pf < 400: return 1
    return 0

def _score_coag(plt: float) -> int:
    if plt < 20:  return 4
    if plt < 50:  return 3
    if plt < 100: return 2
    if plt < 150: return 1
    return 0

def _score_liver(bili: float) -> int:
    if bili >= 12: return 4
    if bili >= 6:  return 3
    if bili >= 2:  return 2
    if bili >= 1.2: return 1
    return 0

def _score_cv(map_val: float, on_vasopressor: bool) -> int:
    if on_vasopressor:
        return 3
    if map_val < 70:
        return 1
    return 0

def _score_cns(gcs: float) -> int:
    if gcs < 6:  return 4
    if gcs < 10: return 3
    if gcs < 13: return 2
    if gcs < 15: return 1
    return 0

def _score_renal(cr: float) -> int:
    if cr >= 5.0: return 4
    if cr >= 3.5: return 3
    if cr >= 2.0: return 2
    if cr >= 1.2: return 1
    return 0


def compute_sofa_deltas(cohort: pd.DataFrame, hosp_dir: Path,
                        icu_dir: Path) -> set:
    """Return set of stay_ids where SOFA increased by >= 2 (0-48h vs baseline). Chunked for low-RAM."""
    cohort_ids = cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].drop_duplicates()
    cohort_stays = set(cohort_ids['stay_id'].astype('int64'))
    lab_items = [CREATININE_ITEMID, PLATELETS_ITEMID, BILIRUBIN_ITEMID, PAO2_ITEMID]
    chart_items = [MAP_ITEMID, GCS_ITEMID, FIO2_ITEMID]

    lab_chunks = []
    try:
        for i, chunk in enumerate(pd.read_csv(
            hosp_dir / 'labevents.csv.gz', compression='gzip',
            usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
            parse_dates=['charttime'], chunksize=1_500_000
        )):
            if (i + 1) % 20 == 0:
                print(f"    sofa labevents chunk {i+1}...")
            chunk = chunk[chunk['itemid'].isin(lab_items)].merge(cohort_ids, on=['subject_id', 'hadm_id'], how='inner')
            if not chunk.empty:
                chunk['hours'] = (chunk['charttime'] - chunk['intime']).dt.total_seconds() / 3600
                chunk = chunk[(chunk['hours'] >= -6) & (chunk['hours'] <= 72)]
                lab_chunks.append(chunk)
        labs = pd.concat(lab_chunks, ignore_index=True) if lab_chunks else pd.DataFrame()
    except Exception as e:
        print(f"  [WARNING] SOFA labs: {e}")
        labs = pd.DataFrame()

    chart_chunks = []
    try:
        for i, chunk in enumerate(pd.read_csv(
            icu_dir / 'chartevents.csv.gz', compression='gzip',
            usecols=['stay_id', 'itemid', 'charttime', 'valuenum'],
            parse_dates=['charttime'], chunksize=2_000_000
        )):
            if (i + 1) % 50 == 0:
                print(f"    sofa chartevents chunk {i+1}...")
            chunk = chunk[(chunk['itemid'].isin(chart_items)) & (chunk['stay_id'].isin(cohort_stays))]
            if not chunk.empty:
                chunk = chunk.merge(cohort_ids[['stay_id', 'intime']].drop_duplicates(), on='stay_id', how='inner')
                chunk['hours'] = (chunk['charttime'] - chunk['intime']).dt.total_seconds() / 3600
                chunk = chunk[(chunk['hours'] >= -6) & (chunk['hours'] <= 72)]
                chart_chunks.append(chunk)
        chart = pd.concat(chart_chunks, ignore_index=True) if chart_chunks else pd.DataFrame()
    except Exception as e:
        print(f"  [WARNING] SOFA chart: {e}")
        chart = pd.DataFrame()

    vaso_inputs = pd.DataFrame()
    try:
        inputs = pd.read_csv(
            icu_dir / 'inputevents.csv.gz', compression='gzip',
            usecols=['stay_id', 'itemid', 'starttime', 'amount'],
            parse_dates=['starttime']
        )
        vaso_inputs = inputs[
            (inputs['itemid'].isin(VASOPRESSOR_ITEMIDS)) & (inputs['amount'] > 0)
        ]
        vaso_inputs = vaso_inputs.merge(
            cohort_ids[['stay_id', 'intime']].drop_duplicates(), on='stay_id', how='inner'
        )
        vaso_inputs['hours'] = (
            (vaso_inputs['starttime'] - vaso_inputs['intime']).dt.total_seconds() / 3600
        )
    except Exception:
        pass

    sofa_delta_stays = set()

    for stay_id in cohort['stay_id'].unique():
        s_labs = labs[labs['stay_id'] == stay_id]
        s_chart = chart[chart['stay_id'] == stay_id] if not chart.empty else pd.DataFrame()
        s_vaso = vaso_inputs[vaso_inputs['stay_id'] == stay_id] if not vaso_inputs.empty else pd.DataFrame()

        baseline_sofa = _window_sofa(s_labs, s_chart, s_vaso, -6, 6)
        peak_sofa = _window_sofa(s_labs, s_chart, s_vaso, 6, 72)

        if peak_sofa - baseline_sofa >= 2:
            sofa_delta_stays.add(stay_id)

    return sofa_delta_stays


def _window_sofa(labs: pd.DataFrame, chart: pd.DataFrame,
                 vaso: pd.DataFrame, h_start: float, h_end: float) -> int:
    """Compute worst-case SOFA score in a time window."""
    w_labs = labs[(labs['hours'] >= h_start) & (labs['hours'] <= h_end)]
    w_chart = chart[(chart['hours'] >= h_start) & (chart['hours'] <= h_end)] if not chart.empty else pd.DataFrame()
    w_vaso = vaso[(vaso['hours'] >= h_start) & (vaso['hours'] <= h_end)] if not vaso.empty else pd.DataFrame()

    # Respiration: PaO2/FiO2
    resp_score = 0
    pao2_vals = w_labs[w_labs['itemid'] == PAO2_ITEMID]['valuenum']
    fio2_vals = w_chart[w_chart['itemid'] == FIO2_ITEMID]['valuenum'] if not w_chart.empty else pd.Series(dtype=float)
    if not pao2_vals.empty and not fio2_vals.empty:
        fio2_val = fio2_vals.median()
        if fio2_val > 1:
            fio2_val /= 100
        if fio2_val > 0:
            pf = pao2_vals.min() / fio2_val
            resp_score = _score_resp(pf)

    # Coagulation: Platelets (worst = lowest)
    coag_score = 0
    plt_vals = w_labs[w_labs['itemid'] == PLATELETS_ITEMID]['valuenum']
    if not plt_vals.empty:
        coag_score = _score_coag(plt_vals.min())

    # Liver: Bilirubin (worst = highest)
    liver_score = 0
    bili_vals = w_labs[w_labs['itemid'] == BILIRUBIN_ITEMID]['valuenum']
    if not bili_vals.empty:
        liver_score = _score_liver(bili_vals.max())

    # CV: MAP + vasopressors
    cv_score = 0
    on_vaso = not w_vaso.empty
    map_vals = w_chart[w_chart['itemid'] == MAP_ITEMID]['valuenum'] if not w_chart.empty else pd.Series(dtype=float)
    map_val = map_vals.min() if not map_vals.empty else 80
    cv_score = _score_cv(map_val, on_vaso)

    # CNS: GCS (worst = lowest)
    cns_score = 0
    gcs_vals = w_chart[w_chart['itemid'] == GCS_ITEMID]['valuenum'] if not w_chart.empty else pd.Series(dtype=float)
    if not gcs_vals.empty:
        cns_score = _score_cns(gcs_vals.min())

    # Renal: Creatinine (worst = highest)
    renal_score = 0
    cr_vals = w_labs[w_labs['itemid'] == CREATININE_ITEMID]['valuenum']
    if not cr_vals.empty:
        renal_score = _score_renal(cr_vals.max())

    return resp_score + coag_score + liver_score + cv_score + cns_score + renal_score
