"""
Per-target exclusions to avoid outcome leakage in shared tournament matrices.

First-24h treatment flags are legitimate predictors for targets like mortality,
but are near-outcome proxies when the label is need_vent / need_vaso / need_rrt
(any time during stay, including the same 0–24h window).
"""

from __future__ import annotations

import pandas as pd

ORGAN_SUPPORT_TARGETS = frozenset({
    'need_vent_any',
    'need_vasopressor_any',
    'need_rrt_any',
})

FIRST24H_TREATMENT_PROXY_COLS = frozenset({
    'ventilation_24h_flag',
    'vasopressor_24h_flag',
    'rrt_24h_flag',
})


def drop_organ_support_leaky_columns(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Drop first-24h treatment flags only when training organ-support targets."""
    if target_name not in ORGAN_SUPPORT_TARGETS:
        return df
    drop = [c for c in FIRST24H_TREATMENT_PROXY_COLS if c in df.columns]
    if not drop:
        return df
    return df.drop(columns=drop)
