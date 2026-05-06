"""
MIMIC-IV ICU Outcome Prediction Dashboard  —  v2  (Post-Tournament, Post-Tuning)
==================================================================================
Loads tuned models from models/tuned/, SHAP images from results/shap/,
and ROC curves from results/roc_curves_tuned/.

Pages
-----
  Dashboard          — cohort statistics and outcome rates
  Predictions        — per-patient risk panel across all 12 targets
  Model Performance  — tournament baseline vs tuned AUC comparison
  SHAP Analysis      — beeswarm / bar / waterfall image viewer
  ROC Curves         — tuned ROC PNG gallery
  About

Setup
-----
  1. python save_tuned_models.py          (one-time, ~15 min)
  2. python shap_analysis.py              (one-time, ~30 min)
  3. streamlit run app.py
"""

import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path
# Ensure repo root is on sys.path so pickled artifacts referencing `src.*` can import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

def _joblib_load_with_torch_cpu_map(path):
    """
    Some saved artifacts (stacking incl. PyTorch FTT) may have been serialized on CUDA.
    Patch torch.load during unpickle so CUDA tensors map to CPU at load time.
    """
    try:
        import joblib as _joblib
        import torch as _torch

        orig = _torch.load

        def patched(*args, **kwargs):
            kwargs.setdefault("map_location", _torch.device("cpu"))
            return orig(*args, **kwargs)

        _torch.load = patched
        try:
            return _joblib.load(path)
        finally:
            _torch.load = orig
    except Exception:
        # fall back to plain joblib (will raise the original error if incompatible)
        return joblib.load(path)

def _ensure_sklearn_imputer_compat(imputer):
    """
    Patch-only: scikit-learn model persistence across versions can miss internal attrs.
    We add the minimal missing attrs needed for transform() in newer versions.
    """
    try:
        import numpy as _np
        # sklearn>=1.7 SimpleImputer.transform may expect _fill_dtype; older pickles may only have _fit_dtype
        if hasattr(imputer, "statistics_") and not hasattr(imputer, "_fill_dtype"):
            fit_dtype = getattr(imputer, "_fit_dtype", None)
            if fit_dtype is not None:
                imputer._fill_dtype = fit_dtype
            else:
                imputer._fill_dtype = _np.dtype("float64")
        return imputer
    except Exception:
        return imputer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Outcome Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path('models/tuned')
PARAMS_PATH   = Path('results/best_hyperparams.json')
SHAP_DIR      = Path('results/shap')
ROC_DIR       = Path('results/roc_curves_tuned')
DATA_DIR      = Path('data/processed/tournament')
FEATURES_PATH = Path('data/processed/features_engineered_v2.csv')

MATRIX_FILES = {
    'IG':    'X_ig_union.csv',
    'ANOVA': 'X_anova_union.csv',
    'MI':    'X_mi_union.csv',
    'LASSO': 'X_lasso_union.csv',
}

# Exact tuning results (from tune_winners.py output)
TUNING_RESULTS = [
    {'Target':'mortality',            'Model':'CatBoost',            'Matrix':'LASSO','Baseline':0.8898,'Tuned':0.8966,'Delta':+0.0068},
    {'Target':'aki_onset',            'Model':'Custom MLP',          'Matrix':'LASSO','Baseline':0.8187,'Tuned':0.8191,'Delta':+0.0004},
    {'Target':'sepsis_onset',         'Model':'Random Forest',       'Matrix':'MI',   'Baseline':0.7805,'Tuned':0.7841,'Delta':+0.0036},
    {'Target':'ards_onset',           'Model':'CatBoost',            'Matrix':'MI',   'Baseline':0.9351,'Tuned':0.9373,'Delta':+0.0022},
    {'Target':'liver_injury_onset',   'Model':'CatBoost',            'Matrix':'IG',   'Baseline':0.9232,'Tuned':0.9290,'Delta':+0.0058},
    {'Target':'need_vent_any',        'Model':'Custom MLP',          'Matrix':'ANOVA','Baseline':0.9786,'Tuned':0.9791,'Delta':+0.0005},
    {'Target':'need_vasopressor_any', 'Model':'Custom MLP',          'Matrix':'LASSO','Baseline':0.9738,'Tuned':0.9739,'Delta':+0.0001},
    {'Target':'need_rrt_any',         'Model':'XGBoost',             'Matrix':'LASSO','Baseline':0.9509,'Tuned':0.9581,'Delta':+0.0072},
    {'Target':'icu_readmit_48h',      'Model':'Logistic Regression', 'Matrix':'ANOVA','Baseline':0.5948,'Tuned':0.5919,'Delta':-0.0029},
    {'Target':'icu_readmit_7d',       'Model':'Logistic Regression', 'Matrix':'ANOVA','Baseline':0.6015,'Tuned':0.6031,'Delta':+0.0016},
    {'Target':'los_category',         'Model':'Custom MLP',          'Matrix':'LASSO','Baseline':0.7666,'Tuned':0.7681,'Delta':+0.0015},
    {'Target':'discharge_disposition','Model':'XGBoost',             'Matrix':'LASSO','Baseline':0.8135,'Tuned':0.8239,'Delta':+0.0104},
]

TASK_TYPE = {
    'mortality':'binary','aki_onset':'binary','sepsis_onset':'binary',
    'ards_onset':'binary','liver_injury_onset':'binary','need_vent_any':'binary',
    'need_vasopressor_any':'binary','need_rrt_any':'binary',
    'icu_readmit_48h':'binary','icu_readmit_7d':'binary',
    'los_category':'multiclass','discharge_disposition':'multiclass',
}

TARGET_LABELS = {
    'mortality':            'Mortality',
    'aki_onset':            'AKI Onset',
    'sepsis_onset':         'Sepsis Onset',
    'ards_onset':           'ARDS Onset',
    'liver_injury_onset':   'Liver Injury',
    'need_vent_any':        'Ventilation',
    'need_vasopressor_any': 'Vasopressors',
    'need_rrt_any':         'RRT',
    'icu_readmit_48h':      'Readmit 48h',
    'icu_readmit_7d':       'Readmit 7d',
    'los_category':         'LOS Category',
    'discharge_disposition':'Discharge Disp.',
}

ALL_TARGET_COLS = list(TASK_TYPE.keys()) + ['los_days']
ID_COLS = ['subject_id','hadm_id','stay_id']


# ── Resource loading ─────────────────────────────────────────────────────────

@st.cache_resource
def load_all_models():
    """Load every tuned model + its prep pipeline from models/tuned/."""
    if not PARAMS_PATH.exists():
        return {}, {}

    with open(PARAMS_PATH) as f:
        params = json.load(f)

    models = {}
    preps  = {}

    for target in params:
        target_cfg = params.get(target) or {}
        target_model = target_cfg.get("model")

        prep_path = MODELS_DIR / f'{target}_prep.pkl'
        if not prep_path.exists():
            continue
        preps[target] = joblib.load(prep_path)

        keras_path    = MODELS_DIR / f'{target}_keras.keras'
        keras_dir     = MODELS_DIR / f'{target}_keras'
        mlp_path      = MODELS_DIR / f'{target}_mlp.keras'
        stacking_path = MODELS_DIR / f'{target}_stacking.pkl'
        pkl_path      = MODELS_DIR / f'{target}.pkl'

        # Decide which artifact to load from params (source of truth).
        try:
            if target_model == "Stacking Ensemble" and stacking_path.exists():
                models[target] = _joblib_load_with_torch_cpu_map(stacking_path)
            elif target_model == "Custom MLP" and (mlp_path.exists() or keras_path.exists() or keras_dir.exists()):
                import tensorflow as tf
                src_path = mlp_path if mlp_path.exists() else (keras_path if keras_path.exists() else keras_dir)
                models[target] = tf.keras.models.load_model(str(src_path))
            else:
                # Non-neural sklearn/xgb/cb models (prefer explicit <target>.pkl; else <target>_<safe>.pkl)
                if not pkl_path.exists() and isinstance(target_model, str):
                    safe = target_model.lower().replace(" ", "_")
                    candidate = MODELS_DIR / f"{target}_{safe}.pkl"
                    if candidate.exists():
                        pkl_path = candidate
                if pkl_path.exists():
                    models[target] = joblib.load(pkl_path)
                # else: skip silently
        except Exception as e:
            # Keep loading other targets if one fails.
            pass

    return models, preps


@st.cache_data
def load_cohort():
    """Load the raw feature CSV for dashboard stats and patient selection."""
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(FEATURES_PATH)
    df.columns = [c.replace('[','').replace(']','').replace('<','lt').replace('>','gt')
                  for c in df.columns]
    return df


@st.cache_data
def load_matrix(matrix_name: str):
    """Load one of the four feature-selection matrices."""
    path = DATA_DIR / MATRIX_FILES[matrix_name]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.replace('[','').replace(']','').replace('<','lt').replace('>','gt')
                  for c in df.columns]
    return df


def predict_for_patient(target: str, patient_raw: pd.Series,
                        models: dict, preps: dict):
    """
    Given a raw patient row from the cohort CSV, run it through the
    correct imputer+scaler for that target's matrix, then return prob.
    """
    if target not in models or target not in preps:
        return None, None

    prep      = preps[target]

    if not isinstance(prep, dict):
        return None, None

    # Backward/forward compatibility with saved prep dict formats
    feat_cols = prep.get('feature_cols') or prep.get('features')
    imputer   = prep.get('imputer')
    scaler    = prep.get('scaler')
    task_type = prep.get('task_type') or TASK_TYPE.get(target, 'binary')

    if not isinstance(feat_cols, (list, tuple)) or imputer is None or scaler is None:
        return None, None
    model     = models[target]

    imputer = _ensure_sklearn_imputer_compat(imputer)

    # Build feature vector (fill missing cols with NaN)
    x = np.array([patient_raw.get(c, np.nan) for c in feat_cols]).reshape(1, -1)

    x = imputer.transform(x)
    x = scaler.transform(x)
    # Ensure numeric dtype that TF/Keras reliably accepts
    try:
        x = np.asarray(x, dtype=np.float32)
    except Exception:
        pass
    try:
        # Stacking artifact (dict with meta + trained_bases)
        if isinstance(model, dict) and ("meta" in model) and ("trained_bases" in model):
            from src.models.stacking_model import get_test_meta_features
            n_classes = int(model.get("n_classes", 2))
            stack_task = model.get("task_type") or task_type
            test_meta = get_test_meta_features(model["trained_bases"], x, stack_task, n_classes)
            proba = model["meta"].predict_proba(test_meta)[0]
            if stack_task == "binary":
                proba = np.array([1.0 - float(proba[1]), float(proba[1])], dtype=float)
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(x)[0]
        else:
            # Keras model
            raw = model.predict(x, verbose=0)
            proba = raw.flatten() if task_type == 'binary' else raw[0]
    except Exception as e:
        return None, None

    if task_type == 'binary':
        return float(proba[1]) if len(proba) > 1 else float(proba[0]), proba
    return None, proba  # multiclass: return full proba array as second value


# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0a0e1a; }
[data-testid="stSidebar"] * { color: #c8d6e5 !important; }
[data-testid="stSidebar"] .stRadio label { 
    padding: 6px 10px; border-radius: 6px; 
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08); }
.metric-card {
    background: #f7f8fc; border: 1px solid #e2e6ef;
    border-radius: 10px; padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val  { font-size: 1.7rem; font-weight: 600; color: #1a2744; }
.metric-card .lbl  { font-size: 0.8rem; color: #7a8aaa; margin-top: 2px; }
.risk-badge-high   { background:#fee2e2; color:#991b1b; border-radius:6px; padding:3px 10px; font-weight:600; }
.risk-badge-medium { background:#fef3c7; color:#92400e; border-radius:6px; padding:3px 10px; font-weight:600; }
.risk-badge-low    { background:#d1fae5; color:#065f46; border-radius:6px; padding:3px 10px; font-weight:600; }
.section-divider   { border-top: 1px solid #e2e6ef; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown("## 🏥 ICU Prediction")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", [
        "📊 Dashboard",
        "🔮 Predictions",
        "📈 Model Performance",
        "🧠 SHAP Analysis",
        "📉 ROC Curves",
        "ℹ️ About",
    ])
    st.sidebar.markdown("---")
    st.sidebar.caption("MIMIC-IV v2.2  •  Minor Project 2026")
    return page


# ── Page: Dashboard ──────────────────────────────────────────────────────────

def page_dashboard(cohort: pd.DataFrame):
    st.header("Cohort Overview")

    if cohort.empty:
        st.error(f"Feature CSV not found at `{FEATURES_PATH}`.")
        return

    # Row 1 — top-level metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total ICU stays",   f"{len(cohort):,}")
    c2.metric("Mortality rate",    f"{cohort['mortality'].mean()*100:.1f}%"
              if 'mortality' in cohort.columns else "N/A")
    c3.metric("Mean LOS (days)",   f"{cohort['los_days'].mean():.1f}"
              if 'los_days' in cohort.columns else "N/A")
    c4.metric("Unique patients",
              f"{cohort['subject_id'].nunique():,}"
              if 'subject_id' in cohort.columns else "N/A")

    # Row 2 — complication rates
    rate_cols = {
        'need_vent_any':'Ventilation','need_vasopressor_any':'Vasopressors',
        'aki_onset':'AKI','ards_onset':'ARDS',
        'sepsis_onset':'Sepsis','need_rrt_any':'RRT',
    }
    available = {k:v for k,v in rate_cols.items() if k in cohort.columns}
    if available:
        st.markdown("#### Complication rates")
        cols = st.columns(len(available))
        for col,(k,label) in zip(cols, available.items()):
            col.metric(label, f"{cohort[k].mean()*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mortality distribution")
        if 'mortality' in cohort.columns:
            vc = cohort['mortality'].value_counts()
            fig = px.pie(values=vc.values, names=['Survived','Died'],
                         color_discrete_sequence=['#22c55e','#ef4444'],
                         hole=0.4)
            fig.update_layout(margin=dict(t=20,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("LOS distribution (days)")
        if 'los_days' in cohort.columns:
            los_clipped = cohort['los_days'].clip(upper=30)
            fig = px.histogram(los_clipped, nbins=40,
                               color_discrete_sequence=['#6366f1'])
            fig.update_layout(xaxis_title="LOS (days, capped at 30)",
                              yaxis_title="Count",
                              margin=dict(t=20,b=10,l=10,r=10),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if 'age' in cohort.columns:
        st.subheader("Age distribution")
        fig = px.histogram(cohort, x='age', nbins=35,
                           color_discrete_sequence=['#0ea5e9'])
        fig.update_layout(xaxis_title="Age (years)", yaxis_title="Count",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Predictions ────────────────────────────────────────────────────────

def page_predictions(cohort: pd.DataFrame, models: dict, preps: dict):
    st.header("Patient Risk Panel")

    if cohort.empty:
        st.warning("Cohort CSV not found."); return
    if not models:
        st.warning("No tuned models found. Run `save_tuned_models.py` first."); return

    # Patient selector
    fmt = (lambda i: f"Patient {cohort.iloc[i].get('subject_id','?')} "
                     f"(Stay {cohort.iloc[i].get('stay_id','?')})")
    idx = st.selectbox("Select patient", range(len(cohort)), format_func=fmt)
    patient = cohort.iloc[idx]

    # Demographics strip
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Age",   str(int(patient.get('age', 0))) + " yrs")
    d2.metric("Gender","M" if patient.get('gender_M',0)==1 else "F")
    d3.metric("Actual mortality","Yes" if patient.get('mortality',0)==1 else "No")
    d4.metric("Actual LOS",     f"{patient.get('los_days',0):.1f} d")

    st.markdown("---")

    # ── Binary binary targets (gauges) ──────────────────────────────────────
    binary_targets = [t for t in TASK_TYPE if TASK_TYPE[t]=='binary' and t in models]

    st.subheader("Risk probabilities")

    # 4 per row
    for row_start in range(0, len(binary_targets), 4):
        chunk = binary_targets[row_start:row_start+4]
        cols  = st.columns(len(chunk))
        for col, target in zip(cols, chunk):
            prob, _ = predict_for_patient(target, patient, models, preps)
            with col:
                if prob is None:
                    st.info(f"{TARGET_LABELS[target]}\n\nN/A")
                    continue
                pct = prob * 100
                color = "#ef4444" if pct>50 else "#f59e0b" if pct>25 else "#22c55e"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(pct, 1),
                    title={'text': TARGET_LABELS[target], 'font':{'size':13}},
                    number={'suffix':'%', 'font':{'size':18}},
                    gauge={
                        'axis':{'range':[0,100],'tickwidth':0.5},
                        'bar':{'color':color,'thickness':0.25},
                        'steps':[
                            {'range':[0,25],'color':'#f0fdf4'},
                            {'range':[25,50],'color':'#fefce8'},
                            {'range':[50,100],'color':'#fef2f2'},
                        ],
                        'threshold':{'line':{'color':'#94a3b8','width':2},
                                     'thickness':0.75,'value':50}
                    }
                ))
                fig.update_layout(height=180, margin=dict(t=30,b=5,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

                # Actual label
                actual = patient.get(target)
                if actual is not None:
                    st.caption(f"Actual: **{'Yes' if int(actual)==1 else 'No'}**")

    st.markdown("---")

    # ── Multiclass targets ───────────────────────────────────────────────────
    st.subheader("Multiclass predictions")
    mc1, mc2 = st.columns(2)

    for col_ui, target in zip([mc1, mc2],
                               [t for t in TASK_TYPE if TASK_TYPE[t]=='multiclass']):
        _, proba = predict_for_patient(target, patient, models, preps)
        with col_ui:
            st.markdown(f"**{TARGET_LABELS[target]}**")
            if proba is None:
                st.info("Model not loaded."); continue

            if target == 'los_category':
                labels = ['Short (<3d)','Medium (3–7d)','Long (>7d)']
            else:
                labels = ['Home','Facility','Death']

            labels = labels[:len(proba)]
            fig = px.bar(x=labels, y=(proba*100).round(1),
                         color=labels,
                         color_discrete_sequence=['#6366f1','#f59e0b','#ef4444'])
            fig.update_layout(showlegend=False, yaxis_title="Probability (%)",
                              xaxis_title="", height=220,
                              margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

            # Actual
            actual = patient.get(target)
            if actual is not None:
                act_label = labels[int(actual)] if int(actual) < len(labels) else str(actual)
                st.caption(f"Actual: **{act_label}**")


# ── Page: Model Performance ──────────────────────────────────────────────────

def page_performance():
    st.header("Model Performance — Baseline vs Tuned")

    st.markdown("""
    **Method:** 4 feature-selection matrices (IG, ANOVA, MI, LASSO) × 5 models (CatBoost, XGBoost,
    Random Forest, Logistic Regression, Custom MLP) = 240 models evaluated per target.
    Winning configuration for each target was tuned with **Optuna Bayesian optimisation** (50 trials, TPE sampler).
    """)

    df = pd.DataFrame(TUNING_RESULTS)
    df['Target_display'] = df['Target'].str.replace('_',' ').str.title()
    df['Delta_str'] = df['Delta'].apply(lambda d: f"+{d:.4f}" if d>=0 else f"{d:.4f}")

    # ── Summary metrics ──────────────────────────────────────────────────────
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Targets improved", f"{(df['Delta']>=0).sum()} / 12")
    col2.metric("Best gain",  f"+{df['Delta'].max():.4f}",
                df.loc[df['Delta'].idxmax(),'Target_display'])
    col3.metric("Avg ΔAuC",   f"+{df['Delta'].mean():.4f}")
    col4.metric("Best tuned AUC", f"{df['Tuned'].max():.4f}",
                df.loc[df['Tuned'].idxmax(),'Target_display'])

    st.markdown("---")

    # ── Grouped bar ──────────────────────────────────────────────────────────
    st.subheader("Baseline vs Tuned AUC")
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Baseline', x=df['Target_display'],
                         y=df['Baseline'], marker_color='#94a3b8',
                         marker_line_width=0.5))
    fig.add_trace(go.Bar(name='Tuned',    x=df['Target_display'],
                         y=df['Tuned'],
                         marker_color=[('#22c55e' if d>=0 else '#ef4444')
                                       for d in df['Delta']],
                         marker_line_width=0.5))
    fig.update_layout(barmode='group', yaxis=dict(range=[0.55,1.0],title='ROC-AUC'),
                      xaxis_tickangle=-35, height=400,
                      legend=dict(orientation='h',yanchor='bottom',y=1.02),
                      margin=dict(t=20,b=80))
    st.plotly_chart(fig, use_container_width=True)

    # ── Delta bar ────────────────────────────────────────────────────────────
    st.subheader("Gain from tuning (ΔAUC)")
    fig2 = go.Figure(go.Bar(
        x=df['Target_display'], y=df['Delta'],
        marker_color=[('#22c55e' if d>=0 else '#ef4444') for d in df['Delta']],
        marker_line_width=0.5,
        text=df['Delta_str'], textposition='outside',
    ))
    fig2.update_layout(yaxis_title='ΔAUC', xaxis_tickangle=-35, height=340,
                       margin=dict(t=30,b=80))
    fig2.add_hline(y=0, line_dash='dash', line_color='#64748b')
    st.plotly_chart(fig2, use_container_width=True)

    # ── Full table ───────────────────────────────────────────────────────────
    st.subheader("Full results table")
    display_df = df[['Target_display','Model','Matrix','Baseline','Tuned','Delta_str']].copy()
    display_df.columns = ['Target','Winner model','Matrix','Baseline AUC','Tuned AUC','ΔAUC']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Readmit note ─────────────────────────────────────────────────────────
    with st.expander("⚠️  ICU readmission targets — why AUC is ~0.60"):
        st.markdown("""
        Both `icu_readmit_48h` (0.5919) and `icu_readmit_7d` (0.6031) score near chance
        despite tuning. This is **not a modelling failure** — it reflects a known clinical
        finding: unplanned ICU readmission is driven largely by post-discharge ward care
        quality and events that are not captured in ICU chart data. Multiple published
        MIMIC-IV studies report AUC 0.60–0.68 for this task. The features that predict
        readmission (nurse staffing, ward workload, care transitions) simply aren't in
        the dataset.
        """)


# ── Page: SHAP Analysis ──────────────────────────────────────────────────────

def page_shap():
    st.header("SHAP Explainability Analysis")

    st.markdown("""
    Each target's winning tuned model was explained using the appropriate SHAP explainer:
    **TreeExplainer** (CatBoost / XGBoost / Random Forest), **LinearExplainer** (Logistic Regression),
    **DeepExplainer** (Custom MLP).
    """)

    targets = [r['Target'] for r in TUNING_RESULTS]
    target  = st.selectbox(
        "Select target",
        targets,
        format_func=lambda t: TARGET_LABELS.get(t, t)
    )

    info = next(r for r in TUNING_RESULTS if r['Target']==target)
    st.caption(f"Model: **{info['Model']}**  |  Matrix: **{info['Matrix']}**  |  Tuned AUC: **{info['Tuned']:.4f}**")

    tab1, tab2, tab3 = st.tabs(["🐝 Beeswarm", "📊 Bar (Global Importance)", "🌊 Waterfall (Highest-risk patient)"])

    def show_img(path, caption):
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.info(f"Image not found: `{path}`  — run `shap_analysis.py` first.")

    with tab1:
        show_img(SHAP_DIR / f'{target}_beeswarm.png',
                 f"Beeswarm — {TARGET_LABELS.get(target,target)}: each dot = one patient. "
                 "X-axis = SHAP value (push on prediction). Color = feature value (blue=low, red=high).")
        with st.expander("How to read this"):
            st.markdown("""
            - **X > 0** → feature pushed the prediction **higher** (more risk)
            - **X < 0** → feature pushed the prediction **lower** (less risk)
            - **Color** → red = patient had a high value for that feature; blue = low value
            - Features are sorted by mean |SHAP| — most important at the top
            """)

    with tab2:
        show_img(SHAP_DIR / f'{target}_bar.png',
                 f"Global importance — mean |SHAP value| across all {2000} test patients.")
        with st.expander("How to read this"):
            st.markdown("""
            Mean |SHAP| is the average absolute contribution of each feature across the entire
            test set. Unlike permutation importance, SHAP values are model-agnostic and properly
            account for feature interactions.
            """)

    with tab3:
        show_img(SHAP_DIR / f'{target}_waterfall.png',
                 "Waterfall — single highest-risk patient. "
                 "Each bar shows how much that feature pushed the prediction up (green) or down (red) from the base value.")
        with st.expander("How to read this"):
            st.markdown("""
            - **Base value** (dashed line) = average model output across the training set
            - Each bar adds or subtracts from that base
            - **Final prediction** = base + sum of all SHAP values for this patient
            - Features are sorted by absolute contribution for this individual patient
            """)


# ── Page: ROC Curves ─────────────────────────────────────────────────────────

def page_roc():
    st.header("ROC Curves — Tuned Models")
    st.markdown("One ROC curve per target, saved after Optuna tuning on the held-out test set (20%).")

    targets = [r['Target'] for r in TUNING_RESULTS]
    cols = st.columns(3)
    for i, target in enumerate(targets):
        path = ROC_DIR / f'roc_tuned_{target}.png'
        with cols[i % 3]:
            if path.exists():
                auc_val = next(r['Tuned'] for r in TUNING_RESULTS if r['Target']==target)
                st.image(str(path),
                         caption=f"{TARGET_LABELS.get(target,target)}  AUC={auc_val:.4f}",
                         use_container_width=True)
            else:
                st.info(f"{target}: PNG not found")


# ── Page: About ──────────────────────────────────────────────────────────────

def page_about():
    st.header("About This Project")
    st.markdown("""
    ## MIMIC-IV ICU Outcome Prediction

    ### Pipeline summary

    | Stage | Description |
    |---|---|
    | **Cohort** | MIMIC-IV v2.2, ~73k ICU stays, adults only |
    | **Feature engineering** | Demographics, vitals, labs, SOFA/APACHE, procedure flags |
    | **Feature selection** | 4 methods (IG, ANOVA, MI, LASSO) × union over 12 targets |
    | **Tournament** | 240 models evaluated (4 matrices × 5 models × 12 targets) |
    | **Tuning** | Optuna Bayesian optimisation — 50 trials, TPE sampler, 3-fold CV |
    | **Explainability** | SHAP (TreeExplainer / LinearExplainer / DeepExplainer) |

    ### Prediction targets (12)
    Binary: Mortality, AKI, ARDS, Sepsis, Liver Injury, Ventilation, Vasopressors, RRT, Readmit 48h, Readmit 7d  
    Multiclass: LOS Category (3-class), Discharge Disposition (3-class)

    ### Model performance highlights
    | Target | Tuned AUC |
    |---|---|
    | Need ventilation | **0.9791** |
    | Need vasopressors | **0.9739** |
    | ARDS onset | **0.9373** |
    | Liver injury | **0.9290** |
    | Need RRT | **0.9581** |
    | Mortality | **0.8966** |

    ### Technologies
    Python 3.11 · Scikit-learn · XGBoost · CatBoost · TensorFlow/Keras  
    Optuna · SHAP · Streamlit · Plotly · Pandas · NumPy

    ---
    > **Research / educational use only.** Not validated for clinical decision-making.
    """)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    page   = sidebar()
    models, preps = load_all_models()
    cohort = load_cohort()

    if page == "📊 Dashboard":
        page_dashboard(cohort)
    elif page == "🔮 Predictions":
        page_predictions(cohort, models, preps)
    elif page == "📈 Model Performance":
        page_performance()
    elif page == "🧠 SHAP Analysis":
        page_shap()
    elif page == "📉 ROC Curves":
        page_roc()
    else:
        page_about()


if __name__ == "__main__":
    main()