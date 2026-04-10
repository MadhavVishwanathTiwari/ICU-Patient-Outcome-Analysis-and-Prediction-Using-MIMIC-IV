"""
MIMIC-IV ICU Outcome Prediction Dashboard
==========================================
Interactive Streamlit dashboard for predicting ICU patient outcomes.

Features:
- Mortality risk prediction
- Length of stay classification
- Model performance visualization
- Feature importance analysis

Author: ML Healthcare Team
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="ICU Outcome Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Model prefix mapping ──────────────────────────────────────────────────────
# Maps results.json key  →  filename prefix used when saving the model
TASK_PREFIX = {
    'mortality':              'mortality',
    'los_category':           'los_class',
    'icu_readmit_48h':        'icu_readmit_48h',
    'icu_readmit_7d':         'icu_readmit_7d',
    'discharge_disposition':  'discharge_disposition',
    'need_vent_any':          'need_vent_any',
    'need_vasopressor_any':   'need_vasopressor_any',
    'need_rrt_any':           'need_rrt_any',
    'aki_onset':              'aki_onset',
    'ards_onset':             'ards_onset',
    'liver_injury_onset':     'liver_injury_onset',
    'sepsis_onset':           'sepsis_onset',
}

# Load models and data
@st.cache_resource
def load_models():
    """
    Load the best-performing trained model for every task.

    Strategy
    --------
    1. Read results.json to find which model won each task (best_model field).
    2. Construct the filename from the task prefix + best_model name and load it.
    3. Fall back gracefully if a file is missing.

    Returns
    -------
    models : dict
        Keys are task names (e.g. 'mortality'), values are loaded estimators.
        Also includes 'scaler'.
    results : dict
        Raw contents of results.json.
    """
    models_dir = Path('models')

    with open(models_dir / 'results.json', 'r') as f:
        results = json.load(f)

    models = {}

    # Scaler is always needed
    models['scaler'] = joblib.load(models_dir / 'scaler.pkl')

    for task_key, prefix in TASK_PREFIX.items():
        if task_key not in results:
            continue  # task wasn't trained

        best_model_name = results[task_key]['best_model']           # e.g. 'catboost'
        fname = f'{prefix}_{best_model_name}.pkl'                   # e.g. 'mortality_catboost.pkl'
        path  = models_dir / fname

        if path.exists():
            models[task_key] = joblib.load(path)
        else:
            st.warning(f"Best model file not found for '{task_key}': {fname}")

    return models, results


@st.cache_data
def load_data():
    """Load feature matrix and cohort data."""
    features = pd.read_csv('data/processed/features_engineered.csv')
    
    # Clean column names (same as training)
    features.columns = [
        col.replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt')
        for col in features.columns
    ]
    
    return features

# Main app
def main():
    st.title("🏥 ICU Patient Outcome Prediction System")
    st.markdown("### Powered by MIMIC-IV Dataset & Machine Learning")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "🔮 Predictions", "📈 Model Performance", "ℹ️ About"]
    )
    
    # Load resources
    try:
        models, results = load_models()
        data = load_data()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure models are trained by running: `python src/models/train_model.py`")
        return
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard(data, results)
    elif page == "🔮 Predictions":
        show_predictions(models, data, results)
    elif page == "📈 Model Performance":
        show_model_performance(results)
    else:
        show_about()

def show_dashboard(data, results):
    """Display main dashboard with overview statistics."""
    st.header("Dashboard Overview")
    
    # Key metrics — row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total ICU Stays", f"{len(data):,}")
    with col2:
        mortality_rate = data['mortality'].mean() * 100
        st.metric("Mortality Rate", f"{mortality_rate:.2f}%")
    with col3:
        mean_los = data['los_days'].mean()
        st.metric("Mean LOS", f"{mean_los:.2f} days")
    with col4:
        model_roc = results.get('mortality', {}).get('test', {}).get('roc_auc', 0)
        st.metric("Mortality ROC-AUC", f"{model_roc:.4f}")

    # Key metrics — row 2 (new targets)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        if 'icu_readmit_7d' in data.columns:
            st.metric("7d Readmit Rate", f"{data['icu_readmit_7d'].mean()*100:.1f}%")
    with col6:
        if 'need_vent_any' in data.columns:
            st.metric("Vent Rate", f"{data['need_vent_any'].mean()*100:.1f}%")
    with col7:
        if 'aki_onset' in data.columns:
            st.metric("AKI Rate", f"{data['aki_onset'].mean()*100:.1f}%")
    with col8:
        if 'sepsis_onset' in data.columns:
            st.metric("Sepsis Rate", f"{data['sepsis_onset'].mean()*100:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mortality Distribution")
        mortality_counts = data['mortality'].value_counts()
        fig = px.pie(
            values=mortality_counts.values,
            names=['Survived', 'Died'],
            color_discrete_sequence=['#00cc96', '#ef553b']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Length of Stay Distribution")
        los_data = data['los_category'].value_counts().sort_index()
        los_labels = ['Short (<3d)', 'Medium (3-7d)', 'Long (>7d)']
        fig = px.bar(
            x=los_labels,
            y=los_data.values,
            color=los_labels,
            color_discrete_sequence=['#636efa', '#ffa15a', '#ef553b']
        )
        fig.update_layout(showlegend=False, xaxis_title="Category", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution
    st.subheader("Patient Age Distribution")
    fig = px.histogram(data, x='age', nbins=30, title="Age Distribution")
    fig.update_layout(xaxis_title="Age (years)", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

def show_predictions(models, data, results):
    """Interactive prediction interface."""
    st.header("Patient Outcome Predictions")

    # Show which model is being used for each task
    with st.expander("ℹ️ Models in use (best by validation ROC-AUC)"):
        rows = []
        for task_key in TASK_PREFIX:
            if task_key in results and task_key in models:
                best = results[task_key]['best_model'].replace('_', ' ').title()
                val_auc = (
                    results[task_key]['validation']
                    .get(results[task_key]['best_model'], {})
                    .get('roc_auc') or
                    results[task_key]['validation']
                    .get(results[task_key]['best_model'], {})
                    .get('roc_auc_ovr_macro')
                )
                rows.append({
                    'Task': task_key.replace('_', ' ').title(),
                    'Best Model': best,
                    'Val ROC-AUC': f"{val_auc:.4f}" if val_auc else 'N/A'
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""
    Enter patient information to predict:
    - **Mortality risk** (probability of in-hospital death)
    - **Length of stay category** (Short, Medium, or Long)
    """)
    
    # Sample patient selector
    st.subheader("Select a Sample Patient")
    patient_idx = st.selectbox(
        "Choose a patient from the cohort:",
        options=range(len(data)),
        format_func=lambda x: f"Patient {data.iloc[x]['subject_id']} (Stay ID: {data.iloc[x]['stay_id']})"
    )
    
    patient = data.iloc[patient_idx]
    
    # Display patient info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Age:** {patient.get('age', 'N/A')}")
    with col2:
        gender = "Male" if patient.get('gender_M', 0) == 1 else "Female"
        st.info(f"**Gender:** {gender}")
    with col3:
        st.info(f"**Actual Mortality:** {'Yes' if patient['mortality'] == 1 else 'No'}")
    
    # Prepare features — exclude IDs and all target columns
    exclude = {
        'subject_id', 'hadm_id', 'stay_id',
        'mortality', 'los_days', 'los_category',
        'icu_readmit_48h', 'icu_readmit_7d', 'discharge_disposition',
        'need_vent_any', 'need_vasopressor_any', 'need_rrt_any',
        'aki_onset', 'ards_onset', 'liver_injury_onset', 'sepsis_onset'
    }
    feature_cols = [c for c in data.columns if c not in exclude]
    X = patient[feature_cols].values.reshape(1, -1)
    X_scaled = models['scaler'].transform(X)
    
    # Make predictions
    st.markdown("---")
    st.subheader("Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Mortality Risk")
        mortality_prob = models['mortality'].predict_proba(X_scaled)[0][1]
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=mortality_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk (%)"},
            delta={'reference': 11.38},  # Average mortality rate
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if mortality_prob > 0.5 else "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        risk_level = "HIGH" if mortality_prob > 0.5 else "MEDIUM" if mortality_prob > 0.25 else "LOW"
        st.markdown(f"**Risk Level:** :{'red' if risk_level=='HIGH' else 'orange' if risk_level=='MEDIUM' else 'green'}[{risk_level}]")
    
    with col2:
        st.markdown("### Length of Stay Prediction")
        los_pred = models['los_category'].predict(X_scaled)[0]
        los_proba = models['los_category'].predict_proba(X_scaled)[0]
        
        los_labels = ['Short (<3d)', 'Medium (3-7d)', 'Long (>7d)']
        predicted_category = los_labels[los_pred]
        
        # Bar chart of probabilities
        fig = px.bar(
            x=los_labels,
            y=los_proba * 100,
            color=los_labels,
            color_discrete_sequence=['#636efa', '#ffa15a', '#ef553b']
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Category",
            yaxis_title="Probability (%)",
            title="LOS Category Probabilities"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Predicted Category:** :blue[{predicted_category}]")
        st.markdown(f"**Confidence:** {los_proba[los_pred]*100:.1f}%")
    
    # -------- Readmission risk --------
    st.markdown("---")
    st.subheader("ICU Readmission Risk")
    rc1, rc2 = st.columns(2)
    for col_ui, key, label in [
        (rc1, 'icu_readmit_48h', '48-Hour'),
        (rc2, 'icu_readmit_7d', '7-Day')
    ]:
        with col_ui:
            if key in models:
                prob = models[key].predict_proba(X_scaled)[0][1]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    title={'text': f"{label} Readmit Risk (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [{'range': [0, 20], 'color': 'lightgreen'},
                                     {'range': [20, 50], 'color': 'yellow'},
                                     {'range': [50, 100], 'color': 'red'}]}
                ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"{label} readmission model not trained yet.")

    # -------- Discharge Disposition --------
    if 'discharge_disposition' in models:
        st.markdown("---")
        st.subheader("Discharge Disposition Prediction")
        dd_proba = models['discharge_disposition'].predict_proba(X_scaled)[0]
        dd_labels = ['Home', 'Facility', 'Death']
        fig = px.bar(x=dd_labels, y=dd_proba * 100,
                     color=dd_labels,
                     color_discrete_sequence=['#00cc96', '#ffa15a', '#ef553b'])
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

    # -------- Organ Support --------
    st.markdown("---")
    st.subheader("Organ Support Predictions")
    os1, os2, os3 = st.columns(3)
    for col_ui, key, label in [
        (os1, 'need_vent_any', 'Mechanical Ventilation'),
        (os2, 'need_vasopressor_any', 'Vasopressors'),
        (os3, 'need_rrt_any', 'Renal Replacement')
    ]:
        with col_ui:
            if key in models:
                prob = models[key].predict_proba(X_scaled)[0][1]
                st.metric(label, f"{prob*100:.1f}%")
                st.progress(float(min(prob, 1.0)))
            else:
                st.info(f"{label} model not trained.")

    # -------- Disease Onset --------
    st.markdown("---")
    st.subheader("Complication Onset Risk")
    dc1, dc2, dc3, dc4 = st.columns(4)
    for col_ui, key, label in [
        (dc1, 'aki_onset', 'AKI'),
        (dc2, 'ards_onset', 'ARDS'),
        (dc3, 'liver_injury_onset', 'Liver Injury'),
        (dc4, 'sepsis_onset', 'Sepsis')
    ]:
        with col_ui:
            if key in models:
                prob = models[key].predict_proba(X_scaled)[0][1]
                st.metric(label, f"{prob*100:.1f}%")
                st.progress(float(min(prob, 1.0)))
            else:
                st.info(f"{label} model N/A")

    # -------- Actual Outcomes --------
    st.markdown("---")
    st.subheader("Actual Outcomes")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.success(f"**Mortality:** {'Yes' if patient['mortality'] == 1 else 'No'}")
    with a2:
        st.success(f"**LOS:** {los_labels[int(patient['los_category'])]}")
    with a3:
        if 'icu_readmit_48h' in patient.index:
            st.info(f"**Readmit 48h:** {'Yes' if patient['icu_readmit_48h'] == 1 else 'No'}")
    with a4:
        if 'icu_readmit_7d' in patient.index:
            st.info(f"**Readmit 7d:** {'Yes' if patient['icu_readmit_7d'] == 1 else 'No'}")

    b1, b2, b3 = st.columns(3)
    with b1:
        if 'discharge_disposition' in patient.index:
            dd_map = {0: 'Home', 1: 'Facility', 2: 'Death'}
            st.info(f"**Discharge:** {dd_map.get(int(patient['discharge_disposition']), 'Unknown')}")
    with b2:
        if 'need_vent_any' in patient.index:
            st.info(f"**Ventilation:** {'Yes' if patient['need_vent_any'] == 1 else 'No'}")
    with b3:
        if 'need_vasopressor_any' in patient.index:
            st.info(f"**Vasopressors:** {'Yes' if patient['need_vasopressor_any'] == 1 else 'No'}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if 'need_rrt_any' in patient.index:
            st.info(f"**RRT:** {'Yes' if patient['need_rrt_any'] == 1 else 'No'}")
    with c2:
        if 'aki_onset' in patient.index:
            st.info(f"**AKI:** {'Yes' if patient['aki_onset'] == 1 else 'No'}")
    with c3:
        if 'ards_onset' in patient.index:
            st.info(f"**ARDS:** {'Yes' if patient['ards_onset'] == 1 else 'No'}")
    with c4:
        if 'sepsis_onset' in patient.index:
            st.info(f"**Sepsis:** {'Yes' if patient['sepsis_onset'] == 1 else 'No'}")

def show_model_performance(results):
    """Display model performance metrics."""
    st.header("Model Performance Evaluation")
    
    # ── 1. Mortality ────────────────────────────────────────────────────────
    st.subheader("1. Mortality Prediction Models")
    
    mortality_val  = results['mortality']['validation']
    mortality_test = results['mortality']['test']
    best_mortality = results['mortality']['best_model']
    
    comparison_data = []
    for model_name, metrics in mortality_val.items():
        comparison_data.append({
            'Model': ('⭐ ' if model_name == best_mortality else '') +
                     model_name.replace('_', ' ').title(),
            'ROC-AUC':  f"{metrics['roc_auc']:.4f}",
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision':f"{metrics['precision']:.4f}",
            'Recall':   f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    st.success(f"**Best Model:** {best_mortality.replace('_', ' ').title()}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test ROC-AUC", f"{mortality_test['roc_auc']:.4f}")
    col2.metric("Accuracy",     f"{mortality_test['accuracy']:.4f}")
    col3.metric("Precision",    f"{mortality_test['precision']:.4f}")
    col4.metric("Recall",       f"{mortality_test['recall']:.4f}")
    
    st.markdown("---")
    
    # ── 2. LOS Category ─────────────────────────────────────────────────────
    # NOTE: results.json key is 'los_category' (not 'los_classification')
    st.subheader("2. Length of Stay Classification Models")

    los_val  = results['los_category']['validation']
    los_test = results['los_category']['test']
    best_los = results['los_category']['best_model']
    
    comparison_data = []
    for model_name, metrics in los_val.items():
        comparison_data.append({
            'Model':      ('⭐ ' if model_name == best_los else '') +
                          model_name.replace('_', ' ').title(),
            'Accuracy':   f"{metrics['accuracy']:.4f}",
            'Precision':  f"{metrics['precision_macro']:.4f}",
            'Recall':     f"{metrics['recall_macro']:.4f}",
            'F1-Score':   f"{metrics['f1_macro']:.4f}",
            'ROC-AUC':    f"{metrics['roc_auc_ovr_macro']:.4f}",
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    st.success(f"**Best Model:** {best_los.replace('_', ' ').title()}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy",  f"{los_test['accuracy']:.4f}")
    col2.metric("Precision",      f"{los_test['precision_macro']:.4f}")
    col3.metric("Recall",         f"{los_test['recall_macro']:.4f}")
    col4.metric("F1-Score",       f"{los_test['f1_macro']:.4f}")

    # ── 3. Extended targets ─────────────────────────────────────────────────
    # Binary tasks: roc_auc, accuracy, precision, recall, f1
    # Multiclass tasks: roc_auc_ovr_macro, accuracy, precision_macro, recall_macro, f1_macro
    BINARY_METRICS = [
        ('roc_auc',   'ROC-AUC'),
        ('accuracy',  'Accuracy'),
        ('precision', 'Precision'),
        ('recall',    'Recall'),
        ('f1',        'F1-Score'),
    ]
    MULTI_METRICS = [
        ('roc_auc_ovr_macro',  'ROC-AUC'),
        ('accuracy',           'Accuracy'),
        ('precision_macro',    'Precision'),
        ('recall_macro',       'Recall'),
        ('f1_macro',           'F1-Score'),
    ]
    MULTICLASS_TASKS = {'discharge_disposition'}

    extended_targets = {
        'icu_readmit_48h':       'ICU Readmission (48h)',
        'icu_readmit_7d':        'ICU Readmission (7d)',
        'discharge_disposition': 'Discharge Disposition',
        'need_vent_any':         'Mechanical Ventilation',
        'need_vasopressor_any':  'Vasopressor Use',
        'need_rrt_any':          'Renal Replacement',
        'aki_onset':             'AKI Onset',
        'ards_onset':            'ARDS Onset',
        'liver_injury_onset':    'Liver Injury Onset',
        'sepsis_onset':          'Sepsis Onset',
    }

    shown_any = False
    for i, (key, title) in enumerate(extended_targets.items()):
        if key not in results:
            continue

        if not shown_any:
            st.markdown("---")
            st.subheader("3. Extended Target Models")
            shown_any = True

        st.markdown(f"#### {i + 3}. {title}")

        task_results = results[key]
        val_results  = task_results.get('validation', {})
        test_metrics = task_results.get('test', {})
        best         = task_results.get('best_model', '')
        metrics      = MULTI_METRICS if key in MULTICLASS_TASKS else BINARY_METRICS

        # Validation comparison table (all models)
        rows = []
        for model_name, model_metrics in val_results.items():
            row = {'Model': ('⭐ ' if model_name == best else '') +
                            model_name.replace('_', ' ').title()}
            for metric_key, metric_label in metrics:
                val = model_metrics.get(metric_key)
                row[metric_label] = f"{val:.4f}" if val is not None else 'N/A'
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.success(f"**Best Model:** {best.replace('_', ' ').title()}")

        # Test set metrics for the best model
        test_cols = st.columns(len(metrics))
        for col, (metric_key, metric_label) in zip(test_cols, metrics):
            val = test_metrics.get(metric_key)
            col.metric(f"Test {metric_label}", f"{val:.4f}" if val is not None else 'N/A')

        st.markdown("---")

def show_about():
    """Display about page."""
    st.header("About This Project")
    
    st.markdown("""
    ## ICU Patient Outcome Prediction System
    
    ### Overview
    This machine learning system predicts patient outcomes in Intensive Care Units using the **MIMIC-IV v2.2 dataset**.
    
    ### Prediction Tasks
    1. **Mortality Prediction** — In-hospital mortality risk
    2. **Length of Stay** — Classification (Short / Medium / Long) + continuous days
    3. **ICU Readmission** — 48-hour and 7-day bounce-back risk
    4. **Discharge Disposition** — Home / Facility / Death
    5. **Organ Support** — Ventilation, Vasopressors, RRT need
    6. **Disease Onset** — AKI, ARDS, Liver Injury, Sepsis
    
    ### Models Used
    - **Logistic Regression** (Baseline)
    - **Random Forest** (Ensemble)
    - **XGBoost** (Gradient Boosting)
    - **CatBoost** (Gradient Boosting)
    
    The best-performing model per task (by validation ROC-AUC) is selected automatically.
    
    ### Dataset
    - **Source:** MIMIC-IV v2.2 (Medical Information Mart for Intensive Care)
    - **Size:** ~73,000 ICU stays from ~51,000 unique patients
    - **Features:** 100+ clinical features including demographics, diagnosis/procedure codes, lab values, and temporal patterns
    
    ### Technologies
    - Python 3.11
    - Scikit-learn, XGBoost, CatBoost
    - Streamlit (Dashboard)
    - Pandas, NumPy (Data processing)
    - Plotly (Visualizations)
    
    ### Team
    ML Healthcare Team - Minor Project 2026
    
    ### Data Privacy
    All patient data is de-identified according to HIPAA standards and MIMIC-IV data use agreements.
    
    ---
    
    **Note:** This is a research/educational tool and should not be used for actual clinical decision-making without proper validation and regulatory approval.
    """)

if __name__ == "__main__":
    main()