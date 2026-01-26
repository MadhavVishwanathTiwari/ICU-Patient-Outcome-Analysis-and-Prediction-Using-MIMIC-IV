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

# Load models and data
@st.cache_resource
def load_models():
    """Load trained models and scaler."""
    models_dir = Path('models')
    
    models = {
        'mortality_xgb': joblib.load(models_dir / 'mortality_xgb.pkl'),
        'los_class_xgb': joblib.load(models_dir / 'los_class_xgb.pkl'),
        'scaler': joblib.load(models_dir / 'scaler.pkl')
    }
    
    # Load results
    with open(models_dir / 'results.json', 'r') as f:
        results = json.load(f)
    
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
        show_predictions(models, data)
    elif page == "📈 Model Performance":
        show_model_performance(results)
    else:
        show_about()

def show_dashboard(data, results):
    """Display main dashboard with overview statistics."""
    st.header("Dashboard Overview")
    
    # Key metrics
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
        model_roc = results['mortality']['test']['roc_auc']
        st.metric("Model ROC-AUC", f"{model_roc:.4f}")
    
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

def show_predictions(models, data):
    """Interactive prediction interface."""
    st.header("Patient Outcome Predictions")
    
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
    
    # Prepare features
    feature_cols = [col for col in data.columns if col not in ['subject_id', 'hadm_id', 'stay_id', 'mortality', 'los_days', 'los_category']]
    X = patient[feature_cols].values.reshape(1, -1)
    X_scaled = models['scaler'].transform(X)
    
    # Make predictions
    st.markdown("---")
    st.subheader("Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Mortality Risk")
        mortality_prob = models['mortality_xgb'].predict_proba(X_scaled)[0][1]
        
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
        los_pred = models['los_class_xgb'].predict(X_scaled)[0]
        los_proba = models['los_class_xgb'].predict_proba(X_scaled)[0]
        
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
    
    # Actual outcomes
    st.markdown("---")
    st.subheader("Actual Outcomes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        actual_mortality = "Yes" if patient['mortality'] == 1 else "No"
        st.success(f"**Actual Mortality:** {actual_mortality}")
    
    with col2:
        actual_los = los_labels[int(patient['los_category'])]
        st.success(f"**Actual LOS Category:** {actual_los}")

def show_model_performance(results):
    """Display model performance metrics."""
    st.header("Model Performance Evaluation")
    
    # Mortality models
    st.subheader("1. Mortality Prediction Models")
    
    mortality_val = results['mortality']['validation']
    mortality_test = results['mortality']['test']
    
    # Create comparison table
    comparison_data = []
    for model_name, metrics in mortality_val.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Best model test results
    best_model = results['mortality']['best_model']
    st.success(f"**Best Model:** {best_model.replace('_', ' ').title()}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test ROC-AUC", f"{mortality_test['roc_auc']:.4f}")
    col2.metric("Accuracy", f"{mortality_test['accuracy']:.4f}")
    col3.metric("Precision", f"{mortality_test['precision']:.4f}")
    col4.metric("Recall", f"{mortality_test['recall']:.4f}")
    
    st.markdown("---")
    
    # LOS Classification models
    st.subheader("2. Length of Stay Classification Models")
    
    los_val = results['los_classification']['validation']
    los_test = results['los_classification']['test']
    
    comparison_data = []
    for model_name, metrics in los_val.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision_macro']:.4f}",
            'Recall': f"{metrics['recall_macro']:.4f}",
            'F1-Score': f"{metrics['f1_macro']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    best_model = results['los_classification']['best_model']
    st.success(f"**Best Model:** {best_model.replace('_', ' ').title()}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", f"{los_test['accuracy']:.4f}")
    col2.metric("Precision", f"{los_test['precision_macro']:.4f}")
    col3.metric("Recall", f"{los_test['recall_macro']:.4f}")
    col4.metric("F1-Score", f"{los_test['f1_macro']:.4f}")

def show_about():
    """Display about page."""
    st.header("About This Project")
    
    st.markdown("""
    ## ICU Patient Outcome Prediction System
    
    ### Overview
    This machine learning system predicts patient outcomes in Intensive Care Units using the **MIMIC-IV v2.2 dataset**.
    
    ### Prediction Tasks
    1. **Mortality Prediction** - Binary classification to predict in-hospital mortality risk
    2. **Length of Stay Classification** - Multi-class prediction of ICU stay duration
    
    ### Models Used
    - **Logistic Regression** (Baseline)
    - **Random Forest**
    - **XGBoost** (Best performing)
    
    ### Dataset
    - **Source:** MIMIC-IV v2.2 (Medical Information Mart for Intensive Care)
    - **Size:** 73,181 ICU stays from 50,920 unique patients
    - **Features:** 112 clinical features including:
      - Demographic information (age, gender, ethnicity)
      - Diagnosis codes (ICD-9/ICD-10)
      - Procedure codes
      - Temporal patterns
    
    ### Performance
    - **Mortality Prediction:** 90.04% ROC-AUC
    - **LOS Classification:** 74.56% Accuracy
    
    ### Technologies
    - Python 3.11
    - Scikit-learn, XGBoost
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
