import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from components.ui_helpers import render_section_header, open_panel, close_panel
from utils.model_utils import load_historical_datasets, load_ml_assets, FEATURE_COLS

def render_analytics_view():
    """Renders the global feature importance and metrics dashboard."""
    df_raw, df_proc = load_historical_datasets()
    model, _, _ = load_ml_assets()
    
    render_section_header("AI Analytics & Validation", "Deep-dive model diagnostics, validation curves, global SHAP feature importances, and correlation details.")
    
    # ----------------------------------------------------
    # Global Feature Importance Chart
    # ----------------------------------------------------
    st.markdown(open_panel(title="Global XGBoost Feature Importance"), unsafe_allow_html=True)
    st.write("Calculated based on the relative Gini gain across all decision trees in the ensemble model.")
    
    # Extract importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in FEATURE_COLS],
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(15) # Top 15 features
    
    fig_imp = px.bar(
        feat_imp,
        y='Feature',
        x='Importance',
        orientation='h',
        color_discrete_sequence=['#4F46E5']
    )
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_family='Outfit',
        margin=dict(l=150, r=20, t=10, b=10),
        xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
        height=350
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown(close_panel(), unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # Model Validation Curves (ROC / PR)
    # ----------------------------------------------------
    col1, col2 = st.columns(2)
    
    # Run train-test split exactly as in the notebook to calculate curves
    X = df_proc.drop(columns=['customer_id', 'churn'], errors='ignore')
    X['tenure_segment'] = X['tenure_segment'].map({
        '0-6 months': 0,
        '6-12 months': 1,
        '12-24 months': 2,
        '24+ months': 3
    }).fillna(0).astype(int)
    y = df_proc['churn']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    with col1:
        st.markdown(open_panel(title="ROC Validation Curve"), unsafe_allow_html=True)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color='#2563EB', width=3),
            name=f'XGBoost (AUC = {roc_auc:.4f})'
        ))
        # Random guess line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Random Guess'
        ))
        fig_roc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit',
            xaxis=dict(title="False Positive Rate", gridcolor='rgba(128,128,128,0.1)'),
            yaxis=dict(title="True Positive Rate", gridcolor='rgba(128,128,128,0.1)'),
            margin=dict(l=20, r=20, t=10, b=10),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown(f'<div style="font-size:0.85rem; text-align:center; color:var(--ec-muted); margin-top:8px;">Validated Area Under Curve: <b>{roc_auc:.4f}</b></div>', unsafe_allow_html=True)
        st.markdown(close_panel(), unsafe_allow_html=True)

    with col2:
        st.markdown(open_panel(title="Precision-Recall Curve"), unsafe_allow_html=True)
        
        prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(rec, prec)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec,
            mode='lines',
            line=dict(color='#8B5CF6', width=3),
            name=f'XGBoost (AUC = {pr_auc:.4f})'
        ))
        fig_pr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit',
            xaxis=dict(title="Recall", gridcolor='rgba(128,128,128,0.1)'),
            yaxis=dict(title="Precision", gridcolor='rgba(128,128,128,0.1)'),
            margin=dict(l=20, r=20, t=10, b=10),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown(f'<div style="font-size:0.85rem; text-align:center; color:var(--ec-muted); margin-top:8px;">Validated PR Area Under Curve: <b>{pr_auc:.4f}</b></div>', unsafe_allow_html=True)
        st.markdown(close_panel(), unsafe_allow_html=True)
        
    # ----------------------------------------------------
    # Correlation Heatmap
    # ----------------------------------------------------
    st.markdown(open_panel(title="Feature Correlation Heatmap"), unsafe_allow_html=True)
    st.write("Interactive heatmap showing linear relationships (Pearson r) between major numerical metrics and actual churn.")
    
    # Pick top numerical features + target
    num_cols = ['age', 'tenure_months', 'monthly_logins', 'weekly_active_days', 
                'avg_session_time', 'features_used', 'monthly_fee', 'total_revenue', 
                'payment_failures', 'support_tickets', 'csat_score', 'nps_score', 'churn']
    
    corr_matrix = df_raw[num_cols].corr()
    
    fig_heat = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1.0, zmax=1.0,
        labels=dict(color="Correlation")
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_family='Outfit',
        margin=dict(l=40, r=20, t=10, b=10),
        height=380
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown(close_panel(), unsafe_allow_html=True)
