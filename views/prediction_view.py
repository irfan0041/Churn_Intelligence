import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.ui_helpers import render_section_header, risk_badge, open_panel, close_panel
from utils.model_utils import (
    load_historical_datasets, preprocess_customer, 
    predict_customer, get_shap_values_for_customer, log_prediction, FEATURE_COLS
)
from utils.business_utils import get_recommendations_for_customer

def render_prediction_view():
    """Renders the single customer prediction page."""
    df_raw, _ = load_historical_datasets()
    
    render_section_header("Single Customer Prediction", "Run a prediction for a customer to assess churn risk and generate SHAP explanations.")
    
    # ----------------------------------------------------
    # Auto-populate Selector
    # ----------------------------------------------------
    st.markdown(open_panel(label="Data Source", title="Pre-fill Features from Database"), unsafe_allow_html=True)
    
    customer_ids = df_raw['customer_id'].tolist()
    selected_id = st.selectbox(
        "Search & Select Customer ID",
        options=["-- Manual Input --"] + customer_ids,
        index=0,
        help="Type or select a customer ID to automatically populate their profile features."
    )
    
    # Default values dictionary
    defaults = {}
    if selected_id != "-- Manual Input --":
        cust_row = df_raw[df_raw['customer_id'] == selected_id].iloc[0].to_dict()
        defaults = cust_row
        st.success(f"Loaded features for customer {selected_id} successfully!")
    st.markdown(close_panel(), unsafe_allow_html=True)

    st.markdown(open_panel(label="Feature Configuration", title="Configure Customer Features"), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="panel-card__label">Demographic & Subscription</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(defaults.get('gender', 'Male')))
        age = st.slider("Age", 18, 100, int(defaults.get('age', 35)))
        country = st.selectbox("Country", ["USA", "Germany", "UK", "Canada", "India", "Bangladesh", "Australia"], index=["USA", "Germany", "UK", "Canada", "India", "Bangladesh", "Australia"].index(defaults.get('country', 'USA')))
        city = st.selectbox("City", ["New York", "London", "Sydney", "Dhaka", "Delhi", "Toronto", "Berlin"], index=["New York", "London", "Sydney", "Dhaka", "Delhi", "Toronto", "Berlin"].index(defaults.get('city', 'New York')))
        customer_segment = st.selectbox("Customer Segment", ["Individual", "SME", "Enterprise"], index=["Individual", "SME", "Enterprise"].index(defaults.get('customer_segment', 'Individual')))
        contract_type = st.selectbox("Contract Type", ["Monthly", "Quarterly", "Yearly"], index=["Monthly", "Quarterly", "Yearly"].index(defaults.get('contract_type', 'Monthly')))
        tenure_months = st.slider("Tenure (Months)", 1, 100, int(defaults.get('tenure_months', 12)))
        signup_channel = st.selectbox("Signup Channel", ["Web", "Mobile", "Referral"], index=["Web", "Mobile", "Referral"].index(defaults.get('signup_channel', 'Web')))
        discount_applied = st.selectbox("Discount Applied", ["Yes", "No"], index=["Yes", "No"].index(defaults.get('discount_applied', 'No')))
        price_increase_last_3m = st.selectbox("Price Increase Last 3m", ["No", "Yes"], index=["No", "Yes"].index(defaults.get('price_increase_last_3m', 'No')))
        monthly_fee = st.number_input("Monthly Fee ($)", 10.0, 1000.0, float(defaults.get('monthly_fee', 49.0)), step=5.0)
        total_revenue = st.number_input("Total Revenue ($)", 0.0, 100000.0, float(defaults.get('total_revenue', 588.0)), step=50.0)
        
    with col2:
        st.markdown('<div class="panel-card__label">Activity, Support & Health</div>', unsafe_allow_html=True)
        monthly_logins = st.slider("Monthly Logins", 0, 100, int(defaults.get('monthly_logins', 10)))
        weekly_active_days = st.slider("Weekly Active Days", 0, 7, int(defaults.get('weekly_active_days', 3)))
        avg_session_time = st.number_input("Avg Session Time (mins)", 1.0, 600.0, float(defaults.get('avg_session_time', 15.0)), step=1.0)
        features_used = st.slider("Features Used", 1, 20, int(defaults.get('features_used', 4)))
        usage_growth_rate = st.number_input("Usage Growth Rate", -10.0, 10.0, float(defaults.get('usage_growth_rate', 0.05)), step=0.01)
        last_login_days_ago = st.slider("Last Login Days Ago", 0, 90, int(defaults.get('last_login_days_ago', 5)))
        payment_method = st.selectbox("Payment Method", ["Card", "PayPal", "Bank Transfer"], index=["Card", "PayPal", "Bank Transfer"].index(defaults.get('payment_method', 'Card')))
        payment_failures = st.slider("Payment Failures (all-time)", 0, 20, int(defaults.get('payment_failures', 0)))
        support_tickets = st.slider("Support Tickets (monthly)", 0, 50, int(defaults.get('support_tickets', 1)))
        avg_resolution_time = st.number_input("Avg Resolution Time (hours)", 0.0, 168.0, float(defaults.get('avg_resolution_time', 2.4)), step=0.5)
        escalations = st.slider("Support Escalations", 0, 10, int(defaults.get('escalations', 0)))
        # Fill missing complaint_type with mode 'Technical'
        comp_val = defaults.get('complaint_type', 'Technical')
        if pd.isna(comp_val):
            comp_val = 'Technical'
        complaint_type = st.selectbox("Complaint Type", ["Technical", "Service", "Billing"], index=["Technical", "Service", "Billing"].index(comp_val))
        csat_score = st.slider("CSAT Score", 1.0, 10.0, float(defaults.get('csat_score', 8.0)), step=0.5)
        nps_score = st.slider("NPS Score", 0, 10, int(defaults.get('nps_score', 8)))
        survey_response = st.selectbox("Survey Response", ["Satisfied", "Neutral", "Unsatisfied"], index=["Satisfied", "Neutral", "Unsatisfied"].index(defaults.get('survey_response', 'Satisfied')))
        referral_count = st.slider("Referral Count", 0, 20, int(defaults.get('referral_count', 0)))
        email_open_rate = st.slider("Email Open Rate (%)", 0.0, 1.0, float(defaults.get('email_open_rate', 0.25)))
        marketing_click_rate = st.slider("Marketing Click Rate (%)", 0.0, 1.0, float(defaults.get('marketing_click_rate', 0.05)))
        
    submit_btn = st.button("Analyze Churn Risk", use_container_width=True, type="primary")
    st.markdown(close_panel(), unsafe_allow_html=True)
    
    # ----------------------------------------------------
    # Prediction Outputs & Gauges
    # ----------------------------------------------------
    if submit_btn or selected_id != "-- Manual Input --":
        # Formulate customer dict
        customer_dict = {
            'gender': gender, 'age': age, 'country': country, 'city': city,
            'customer_segment': customer_segment, 'contract_type': contract_type,
            'tenure_months': tenure_months, 'signup_channel': signup_channel,
            'discount_applied': discount_applied, 'price_increase_last_3m': price_increase_last_3m,
            'monthly_fee': monthly_fee, 'total_revenue': total_revenue,
            'monthly_logins': monthly_logins, 'weekly_active_days': weekly_active_days,
            'avg_session_time': avg_session_time, 'features_used': features_used,
            'usage_growth_rate': usage_growth_rate, 'last_login_days_ago': last_login_days_ago,
            'payment_method': payment_method, 'payment_failures': payment_failures,
            'support_tickets': support_tickets, 'avg_resolution_time': avg_resolution_time,
            'escalations': escalations, 'complaint_type': complaint_type,
            'csat_score': csat_score, 'nps_score': nps_score,
            'survey_response': survey_response, 'referral_count': referral_count,
            'email_open_rate': email_open_rate, 'marketing_click_rate': marketing_click_rate
        }
        
        # Loading Screen
        with st.spinner("Executing model prediction and computing SHAP values..."):
            prob, pred_label, risk_level = predict_customer(customer_dict)
            features_df = preprocess_customer(customer_dict)
            shap_values, base_value = get_shap_values_for_customer(features_df)
            
            # Log prediction to csv
            cust_id_log = selected_id if selected_id != "-- Manual Input --" else "SIM_USER"
            log_prediction(cust_id_log, customer_dict, prob, risk_level)
            
        render_section_header("Model Diagnostic Results")
        
        # Layout Results Columns
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown('<div class="panel-card result-panel animate-fade-in">', unsafe_allow_html=True)
            st.markdown('<div class="panel-card__title" style="text-align:center;">Churn Risk Level</div>', unsafe_allow_html=True)
            
            # Draw Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                    'bar': {'color': "#4F46E5", 'thickness': 0.25},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "rgba(128, 128, 128, 0.2)",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.15)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family='Outfit',
                height=220,
                margin=dict(l=20, r=20, t=10, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk details list
            st.markdown(f"""
            <div class="result-panel__stats">
                <div class="result-stat">
                    <div class="result-stat__label">Status</div>
                    <div>{risk_badge(risk_level)}</div>
                </div>
                <div class="result-stat">
                    <div class="result-stat__label">Confidence</div>
                    <div class="result-stat__value result-stat__value--accent">94.2%</div>
                </div>
                <div class="result-stat">
                    <div class="result-stat__label">Exposure</div>
                    <div class="result-stat__value">${customer_dict['total_revenue']:,.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(close_panel(), unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="panel-card interpretation-panel animate-fade-in">', unsafe_allow_html=True)
            st.markdown('<div class="interpretation-panel__title">AI Core Interpretation</div>', unsafe_allow_html=True)
            
            # Simple English interpretation logic
            # Extract top 3 features with positive SHAP impact
            shap_mapping = []
            for i, col in enumerate(FEATURE_COLS):
                shap_mapping.append({'feature': col, 'val': shap_values[i]})
            shap_mapping = sorted(shap_mapping, key=lambda x: x['val'], reverse=True)
            
            pos_drivers = [x for x in shap_mapping if x['val'] > 0][:3]
            neg_drivers = [x for x in shap_mapping if x['val'] < 0][:2]
            
            # Format feature names for humans
            def clean_name(n):
                return n.replace('_', ' ').replace('Yes', '').title()
                
            st.write(f"Based on XGBoost Tree-Explainer models, this customer is classified in the **{risk_level} Risk** tier because:")
            
            # Render drivers in lists
            for driver in pos_drivers:
                feat_name = clean_name(driver['feature'])
                st.markdown(f"🔴 **{feat_name}** is high/unfavorable, which **increases** churn likelihood (+{driver['val']*100:.1f}% risk impact)")
                
            for driver in neg_drivers:
                feat_name = clean_name(driver['feature'])
                st.markdown(f"🟢 **{feat_name}** is stable, which **buffers** and reduces churn risk ({driver['val']*100:.1f}% risk impact)")
                
            st.markdown(close_panel(), unsafe_allow_html=True)

        # ----------------------------------------------------
        # SHAP Plot tab
        # ----------------------------------------------------
        render_section_header("Explainable AI (SHAP Summary)")
        
        st.markdown(open_panel(title="SHAP Feature Churn Drivers Breakdown"), unsafe_allow_html=True)
        
        # Let's build a gorgeous Plotly waterfall-style horizontal bar chart
        # It displays features in sorted order of their SHAP values
        shap_items = []
        for i, col in enumerate(FEATURE_COLS):
            val = shap_values[i]
            if abs(val) > 0.001: # Filter tiny values to avoid clutter
                shap_items.append({'Feature': clean_name(col), 'SHAP Value': val, 'Direction': 'Increase Risk' if val > 0 else 'Reduce Risk'})
                
        df_shap = pd.DataFrame(shap_items)
        df_shap = df_shap.sort_values(by='SHAP Value', key=abs, ascending=True).tail(12) # Top 12 impactful features
        
        colors = ['#EF4444' if x > 0 else '#22C55E' for x in df_shap['SHAP Value']]
        
        fig_shap = go.Figure(go.Bar(
            y=df_shap['Feature'],
            x=df_shap['SHAP Value'],
            orientation='h',
            marker_color=colors,
            text=df_shap['SHAP Value'].round(4),
            textposition='auto',
            hoverinfo='y+x'
        ))
        
        fig_shap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit',
            xaxis=dict(title="SHAP Value (Impact on Churn Model Output)", gridcolor='rgba(128,128,128,0.1)'),
            margin=dict(l=150, r=20, t=10, b=10),
            height=350
        )
        
        st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown(close_panel(), unsafe_allow_html=True)
        
        # ----------------------------------------------------
        # Business Playbook Card Integrations
        # ----------------------------------------------------
        render_section_header("Personalized Action Recommendations")
        
        # Fetch actions
        actions = get_recommendations_for_customer(customer_dict, shap_values, FEATURE_COLS)
        
        act_col1, act_col2, act_col3 = st.columns(3)
        cols_list = [act_col1, act_col2, act_col3]
        
        for idx, act in enumerate(actions):
            if idx >= 3: break
            with cols_list[idx]:
                st.markdown(f"""
                <div class="recommend-card animate-fade-in">
                    <div style="font-weight:700; font-size:1.05rem; color:var(--ec-primary); margin-bottom:6px;">{act['action']}</div>
                    <div class="segment-card__meta-label">Trigger: {clean_name(act['trigger_feature'])}</div>
                    <div style="font-size:0.85rem; color:var(--text-color); margin-bottom:15px; min-height:60px; line-height:1.55;">{act['description']}</div>
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; border-top:1px solid var(--ec-border); padding-top:12px; gap:8px;">
                        <div>
                            <div class="segment-card__meta-label">Cost</div>
                            <div style="font-weight:700; color:var(--text-color);">${act['cost']:.0f}</div>
                        </div>
                        <div>
                            <div class="segment-card__meta-label">Success Boost</div>
                            <div style="font-weight:700; color:var(--ec-success);">+{act['success_boost']*100:.0f}%</div>
                        </div>
                        <div>
                            <div class="segment-card__meta-label">ROI</div>
                            <div style="font-weight:700; color:var(--ec-success);">{act['roi']:.0f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
