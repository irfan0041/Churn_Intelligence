import streamlit as st
import pandas as pd
from components.ui_helpers import render_section_header, risk_badge, open_panel, close_panel, skeleton_loader
from utils.model_utils import load_historical_datasets, load_ml_assets
from utils.business_utils import segment_customers

def render_segmentation_view():
    """Renders the filterable customer segmentation grid."""
    df_raw, df_proc = load_historical_datasets()
    model, _, _ = load_ml_assets()
    
    # Run predictions & segments
    load_placeholder = st.empty()
    with load_placeholder.container():
        skeleton_loader(variant="kpi")
    with st.spinner("Analyzing segment profiles..."):
        df_scored = segment_customers(df_raw, df_proc, model)
    load_placeholder.empty()
        
    render_section_header("Customer Segmentation", "Segment cohorts based on risk boundaries. Target key retention playbooks.")
    
    # ----------------------------------------------------
    # Filters Bar
    # ----------------------------------------------------
    st.markdown(open_panel(title="Segment Cohort Filters"), unsafe_allow_html=True)
    
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
    
    with filt_col1:
        search_id = st.text_input("Search Customer ID", placeholder="e.g. CUST_00001")
        
    with filt_col2:
        risk_filter = st.multiselect("Risk Tier", ["High", "Medium", "Low"], default=["High", "Medium"])
        
    with filt_col3:
        segment_filter = st.multiselect("Segment Type", ["Individual", "SME", "Enterprise"], default=["Individual", "SME", "Enterprise"])
        
    with filt_col4:
        sort_by = st.selectbox("Sort By", ["Risk % (High to Low)", "Risk % (Low to High)", "Lifetime Value (High to Low)"])
        
    st.markdown(close_panel(), unsafe_allow_html=True)
    df_filtered = df_scored.copy()
    
    if search_id:
        df_filtered = df_filtered[df_filtered['customer_id'].str.contains(search_id, case=False)]
        
    if risk_filter:
        df_filtered = df_filtered[df_filtered['risk_tier'].isin(risk_filter)]
        
    if segment_filter:
        df_filtered = df_filtered[df_filtered['customer_segment'].isin(segment_filter)]
        
    # Apply Sorting
    if sort_by == "Risk % (High to Low)":
        df_filtered = df_filtered.sort_values(by='churn_probability', ascending=False)
    elif sort_by == "Risk % (Low to High)":
        df_filtered = df_filtered.sort_values(by='churn_probability', ascending=True)
    else:
        df_filtered = df_filtered.sort_values(by='total_revenue', ascending=False)
        
    # Render count metrics
    st.write(f"Showing **{len(df_filtered)}** of **{len(df_scored)}** matching accounts.")
    
    # ----------------------------------------------------
    # Customer Grid
    # ----------------------------------------------------
    # Paginate results to avoid lags (e.g. 12 cards per page)
    cards_per_page = 12
    num_pages = (len(df_filtered) - 1) // cards_per_page + 1
    
    if num_pages > 1:
        page_num = st.number_input("Page", min_value=1, max_value=num_pages, value=1)
    else:
        page_num = 1
        
    start_idx = (page_num - 1) * cards_per_page
    end_idx = start_idx + cards_per_page
    
    df_page = df_filtered.iloc[start_idx:end_idx]
    
    # Render cards in grids (3 columns)
    grid_cols = st.columns(3)
    
    for idx, (index, row) in enumerate(df_page.iterrows()):
        col = grid_cols[idx % 3]
        
        # Format metrics
        risk_pct = row['churn_probability'] * 100
        ltv = row['total_revenue']
        risk_lvl = row['risk_tier']
        priority = row['retention_priority']
        
        # Set action recommendation string
        if risk_lvl == 'High':
            recom_action = "Executive Care Call"
            action_border = "#EF4444"
        elif risk_lvl == 'Medium':
            recom_action = "Offer Billing Credit"
            action_border = "#F59E0B"
        else:
            recom_action = "Appreciation Check-in"
            action_border = "#22C55E"
            
        with col:
            st.markdown(f"""
            <div class="segment-card animate-fade-in" style="border-left: 4px solid {action_border};">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                        <span class="segment-card__id">{row['customer_id']}</span>
                        <span>{risk_badge(risk_lvl)}</span>
                    </div>
                    <div class="segment-card__meta-label">Segment / Contract</div>
                    <div class="segment-card__meta-value">{row['customer_segment']} · {row['contract_type']}</div>
                    <div class="segment-card__meta-label">Priority</div>
                    <div class="segment-card__meta-value" style="font-weight:600; color:{action_border};">{priority}</div>
                    <div class="segment-card__meta-label">Lifetime Value (LTV)</div>
                    <div class="segment-card__meta-value" style="font-size:1.05rem; font-weight:700;">${ltv:,.0f}</div>
                    <div class="segment-card__meta-label">Next Recommended Action</div>
                    <div class="segment-card__meta-value">{recom_action}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Place button inside standard streamlit context for redirection
            if st.button(f"Inspect Profile", key=f"inspect_{row['customer_id']}", use_container_width=True):
                st.session_state['selected_customer'] = row['customer_id']
                st.session_state['current_tab'] = "Prediction Diagnostics"
                st.rerun()
