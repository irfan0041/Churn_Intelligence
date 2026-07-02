import streamlit as st
import pandas as pd
import io
import os
from components.ui_helpers import render_section_header, open_panel, close_panel
from utils.model_utils import load_historical_datasets, load_ml_assets, preprocess_customer, get_shap_values_for_customer, FEATURE_COLS
from utils.business_utils import segment_customers, get_recommendations_for_customer
from utils.pdf_utils import generate_customer_pdf

def render_reports_view():
    """Renders the reports page containing batch and individual downloads."""
    df_raw, df_proc = load_historical_datasets()
    model, _, _ = load_ml_assets()
    
    render_section_header("Reporting & Data Exports", "Download comprehensive business analysis logs, spreadsheets, and individual customer PDF report cards.")
    
    # ----------------------------------------------------
    # Section 1: Batch Export
    # ----------------------------------------------------
    st.markdown(open_panel(label="Batch Export", title="Batch Database Export"), unsafe_allow_html=True)
    st.write("Calculate churn risks for the entire active database and download as structured spreadsheets.")
    
    if st.button("Generate Scored Database Table", type="primary"):
        with st.spinner("Processing batch telemetry, calculating probabilities..."):
            df_scored = segment_customers(df_raw, df_proc, model)
            
            # Show preview
            st.markdown('<div class="panel-card__title">Scored Table Preview</div>', unsafe_allow_html=True)
            st.dataframe(df_scored[['customer_id', 'churn_probability', 'risk_tier', 'retention_priority', 'total_revenue', 'customer_segment']].head(6))
            
            # Export CSV
            csv_data = df_scored.to_csv(index=False).encode('utf-8')
            
            # Export Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_scored.to_excel(writer, index=False, sheet_name='Scored Customers')
            excel_data = excel_buffer.getvalue()
            
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    label="📥 Download CSV Spreadsheet",
                    data=csv_data,
                    file_name="explainchurn_scored_database.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with exp_col2:
                st.download_button(
                    label="📥 Download Excel Spreadsheet (XLSX)",
                    data=excel_data,
                    file_name="explainchurn_scored_database.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    st.markdown(close_panel(), unsafe_allow_html=True)

    st.markdown(open_panel(label="PDF Reports", title="Client-Ready PDF Exporter"), unsafe_allow_html=True)
    st.write("Generate and download a corporate PDF report card including demographic details, risk ratings, SHAP factors, and a personalized retention campaign roadmap.")
    
    # Dropdown to pick customer
    customer_ids = df_raw['customer_id'].tolist()
    pdf_cust_id = st.selectbox(
        "Select Customer ID for PDF Exporter",
        options=["-- Select Customer --"] + customer_ids,
        key="pdf_selector"
    )
    
    if pdf_cust_id != "-- Select Customer --":
        cust_row = df_raw[df_raw['customer_id'] == pdf_cust_id].iloc[0].to_dict()
        
        if st.button("Generate PDF Report", key="generate_pdf_btn", type="primary"):
            with st.spinner(f"Compiling PDF report card for {pdf_cust_id}..."):
                # Preprocess, Predict, SHAP
                features_df = preprocess_customer(cust_row)
                
                # Predict
                prob = float(model.predict_proba(features_df)[0, 1])
                risk_lvl = "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
                
                # SHAP
                shap_values, _ = get_shap_values_for_customer(features_df)
                
                # Recommendations
                recoms = get_recommendations_for_customer(cust_row, shap_values, FEATURE_COLS)
                
                # Generate PDF Bytes
                pdf_bytes = generate_customer_pdf(
                    pdf_cust_id,
                    cust_row,
                    prob,
                    risk_lvl,
                    recoms,
                    shap_values,
                    FEATURE_COLS
                )
                
                st.success(f"PDF Report generated successfully for {pdf_cust_id}!")
                st.download_button(
                    label=f"📥 Download Report Card ({pdf_cust_id}.pdf)",
                    data=pdf_bytes,
                    file_name=f"explainchurn_report_{pdf_cust_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    st.markdown(close_panel(), unsafe_allow_html=True)
    history_path = os.path.join('data', 'prediction_history.csv')
    if os.path.exists(history_path):
        st.markdown(open_panel(label="History", title="Simulation History Exporter"), unsafe_allow_html=True)
        st.write("Download the logs of all customer simulations performed in this session.")
        
        try:
            df_hist = pd.read_csv(history_path)
            st.write(f"Telemetry log contains **{len(df_hist)}** records.")
            
            csv_hist = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Simulation History Log (CSV)",
                data=csv_hist,
                file_name="explainchurn_simulation_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            pass
        st.markdown(close_panel(), unsafe_allow_html=True)
