import os
import streamlit as st
from components.ui_helpers import (
    inject_custom_css,
    render_footer,
    render_sidebar_brand,
    render_sidebar_nav_label,
    render_sidebar_telemetry,
)
from utils.model_utils import log_system_action, load_ml_assets, load_historical_datasets

# Set up page configurations
st.set_page_config(
    page_title="ExplainChurn AI - Enterprise Customer Churn Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session States
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    log_system_action(
        "Platform Bootloader Initialized",
        "INFO",
        "ExplainChurn AI dashboard successfully booted. Assets cached.",
    )

if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = "Executive Dashboard"

if "selected_customer" not in st.session_state:
    st.session_state["selected_customer"] = "-- Manual Input --"

# Inject custom CSS style overrides
inject_custom_css()

# Pre-load ML and database assets so first loads are fast
try:
    load_ml_assets()
    load_historical_datasets()
except Exception as e:
    st.error(f"Error loading system assets: {str(e)}")

# ----------------------------------------------------
# Sidebar Navigation Panel (FIXED FOR REDIRECTION)
# ----------------------------------------------------
render_sidebar_brand()
render_sidebar_nav_label()

tabs = [
    "Executive Dashboard",
    "Prediction Diagnostics",
    "Customer Segmentation",
    "Analytics & Validation",
    "Reports & Exporters",
    "Administrative Panel",
]

# Get index dynamically from session_state to prevent widget-lock errors
default_index = tabs.index(st.session_state.get("current_tab", "Executive Dashboard"))

selected_tab = st.sidebar.radio(
    "Navigation Menu",
    options=tabs,
    index=default_index,
    label_visibility="collapsed",
)

# Manually track the active tab in session state safely
st.session_state["current_tab"] = selected_tab

render_sidebar_telemetry()

# ----------------------------------------------------
# Dashboard Content Router
# ----------------------------------------------------
from views.dashboard_view import render_dashboard_view
from views.prediction_view import render_prediction_view
from views.segmentation_view import render_segmentation_view
from views.analytics_view import render_analytics_view
from views.reports_view import render_reports_view
from views.admin_view import render_admin_view

if selected_tab == "Executive Dashboard":
    render_dashboard_view()
elif selected_tab == "Prediction Diagnostics":
    render_prediction_view()
elif selected_tab == "Customer Segmentation":
    render_segmentation_view()
elif selected_tab == "Analytics & Validation":
    render_analytics_view()
elif selected_tab == "Reports & Exporters":
    render_reports_view()
elif selected_tab == "Administrative Panel":
    render_admin_view()

render_footer()