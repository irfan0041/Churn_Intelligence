import os
import joblib
import pandas as pd
import numpy as np
import shap
import shap.explainers._tree as shap_tree
import streamlit as st
from datetime import datetime

# Monkey-patch SHAP parser to support brackets in newer XGBoost base_scores
original_decode = shap_tree.decode_ubjson_buffer

def patched_decode(*args, **kwargs):
    result = original_decode(*args, **kwargs)
    try:
        if "learner" in result and "learner_model_param" in result["learner"]:
            param = result["learner"]["learner_model_param"]
            if "base_score" in param and isinstance(param["base_score"], str):
                bs = param["base_score"]
                if bs.startswith('[') and bs.endswith(']'):
                    param["base_score"] = bs[1:-1]
    except Exception:
        pass
    return result

shap_tree.decode_ubjson_buffer = patched_decode

# Feature list expected by XGBoost model (47 features)
FEATURE_COLS = [
    'age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
    'avg_session_time', 'features_used', 'usage_growth_rate',
    'last_login_days_ago', 'monthly_fee', 'total_revenue', 'payment_failures',
    'support_tickets', 'avg_resolution_time', 'csat_score', 'escalations',
    'email_open_rate', 'marketing_click_rate', 'nps_score', 'referral_count',
    'tenure_segment', 'gender_Male', 'country_Bangladesh', 'country_Canada',
    'country_Germany', 'country_India', 'country_UK', 'country_USA', 'city_Delhi',
    'city_Dhaka', 'city_London', 'city_New York', 'city_Sydney', 'city_Toronto',
    'customer_segment_Individual', 'customer_segment_SME',
    'signup_channel_Referral', 'signup_channel_Web', 'contract_type_Quarterly',
    'contract_type_Yearly', 'payment_method_Card', 'payment_method_PayPal',
    'discount_applied_Yes', 'price_increase_last_3m_Yes',
    'complaint_type_Service', 'complaint_type_Technical',
    'survey_response_Satisfied', 'survey_response_Unsatisfied'
]

@st.cache_resource
def load_ml_assets():
    """Load model, scaler, and shap explainer once and cache them."""
    model_path = os.path.join('models', 'trained_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Pre-initialize SHAP TreeExplainer for fast on-demand explainability
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, explainer

@st.cache_data
def load_historical_datasets():
    """Load and cache the historical datasets."""
    raw_path = os.path.join('data', 'customer_churn_business_dataset.csv')
    proc_path = os.path.join('data', 'churn_processed.csv')
    
    df_raw = pd.read_csv(raw_path)
    df_proc = pd.read_csv(proc_path)
    
    return df_raw, df_proc

def preprocess_customer(data: dict) -> pd.DataFrame:
    """
    Transforms a single customer raw data dictionary into a 1-row DataFrame 
    containing the 47 preprocessed features expected by the model.
    """
    # Fill missing complaint_type with mode 'Technical'
    complaint = data.get('complaint_type', 'Technical')
    if pd.isna(complaint) or not complaint:
         complaint = 'Technical'

    # Compute tenure_segment
    tenure = int(data.get('tenure_months', 1))
    if tenure <= 6:
        tenure_seg = 0
    elif tenure <= 12:
        tenure_seg = 1
    elif tenure <= 24:
        tenure_seg = 2
    else:
        tenure_seg = 3

    # Construct the dictionary of preprocessed values
    processed_dict = {
        'age': int(data.get('age', 35)),
        'tenure_months': tenure,
        'monthly_logins': int(data.get('monthly_logins', 10)),
        'weekly_active_days': int(data.get('weekly_active_days', 3)),
        'avg_session_time': float(data.get('avg_session_time', 15.0)),
        'features_used': int(data.get('features_used', 3)),
        'usage_growth_rate': float(data.get('usage_growth_rate', 0.0)),
        'last_login_days_ago': int(data.get('last_login_days_ago', 5)),
        'monthly_fee': float(data.get('monthly_fee', 50.0)),
        'total_revenue': float(data.get('total_revenue', 500.0)),
        'payment_failures': int(data.get('payment_failures', 0)),
        'support_tickets': int(data.get('support_tickets', 0)),
        'avg_resolution_time': float(data.get('avg_resolution_time', 0.0)),
        'csat_score': float(data.get('csat_score', 8.0)),
        'escalations': int(data.get('escalations', 0)),
        'email_open_rate': float(data.get('email_open_rate', 0.2)),
        'marketing_click_rate': float(data.get('marketing_click_rate', 0.05)),
        'nps_score': int(data.get('nps_score', 8)),
        'referral_count': int(data.get('referral_count', 0)),
        'tenure_segment': tenure_seg,
        
        # Gender
        'gender_Male': data.get('gender') == 'Male',
        
        # Country (Australia is dropped category)
        'country_Bangladesh': data.get('country') == 'Bangladesh',
        'country_Canada': data.get('country') == 'Canada',
        'country_Germany': data.get('country') == 'Germany',
        'country_India': data.get('country') == 'India',
        'country_UK': data.get('country') == 'UK',
        'country_USA': data.get('country') == 'USA',
        
        # City (Berlin is dropped category)
        'city_Delhi': data.get('city') == 'Delhi',
        'city_Dhaka': data.get('city') == 'Dhaka',
        'city_London': data.get('city') == 'London',
        'city_New York': data.get('city') == 'New York',
        'city_Sydney': data.get('city') == 'Sydney',
        'city_Toronto': data.get('city') == 'Toronto',
        
        # Customer Segment (Enterprise is dropped category)
        'customer_segment_Individual': data.get('customer_segment') == 'Individual',
        'customer_segment_SME': data.get('customer_segment') == 'SME',
        
        # Signup Channel (Mobile is dropped category)
        'signup_channel_Referral': data.get('signup_channel') == 'Referral',
        'signup_channel_Web': data.get('signup_channel') == 'Web',
        
        # Contract Type (Monthly is dropped category)
        'contract_type_Quarterly': data.get('contract_type') == 'Quarterly',
        'contract_type_Yearly': data.get('contract_type') == 'Yearly',
        
        # Payment Method (Bank Transfer is dropped category)
        'payment_method_Card': data.get('payment_method') == 'Card',
        'payment_method_PayPal': data.get('payment_method') == 'PayPal',
        
        # Boolean inputs
        'discount_applied_Yes': data.get('discount_applied') == 'Yes',
        'price_increase_last_3m_Yes': data.get('price_increase_last_3m') == 'Yes',
        
        # Complaint Type (Billing is dropped category)
        'complaint_type_Service': complaint == 'Service',
        'complaint_type_Technical': complaint == 'Technical',
        
        # Survey Response (Neutral is dropped category)
        'survey_response_Satisfied': data.get('survey_response') == 'Satisfied',
        'survey_response_Unsatisfied': data.get('survey_response') == 'Unsatisfied',
    }
    
    # Create DataFrame and ensure correct column order
    df = pd.DataFrame([processed_dict])
    df = df[FEATURE_COLS]
    return df

def preprocess_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a DataFrame of raw customer data into a DataFrame
    containing the 47 preprocessed features expected by the model.
    """
    # Preprocess each row and concatenate
    records = df_raw.to_dict(orient='records')
    processed_dfs = [preprocess_customer(r) for r in records]
    df_processed = pd.concat(processed_dfs, ignore_index=True)
    return df_processed

def predict_customer(customer_raw: dict) -> tuple[float, int, str]:
    """
    Predicts churn probability, predicted class, and risk level.
    Returns: (probability, class_label, risk_level)
    """
    model, _, _ = load_ml_assets()
    features_df = preprocess_customer(customer_raw)
    
    # Predict using XGBoost (on unscaled data)
    prob = float(model.predict_proba(features_df)[0, 1])
    pred_label = int(prob >= 0.5)
    
    # Categorize risk level
    if prob >= 0.7:
        risk_level = "High"
    elif prob >= 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    return prob, pred_label, risk_level

def get_shap_values_for_customer(features_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Returns SHAP values and base value (expected value) for a single customer.
    """
    _, _, explainer = load_ml_assets()
    shap_output = explainer(features_df)
    
    # For binary classification, TreeExplainer returns values for the positive class (churn)
    if isinstance(shap_output, list):
        shap_values = shap_output[1]
    else:
        shap_values = shap_output
        
    return shap_values.values[0], explainer.expected_value

def log_prediction(customer_id: str, raw_data: dict, probability: float, risk_level: str):
    """Logs the prediction to data/prediction_history.csv to persist histories."""
    history_file = os.path.join('data', 'prediction_history.csv')
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'customer_id': customer_id,
        'probability': round(probability, 4),
        'risk_level': risk_level,
        **{k: v for k, v in raw_data.items() if k not in ['customer_id', 'churn']}
    }
    
    df_new = pd.DataFrame([log_entry])
    
    if os.path.exists(history_file):
        try:
            df_history = pd.read_csv(history_file)
            df_history = pd.concat([df_history, df_new], ignore_index=True)
            df_history.to_csv(history_file, index=False)
        except Exception as e:
            # Fallback if corrupted
            df_new.to_csv(history_file, index=False)
    else:
        # Create directories if missing
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        df_new.to_csv(history_file, index=False)

def log_system_action(action: str, status: str = "INFO", details: str = ""):
    """Logs a system action/health event to data/system_logs.csv."""
    logs_file = os.path.join('data', 'system_logs.csv')
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'action': action,
        'status': status,
        'details': details
    }
    
    df_new = pd.DataFrame([log_entry])
    
    if os.path.exists(logs_file):
        try:
            df_logs = pd.read_csv(logs_file)
            df_logs = pd.concat([df_logs, df_new], ignore_index=True)
            # Cap logs to last 1000 items
            if len(df_logs) > 1000:
                df_logs = df_logs.tail(1000)
            df_logs.to_csv(logs_file, index=False)
        except Exception as e:
            df_new.to_csv(logs_file, index=False)
    else:
        os.makedirs(os.path.dirname(logs_file), exist_ok=True)
        df_new.to_csv(logs_file, index=False)
