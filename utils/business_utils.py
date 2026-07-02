import numpy as np
import pandas as pd

# Default assumptions
DEFAULT_LTV = 5000.0       # Average lifetime value
DEFAULT_COST = 500.0       # Intervention cost
DEFAULT_SUCCESS = 0.70     # 70% baseline success rate

# Retention Action Definitions
RECOMMENDATION_CATALOG = {
    'contract_type': {
        'action': 'Offer Annual Loyalty Contract',
        'description': 'Transition the customer from Monthly/Quarterly billing to an Annual Contract with a 15% discount. Locks in commitment.',
        'cost': 400.0,
        'boost': 0.25
    },
    'tenure_months': {
        'action': 'Offer Annual Loyalty Contract',
        'description': 'Transition the customer from Monthly/Quarterly billing to an Annual Contract with a 15% discount. Locks in commitment.',
        'cost': 400.0,
        'boost': 0.25
    },
    'support_tickets': {
        'action': 'VIP Support Escalation & Care Manager',
        'description': 'Assign a dedicated Customer Care Lead. Schedule a 1-on-1 resolution call to address product blockages and outstanding tickets.',
        'cost': 600.0,
        'boost': 0.30
    },
    'escalations': {
        'action': 'VIP Support Escalation & Care Manager',
        'description': 'Assign a dedicated Customer Care Lead. Schedule a 1-on-1 resolution call to address product blockages and outstanding tickets.',
        'cost': 600.0,
        'boost': 0.30
    },
    'avg_resolution_time': {
        'action': 'VIP Support Escalation & Care Manager',
        'description': 'Assign a dedicated Customer Care Lead. Schedule a 1-on-1 resolution call to address product blockages and outstanding tickets.',
        'cost': 600.0,
        'boost': 0.30
    },
    'monthly_fee': {
        'action': 'Targeted Loyalty Billing Credit',
        'description': 'Apply a temporary 3-month credit of 20% on their billing plan. Reduces short-term financial pressure while they re-evaluate utility.',
        'cost': 250.0,
        'boost': 0.20
    },
    'total_revenue': {
        'action': 'Targeted Loyalty Billing Credit',
        'description': 'Apply a temporary 3-month credit of 20% on their billing plan. Reduces short-term financial pressure while they re-evaluate utility.',
        'cost': 250.0,
        'boost': 0.20
    },
    'payment_failures': {
        'action': 'Billing Upgrade Intervention',
        'description': 'Reach out to help update billing details. Offer card update discount or PayPal integration. Provide a 10-day grace period.',
        'cost': 100.0,
        'boost': 0.35
    },
    'weekly_active_days': {
        'action': 'Re-engagement & Training Webinar',
        'description': 'Trigger automated product tutorials and invite to a personalized customer onboarding refresh workshop. Boosts feature adoption.',
        'cost': 80.0,
        'boost': 0.15
    },
    'monthly_logins': {
        'action': 'Re-engagement & Training Webinar',
        'description': 'Trigger automated product tutorials and invite to a personalized customer onboarding refresh workshop. Boosts feature adoption.',
        'cost': 80.0,
        'boost': 0.15
    },
    'features_used': {
        'action': 'Feature Training & Upgrade Consultation',
        'description': 'Trigger automated product tutorials and invite to a personalized customer onboarding refresh workshop. Boosts feature adoption.',
        'cost': 120.0,
        'boost': 0.18
    },
    'nps_score': {
        'action': 'Executive Satisfaction Recovery Call',
        'description': 'Schedule an executive recovery outreach call from the VP of Success. Resolve structural complaints and offer a $100 service voucher.',
        'cost': 350.0,
        'boost': 0.22
    },
    'csat_score': {
        'action': 'Executive Satisfaction Recovery Call',
        'description': 'Schedule an executive recovery outreach call from the VP of Success. Resolve structural complaints and offer a $100 service voucher.',
        'cost': 350.0,
        'boost': 0.22
    },
    'default': {
        'action': 'Personal Relationship Call & Gift Voucher',
        'description': 'Send a personalized note and a $50 loyalty voucher to express appreciation and review their ongoing business goals.',
        'cost': 150.0,
        'boost': 0.12
    }
}

def calculate_segment_roi(segment_size: int, churn_rate: float, ltv: float = DEFAULT_LTV, cost: float = DEFAULT_COST, success_rate: float = DEFAULT_SUCCESS) -> dict:
    """
    Computes business ROI metrics for customer retention interventions.
    """
    expected_churners = segment_size * churn_rate
    intervention_cost = segment_size * cost
    
    # Expected customers saved = expected churners * success rate of campaign
    saved_customers = expected_churners * success_rate
    revenue_saved = saved_customers * ltv
    
    net_benefit = revenue_saved - intervention_cost
    roi_percent = (net_benefit / intervention_cost) * 100 if intervention_cost > 0 else 0.0
    
    return {
        'segment_size': segment_size,
        'expected_churners': round(expected_churners),
        'intervention_cost': intervention_cost,
        'saved_customers': round(saved_customers, 1),
        'revenue_saved': revenue_saved,
        'net_benefit': net_benefit,
        'roi_percent': roi_percent,
        'recommended': net_benefit > 0
    }

def get_recommendations_for_customer(customer_raw: dict, shap_vals: np.ndarray, feature_names: list) -> list[dict]:
    """
    Analyzes positive SHAP drivers for a single customer and returns
    personalized action cards, with customized ROI estimations.
    """
    # Map raw features to see if anything is contributing to churn
    # Find features with positive SHAP value (which increases churn risk)
    positive_drivers = []
    for i, name in enumerate(feature_names):
        shap_val = shap_vals[i]
        if shap_val > 0:
            # Clean up encoded feature name to match our recommendation keys
            base_name = name.split('_')[0]
            if name.startswith('gender_'):
                base_name = 'gender'
            elif name.startswith('country_'):
                base_name = 'country'
            elif name.startswith('city_'):
                base_name = 'city'
            elif name.startswith('customer_segment_'):
                base_name = 'customer_segment'
            elif name.startswith('signup_channel_'):
                base_name = 'signup_channel'
            elif name.startswith('contract_type_'):
                base_name = 'contract_type'
            elif name.startswith('payment_method_'):
                base_name = 'payment_method'
            elif name.startswith('discount_applied_'):
                base_name = 'discount_applied'
            elif name.startswith('price_increase_last_3m_'):
                base_name = 'price_increase_last_3m'
            elif name.startswith('complaint_type_'):
                base_name = 'complaint_type'
            elif name.startswith('survey_response_'):
                base_name = 'survey_response'
                
            positive_drivers.append({
                'feature': name,
                'base_feature': base_name,
                'shap_val': shap_val,
                'val': customer_raw.get(base_name, 'N/A')
            })
            
    # Sort positive drivers by SHAP impact descending
    positive_drivers = sorted(positive_drivers, key=lambda x: x['shap_val'], reverse=True)
    
    # Select actions based on top positive drivers
    selected_actions = []
    seen_actions = set()
    
    # Customer LTV
    ltv = float(customer_raw.get('total_revenue', 1000.0))
    if ltv < 500: # Ensure minimum baseline LTV for ROI calculations
        ltv = 1500.0
        
    for driver in positive_drivers:
        bf = driver['base_feature']
        cat = RECOMMENDATION_CATALOG.get(bf, RECOMMENDATION_CATALOG['default'])
        
        # Avoid duplicate actions
        if cat['action'] not in seen_actions:
            seen_actions.add(cat['action'])
            
            # Custom ROI for this action and customer
            cost = cat['cost']
            boost = cat['boost']
            
            # Expected saved value: boost (success probability) * customer LTV
            expected_saved = boost * ltv
            net_benefit = expected_saved - cost
            action_roi = (net_benefit / cost) * 100 if cost > 0 else 0
            
            selected_actions.append({
                'trigger_feature': driver['feature'],
                'feature_value': driver['val'],
                'shap_impact': driver['shap_val'],
                'action': cat['action'],
                'description': cat['description'],
                'cost': cost,
                'success_boost': boost,
                'expected_saved_val': expected_saved,
                'net_benefit': net_benefit,
                'roi': action_roi
            })
            
            # Limit to top 3 actions
            if len(selected_actions) >= 3:
                break
                
    # Fallback to default if no positive drivers found
    if not selected_actions:
        cat = RECOMMENDATION_CATALOG['default']
        cost = cat['cost']
        boost = cat['boost']
        expected_saved = boost * ltv
        net_benefit = expected_saved - cost
        action_roi = (net_benefit / cost) * 100
        
        selected_actions.append({
            'trigger_feature': 'churn_probability',
            'feature_value': 'General Risk',
            'shap_impact': 0.0,
            'action': cat['action'],
            'description': cat['description'],
            'cost': cost,
            'success_boost': boost,
            'expected_saved_val': expected_saved,
            'net_benefit': net_benefit,
            'roi': action_roi
        })
        
    return selected_actions

def segment_customers(df_raw: pd.DataFrame, df_proc: pd.DataFrame, model) -> pd.DataFrame:
    """
    Appends predictions and segments customers into High, Medium, Low risk.
    """
    # Avoid modifying inputs
    df = df_raw.copy()
    
    # Convert tenure_segment for prediction features
    X = df_proc.drop(columns=['customer_id', 'churn'], errors='ignore')
    X['tenure_segment'] = X['tenure_segment'].map({
        '0-6 months': 0,
        '6-12 months': 1,
        '12-24 months': 2,
        '24+ months': 3
    }).fillna(0).astype(int)
    
    # Predict probabilities (using XGBoost on unscaled data)
    probs = model.predict_proba(X)[:, 1]
    
    df['churn_probability'] = probs
    
    # Segmentation category
    df['risk_tier'] = pd.cut(
        probs, 
        bins=[-np.inf, 0.3, 0.7, np.inf], 
        labels=['Low', 'Medium', 'High']
    )
    
    # Priority
    df['retention_priority'] = pd.cut(
        probs, 
        bins=[-np.inf, 0.3, 0.7, np.inf], 
        labels=['Low Priority', 'Medium Priority', 'Immediate VIP Support']
    )
    
    return df
