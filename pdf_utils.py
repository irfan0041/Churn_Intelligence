import os
from fpdf import FPDF
from datetime import datetime
import numpy as np

class ChurnReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(15, 20, 15)
        self.add_page()
        
    def header(self):
        # Draw background header block (Dark Navy #0F172A)
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 40, 'F')
        
        # Draw accent bar (Royal Blue #2563EB)
        self.set_fill_color(37, 99, 235)
        self.rect(0, 38, 210, 2, 'F')
        
        # Set text position & styling
        self.set_xy(15, 10)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 20)
        self.cell(0, 10, 'EXPLAINCHURN AI', ln=True)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(156, 163, 175) # Light gray
        self.cell(0, 5, 'Enterprise Customer Risk Assessment & Retention Report', ln=True)
        
        self.set_text_color(0, 0, 0)
        self.set_y(48)

    def footer(self):
        self.set_y(-20)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(156, 163, 175)
        self.cell(0, 10, f'Confidential | ExplainChurn AI Platform | Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='L')
        self.cell(0, 10, f'Page {self.page_no()}', align='R')

    def add_section_header(self, title):
        self.ln(6)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(15, 23, 42) # Navy
        self.cell(0, 8, title, ln=True)
        # Accent line under section header
        self.set_fill_color(79, 70, 229)
        self.rect(self.get_x(), self.get_y(), 180, 1, 'F')
        self.ln(4)

def generate_customer_pdf(customer_id: str, raw_data: dict, probability: float, risk_level: str, recommendations: list, shap_vals: np.ndarray, feature_names: list) -> bytes:
    """
    Generates a beautifully formatted PDF report for a customer and returns the bytes.
    """
    pdf = ChurnReportPDF()
    
    # ----------------------------------------------------
    # SECTION 1: CUSTOMER METADATA
    # ----------------------------------------------------
    pdf.add_section_header('CUSTOMER DEMOGRAPHIC & CONTRACT PROFILE')
    
    pdf.set_font('Helvetica', '', 10)
    
    # Table of customer features
    profile_data = [
        ('Customer ID:', customer_id, 'Tenure Months:', f"{raw_data.get('tenure_months', 'N/A')} mo"),
        ('Customer Segment:', raw_data.get('customer_segment', 'N/A'), 'Contract Type:', raw_data.get('contract_type', 'N/A')),
        ('Age / Gender:', f"{raw_data.get('age', 'N/A')} / {raw_data.get('gender', 'N/A')}", 'Payment Method:', raw_data.get('payment_method', 'N/A')),
        ('Country / City:', f"{raw_data.get('country', 'N/A')} / {raw_data.get('city', 'N/A')}", 'Discount Applied:', raw_data.get('discount_applied', 'N/A'))
    ]
    
    for row in profile_data:
        # Col 1
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(40, 7, row[0], border=0)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(50, 7, str(row[1]), border=0)
        
        # Col 2
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(40, 7, row[2], border=0)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(50, 7, str(row[3]), border=0, ln=True)
        
    # ----------------------------------------------------
    # SECTION 2: CHURN RISK PROFILE
    # ----------------------------------------------------
    pdf.add_section_header('PREDICTED RISK ASSESSMENT')
    
    # Add a visual panel (colored border card)
    pdf.set_fill_color(248, 250, 252) # Light blueish grey
    pdf.rect(15, pdf.get_y(), 180, 24, 'F')
    
    # Text inside panel
    pdf.set_y(pdf.get_y() + 4)
    pdf.set_x(20)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(50, 6, 'CHURN PROBABILITY:')
    pdf.cell(40, 6, 'RISK RATING:')
    pdf.cell(50, 6, 'ESTIMATED REVENUE AT RISK (LTV):', ln=True)
    
    pdf.set_x(20)
    pdf.set_font('Helvetica', 'B', 14)
    # Highlight Churn probability
    pdf.set_text_color(239, 68, 68) # Red
    pdf.cell(50, 8, f"{probability * 100:.2f}%")
    
    # Highlight Risk Rating with Colors
    if risk_level == "High":
        pdf.set_text_color(239, 68, 68)
    elif risk_level == "Medium":
        pdf.set_text_color(245, 158, 11) # Orange
    else:
        pdf.set_text_color(34, 197, 94) # Green
    pdf.cell(40, 8, f"{risk_level.upper()}")
    
    # LTV
    pdf.set_text_color(15, 23, 42)
    ltv = float(raw_data.get('total_revenue', 1000.0))
    if ltv < 500: ltv = 1500.0
    pdf.cell(50, 8, f"${ltv:,.2f}", ln=True)
    
    # Reset text color
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    
    # ----------------------------------------------------
    # SECTION 3: SHAP EXPLANATIONS & CHURN DRIVERS
    # ----------------------------------------------------
    pdf.add_section_header('EXPLAINABLE AI - SHAP RISK DRIVERS')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 5, 'The primary factors driving this customer\'s predicted behavior, quantified by SHAP analysis:', ln=True)
    pdf.ln(2)
    
    # Extract top positive (increasing risk) and negative (decreasing risk) features
    shap_drivers = []
    for i, name in enumerate(feature_names):
        val = shap_vals[i]
        # Map encoded name to business name
        cleaned_name = name.replace('_', ' ').title()
        shap_drivers.append((cleaned_name, val))
        
    # Sort by absolute SHAP value
    shap_drivers_sorted = sorted(shap_drivers, key=lambda x: abs(x[1]), reverse=True)
    
    # Display top 5 SHAP values in a clean table
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(80, 6, '  SHAP Risk Feature', border='B', fill=True)
    pdf.cell(50, 6, 'Impact Score', border='B', fill=True)
    pdf.cell(50, 6, 'Behavior Impact', border='B', fill=True, ln=True)
    
    pdf.set_font('Helvetica', '', 9)
    for name, val in shap_drivers_sorted[:5]:
        pdf.cell(80, 6, f"  {name}")
        
        # Color score
        if val > 0:
            pdf.set_text_color(239, 68, 68)
            direction = "Increases Risk"
            symbol = f"+{val:.4f}"
        else:
            pdf.set_text_color(34, 197, 94)
            direction = "Decreases Risk"
            symbol = f"{val:.4f}"
            
        pdf.cell(50, 6, symbol)
        pdf.cell(50, 6, direction, ln=True)
        pdf.set_text_color(0, 0, 0)
        
    pdf.ln(4)
    
    # Write English description summary
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Executive Summary Translation:', ln=True)
    pdf.set_font('Helvetica', 'I', 9)
    
    top_pos_features = [name for name, val in shap_drivers_sorted if val > 0][:3]
    features_str = ", ".join([f"'{f}'" for f in top_pos_features])
    summary_text = (
        f"This customer exhibits a Churn Risk of {probability * 100:.1f}%. The prediction is highly driven by "
        f"{features_str}. Immediate loyalty intervention is advised to mitigate retention risk."
    )
    pdf.multi_cell(180, 5, summary_text)
    
    # ----------------------------------------------------
    # SECTION 4: RETENTION STRATEGY & ROI
    # ----------------------------------------------------
    pdf.add_section_header('RECOMMENDED RETENTION PLAYBOOK')
    
    # Table headers
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(60, 6, '  Retention Action', border='B', fill=True)
    pdf.cell(40, 6, 'Intervention Cost', border='B', fill=True)
    pdf.cell(40, 6, 'Success Probability Boost', border='B', fill=True)
    pdf.cell(40, 6, 'Estimated Action ROI', border='B', fill=True, ln=True)
    
    pdf.set_font('Helvetica', '', 9)
    for action in recommendations:
        pdf.cell(60, 6, f"  {action['action']}")
        pdf.cell(40, 6, f"${action['cost']:.2f}")
        pdf.cell(40, 6, f"+{action['success_boost']*100:.0f}%")
        
        # Color ROI
        if action['roi'] > 0:
            pdf.set_text_color(34, 197, 94)
            roi_str = f"+{action['roi']:.1f}%"
        else:
            pdf.set_text_color(239, 68, 68)
            roi_str = f"{action['roi']:.1f}%"
            
        pdf.cell(40, 6, roi_str, ln=True)
        pdf.set_text_color(0, 0, 0)
        
    pdf.ln(4)
    
    # Action item detail description blocks
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Playbook Action Details:', ln=True)
    for i, action in enumerate(recommendations, start=1):
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(0, 5, f"{i}. {action['action']} (Trigger: {action['trigger_feature'].replace('_', ' ').title()})", ln=True)
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(180, 4, action['description'])
        pdf.ln(1)
        
    # Return PDF byte representation
    return bytes(pdf.output())
