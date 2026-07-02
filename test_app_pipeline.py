import os
import sys
import numpy as np
import pandas as pd
import joblib

# Ensure we import from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_utils import load_ml_assets, load_historical_datasets, preprocess_customer, predict_customer, get_shap_values_for_customer
from utils.business_utils import get_recommendations_for_customer
from utils.pdf_utils import generate_customer_pdf

def test_pipeline():
    print("==================================================")
    print("STARTING PIPELINE DIAGNOSTIC TESTS")
    print("==================================================")
    
    # 1. Test loading assets
    print("\n[Test 1/5] Loading Model, Scaler & SHAP Assets...")
    try:
        model, scaler, explainer = load_ml_assets()
        print("[OK] Model type:", type(model))
        print("[OK] Scaler type:", type(scaler))
        print("[OK] Explainer type:", type(explainer))
    except Exception as e:
        print("[FAIL] Test 1 Failed:", str(e))
        return False
        
    # 2. Test loading dataset
    print("\n[Test 2/5] Loading Historical Datasets...")
    try:
        df_raw, df_proc = load_historical_datasets()
        print(f"[OK] Raw dataset loaded: {df_raw.shape} rows/cols")
        print(f"[OK] Processed dataset loaded: {df_proc.shape} rows/cols")
    except Exception as e:
        print("[FAIL] Test 2 Failed:", str(e))
        return False
        
    # 3. Test preprocessing single customer
    print("\n[Test 3/5] Verifying Preprocessing Feature Alignment...")
    try:
        # Pick the first raw row
        sample_raw = df_raw.iloc[0].to_dict()
        features_df = preprocess_customer(sample_raw)
        
        print("[OK] Preprocessed DataFrame shape:", features_df.shape)
        if features_df.shape[1] != 47:
            print(f"[FAIL] Feature count mismatch: Expected 47, got {features_df.shape[1]}")
            return False
            
        print("[OK] Features perfectly match model signature:", list(features_df.columns) == list(model.feature_names_in_))
    except Exception as e:
        print("[FAIL] Test 3 Failed:", str(e))
        return False
        
    # 4. Test prediction & SHAP explainability
    print("\n[Test 4/5] Running Prediction & SHAP Calculations...")
    try:
        prob, pred_label, risk_level = predict_customer(sample_raw)
        print(f"[OK] Prediction Probability: {prob:.4f} (Class: {pred_label}, Risk: {risk_level})")
        if not (0.0 <= prob <= 1.0):
            print("[FAIL] Invalid prediction probability!")
            return False
            
        shap_vals, base_value = get_shap_values_for_customer(features_df)
        print("[OK] SHAP Values computed. Vector length:", len(shap_vals))
        print("[OK] Base Value expected:", base_value)
    except Exception as e:
        print("[FAIL] Test 4 Failed:", str(e))
        return False
        
    # 5. Test PDF Generation
    print("\n[Test 5/5] Generating PDF Report Card...")
    try:
        from utils.model_utils import FEATURE_COLS
        recoms = get_recommendations_for_customer(sample_raw, shap_vals, FEATURE_COLS)
        pdf_bytes = generate_customer_pdf(
            sample_raw['customer_id'],
            sample_raw,
            prob,
            risk_level=risk_level,
            recommendations=recoms,
            shap_vals=shap_vals,
            feature_names=FEATURE_COLS
        )
        print(f"[OK] PDF successfully generated. Size: {len(pdf_bytes)} bytes")
    except Exception as e:
        print("[FAIL] Test 5 Failed:", str(e))
        return False
        
    print("\n==================================================")
    print("[SUCCESS] ALL PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
    return True

if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)
