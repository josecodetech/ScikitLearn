#!/usr/bin/env python3
"""Diagnose the current state of the model and environment."""

import os
import sys
from datetime import datetime

print("\n" + "="*70)
print("DIAGNOSTIC CHECK - MODEL AND ENVIRONMENT")
print("="*70 + "\n")

# Check Python and scikit-learn versions
import sklearn
print(f"✓ Python: {sys.version.split()[0]}")
print(f"✓ scikit-learn: {sklearn.__version__}\n")

# Check model file
model_path = 'modelo_titanic.pkl'
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    print(f"✓ Model file exists: {model_path}")
    print(f"  - Size: {file_size:,} bytes")
    print(f"  - Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
else:
    print(f"✗ Model file MISSING: {model_path}\n")

# Try to load the model
print("Attempting to load model...")
try:
    import warnings
    import joblib
    warnings.filterwarnings('ignore')
    
    modelo = joblib.load(model_path)
    print(f"✓ Model loaded successfully\n")
    
    # Try a test prediction
    print("Testing prediction...")
    import pandas as pd
    test_data = pd.DataFrame({
        'pclass': [1],
        'sex': ['male'],
        'age': [20.0],
        'fare': [20.0]
    })
    
    pred = modelo.predict(test_data)[0]
    prob = modelo.predict_proba(test_data)[0]
    result = "Sobrevive" if pred == 1 else "No Sobrevive"
    confidence = prob[1] if pred == 1 else prob[0]
    
    print(f"✓ Prediction works!")
    print(f"  Result: {result}")
    print(f"  Confidence: {confidence*100:.2f}%\n")
    
except Exception as e:
    print(f"✗ ERROR loading/using model: {type(e).__name__}: {e}\n")
    import traceback
    traceback.print_exc()

print("="*70)
print("Recommendation:")
print("="*70)
print("""
If you see errors above, run: python install_and_retrain.py

Then:
1. Stop FastAPI (Ctrl+C in the server terminal)
2. Start FastAPI again: python 17_fastAPI.py
3. Test again
""")
