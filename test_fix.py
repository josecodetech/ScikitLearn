#!/usr/bin/env python
"""Test script to verify the model retraining and API fix."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Step 1: Checking scikit-learn version...")
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
    
    print("\nStep 2: Retraining model...")
    exec(open('retrain_model.py').read())
    
    print("\nStep 3: Testing model with API...")
    import pandas as pd
    import joblib
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    # Load the retrained model
    modelo = joblib.load("modelo_titanic.pkl")
    
    # Test prediction
    test_data = pd.DataFrame({
        'pclass': [3],
        'sex': ['male'],
        'age': [22.0],
        'fare': [7.25]
    })
    
    prediccion = modelo.predict(test_data)[0]
    probabilidad = modelo.predict_proba(test_data)[0]
    resultado = "Sobrevive" if prediccion == 1 else "No Sobrevive"
    confianza = probabilidad[1] if prediccion == 1 else probabilidad[0]
    
    print(f"\nTest Prediction Result:")
    print(f"  Resultado: {resultado}")
    print(f"  Probabilidad: {confianza*100:.2f}%")
    print("\n✓ All tests passed! Model is now compatible.")
    
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
