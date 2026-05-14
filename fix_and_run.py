#!/usr/bin/env python3
"""
Complete fix: Retrain model and test the API immediately
Run with: python fix_and_run.py
"""

import sys
import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings

# Suppress version warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

print("=" * 60)
print("TITANIC MODEL RETRAINING - SCIKIT-LEARN 1.8.0 FIX")
print("=" * 60)

try:
    # Step 1: Verify versions
    import sklearn
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    print(f"✓ scikit-learn version: {sklearn.__version__}")
    
    # Step 2: Load and prepare data
    print("\n[1/4] Loading Titanic dataset...")
    df = sns.load_dataset('titanic')
    print(f"✓ Dataset loaded: {len(df)} records")
    
    X = df[['pclass','sex','age','fare']]
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Step 3: Build pipeline
    print("\n[2/4] Building preprocessing pipeline...")
    transformer_numerico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    transformer_categorico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocesador = ColumnTransformer(transformers=[
        ('num', transformer_numerico, ['age','fare']),
        ('cat', transformer_categorico, ['sex', 'pclass'])
    ])
    
    pipeline_titanic = Pipeline(steps=[
        ('preprocesador', preprocesador),
        ('clasificador', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    print("✓ Pipeline configured")
    
    # Step 4: Train model
    print("\n[3/4] Training model...")
    pipeline_titanic.fit(X_train, y_train)
    predicciones = pipeline_titanic.predict(X_test)
    accuracy = (predicciones == y_test).mean()
    print(f"✓ Model trained with {accuracy*100:.2f}% accuracy")
    
    # Step 5: Save model
    print("\n[4/4] Saving model...")
    joblib.dump(pipeline_titanic, 'modelo_titanic.pkl')
    print("✓ Model saved as 'modelo_titanic.pkl'")
    
    # Step 6: Test prediction
    print("\n" + "=" * 60)
    print("TESTING MODEL WITH API-LIKE REQUEST")
    print("=" * 60)
    
    test_passenger = pd.DataFrame({
        'pclass': [1],
        'sex': ['male'],
        'age': [20.0],
        'fare': [20.0]
    })
    
    prediccion = pipeline_titanic.predict(test_passenger)[0]
    probabilidad = pipeline_titanic.predict_proba(test_passenger)[0]
    resultado = "Sobrevive" if prediccion == 1 else "No Sobrevive"
    confianza = probabilidad[1] if prediccion == 1 else probabilidad[0]
    
    print(f"\nTest Input:")
    print(f"  pclass: 1")
    print(f"  sex: male")
    print(f"  age: 20")
    print(f"  fare: 20")
    print(f"\nAPI Response:")
    print(f"  resultado: {resultado}")
    print(f"  probabilidad: {confianza*100:.2f}%")
    
    # Step 7: Verify the saved model works
    print("\n" + "=" * 60)
    print("VERIFYING SAVED MODEL")
    print("=" * 60)
    
    loaded_model = joblib.load('modelo_titanic.pkl')
    test_pred = loaded_model.predict(test_passenger)[0]
    print(f"\n✓ Loaded model prediction: {'Sobrevive' if test_pred == 1 else 'No Sobrevive'}")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS! Model is ready for FastAPI")
    print("=" * 60)
    print("\nYour API is now fixed. Try the /predict endpoint:")
    print("  curl -X POST http://127.0.0.1:8000/predict \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"pclass\": 1, \"sex\": \"male\", \"age\": 20, \"fare\": 20}'")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
