#!/usr/bin/env python3
"""
Complete fix script: Install dependencies and retrain model
Run with: python install_and_retrain.py
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*70)
print("INSTALLING DEPENDENCIES AND RETRAINING MODEL")
print("="*70 + "\n")

# Install seaborn if not present
print("Installing seaborn...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "-q"])
print("✓ seaborn installed\n")

# Now import and run the training
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

print("Step 1: Loading Titanic dataset...")
df = sns.load_dataset('titanic')
X = df[['pclass','sex','age','fare']]
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Loaded {len(df)} records (train: {len(X_train)}, test: {len(X_test)})\n")

print("Step 2: Building pipeline...")
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
print("✓ Pipeline built\n")

print("Step 3: Training model...")
pipeline_titanic.fit(X_train, y_train)
accuracy = pipeline_titanic.score(X_test, y_test)
print(f"✓ Model trained with {accuracy*100:.2f}% accuracy\n")

print("Step 4: Saving model...")
joblib.dump(pipeline_titanic, 'modelo_titanic.pkl')
print("✓ Model saved to 'modelo_titanic.pkl'\n")

print("Step 5: Testing prediction...")
test_data = pd.DataFrame({
    'pclass': [1],
    'sex': ['male'],
    'age': [20.0],
    'fare': [20.0]
})
pred = pipeline_titanic.predict(test_data)[0]
prob = pipeline_titanic.predict_proba(test_data)[0]
result = "Sobrevive" if pred == 1 else "No Sobrevive"
confidence = prob[1] if pred == 1 else prob[0]
print(f"✓ Test prediction: {result} ({confidence*100:.2f}% confidence)\n")

print("="*70)
print("✓ SUCCESS! Model is now compatible with scikit-learn 1.8.0")
print("="*70)
print("\nNext steps:")
print("1. Stop FastAPI if running (Ctrl+C)")
print("2. Restart: python 17_fastAPI.py")
print("3. Test your API endpoint!")
print("\n")
