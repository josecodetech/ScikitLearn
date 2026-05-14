"""Retrain the Titanic model to fix scikit-learn compatibility issues."""

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

# Load Titanic dataset
df = sns.load_dataset('titanic')
print("--- TITANIC DATASET INFO ---")
print(df[['survived','pclass','sex','age','fare']].head())

# Prepare data
X = df[['pclass','sex','age','fare']]
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining with {len(X_train)} passengers, testing with {len(X_test)}.")

# Create preprocessing pipelines
transformer_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

transformer_categorico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors
preprocesador = ColumnTransformer(transformers=[
    ('num', transformer_numerico, ['age','fare']),
    ('cat', transformer_categorico, ['sex', 'pclass'])
])

# Create final pipeline
pipeline_titanic = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('clasificador', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
print("\n--- Training Titanic Model ---")
pipeline_titanic.fit(X_train, y_train)
predicciones = pipeline_titanic.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, predicciones))

# Save model
joblib.dump(pipeline_titanic, 'modelo_titanic.pkl')
print("\nModel saved successfully!")
