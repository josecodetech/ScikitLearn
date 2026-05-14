from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Creamos la aplicacion
app = FastAPI(title="API de Supervivencia del Titanic")

# Definimos el esquema del pasajero
class Pasajero(BaseModel):
    pclass: int
    sex: str
    age: float
    fare: float

# Cargamos el modelo entrenado
modelo = joblib.load("modelo_titanic.pkl")

@app.post("/predict")
def predecir_supervivencia(pasajero: Pasajero):
    # convertimos datos a dataframe
    datos = pd.DataFrame([pasajero.dict()])
    # realizamos la prediccion
    prediccion = modelo.predict(datos)[0]
    probabilidad = modelo.predict_proba(datos)[0]
    # devolvemos la respuesta en formato JSON
    resultado = "Sobrevive" if prediccion == 1 else "No Sobrevive"
    confianza = probabilidad[1] if prediccion == 1 else probabilidad[0]
    return {
        "status": "success",
        "resultado": resultado,
        "probabilidad": f"{confianza*100:.2f}%"
    }