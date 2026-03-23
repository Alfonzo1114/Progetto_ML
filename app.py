from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from schema import PatientData

app = FastAPI(title="Stroke Prediction ML API")
model_state = None

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

@app.on_event("startup")
def load_model():
    global model_state
    try:
        with open("modelo_entrenado.pkl", "rb") as f:
            model_state = pickle.load(f)
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el modelo. ¿Has ejecutado el notebook para guardar 'modelo_entrenado.pkl'? Error: {e}")

@app.post("/predict")
def predict(data: PatientData):
    if model_state is None:
        raise HTTPException(
            status_code=500, 
            detail="Error 500: El archivo 'modelo_entrenado.pkl' no fue encontrado. Ve a Jupyter y da clic en 'Run All' para que el código genere el archivo localmente."
        )
    
    log_theta = model_state['log_theta']
    mu_x = model_state['mu_x']
    sigma_x = model_state['sigma_x']
    umbral = model_state['umbral']

    # Convertimos los datos intuitivos en el vector de 16 variables exactas que espera el modelo matemático.
    # Así el servidor abstrae toda la complejidad y el cliente sólo envía campos normales (edad, BMI, género)
    features_list = [
        data.age,
        data.hypertension,
        data.heart_disease,
        data.avg_glucose_level,
        data.bmi,
        1.0 if data.gender == 'Male' else 0.0,
        1.0 if data.gender == 'Other' else 0.0,
        1.0 if data.ever_married == 'Yes' else 0.0,
        1.0 if data.work_type == 'Never_worked' else 0.0,
        1.0 if data.work_type == 'Private' else 0.0,
        1.0 if data.work_type == 'Self-employed' else 0.0,
        1.0 if data.work_type == 'children' else 0.0,
        1.0 if data.Residence_type == 'Urban' else 0.0,
        1.0 if data.smoking_status == 'formerly smoked' else 0.0,
        1.0 if data.smoking_status == 'never smoked' else 0.0,
        1.0 if data.smoking_status == 'smokes' else 0.0
    ]
    
    features = np.array(features_list, dtype=np.float32)

    features_norm = (features - mu_x) / sigma_x
    X_b = np.append(features_norm, 1.0)
    z = np.dot(X_b, log_theta)
    prob_stroke = float(np.squeeze(sigmoid(z)))
    prediccion = int(prob_stroke >= umbral)

    return {
        "modelo": "Clasificador Logístico (Ponderado)",
        "probabilidad_stroke": prob_stroke,
        "prediccion_binaria": prediccion,
        "umbral_utilizado": umbral
    }
