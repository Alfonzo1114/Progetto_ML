from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from schema import PatientData, load_runtime_predictor

runtime_model = None
MODEL_FILE = "modelo_entregado.pkl"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime_model
    try:
        runtime_model = load_runtime_predictor(MODEL_FILE)
        print(f"Modelo cargado exitosamente desde '{MODEL_FILE}'.")
    except Exception as e:
        print(
            "Advertencia: No se pudo cargar el modelo. "
            f"Archivo esperado: '{MODEL_FILE}'. Error: {e}"
        )
    yield


app = FastAPI(title="Stroke Prediction ML API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(data: PatientData):
    if runtime_model is None:
        raise HTTPException(
            status_code=500, 
            detail=(
                "Error 500: No se encontro un archivo de modelo valido. "
                "Genera el modelo desde notebook (por ejemplo 'modelo_entregado.pkl')."
            )
        )

    try:
        return runtime_model.predict(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}") from e
