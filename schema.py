from pydantic import BaseModel, Field
from typing import Literal
import pickle
import numpy as np

# Este nuevo esquema es altamente intuitivo para realizar pruebas
class PatientData(BaseModel):
    age: float = Field(..., description="Edad del paciente")
    gender: Literal['Male', 'Female', 'Other'] = Field('Male', description="Género")
    hypertension: int = Field(0, description="1 si tiene hipertensión, 0 si no")
    heart_disease: int = Field(0, description="1 si tiene enfermedad cardíaca, 0 si no")
    ever_married: Literal['Yes', 'No'] = Field('Yes', description="¿Se ha casado alguna vez?")
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'] = Field('Private', description="Tipo de trabajo principal")
    Residence_type: Literal['Urban', 'Rural'] = Field('Urban', description="Tipo de zona residencial")
    avg_glucose_level: float = Field(..., description="Nivel promedio de glucosa en sangre")
    bmi: float = Field(..., description="Índice de masa corporal")
    smoking_status: Literal['formerly smoked', 'never smoked', 'smokes', 'Unknown'] = Field('never smoked', description="Estatus de tabaquismo")


class StrokeRuntimePredictor:
    __slots__ = (
        "mu_x",
        "sigma_x",
        "umbral",
        "log_theta",
        "lin_theta",
        "dt_tree",
        "dt_prediction_threshold",
        "mlp_params",
        "adaboost_stumps",
    )

    def __init__(
        self,
        mu_x,
        sigma_x,
        umbral,
        log_theta=None,
        lin_theta=None,
        dt_tree=None,
        dt_prediction_threshold=None,
        mlp_params=None,
        adaboost_stumps=None,
    ):
        self.mu_x = np.asarray(mu_x, dtype=np.float32)
        sigma = np.asarray(sigma_x, dtype=np.float32)
        self.sigma_x = np.where(sigma == 0, 1.0, sigma)
        self.umbral = float(umbral)
        self.log_theta = None if log_theta is None else np.asarray(log_theta)
        self.lin_theta = None if lin_theta is None else np.asarray(lin_theta)
        self.dt_tree = dt_tree
        self.dt_prediction_threshold = float(
            dt_prediction_threshold if dt_prediction_threshold is not None else self.umbral
        )
        self.mlp_params = mlp_params
        self.adaboost_stumps = adaboost_stumps

    @classmethod
    def from_state(cls, state: dict):
        return cls(
            mu_x=state["mu_x"],
            sigma_x=state["sigma_x"],
            umbral=state.get("umbral", 0.5),
            log_theta=state.get("log_theta"),
            lin_theta=state.get("lin_theta"),
            dt_tree=state.get("dt_enhanced_tree"),
            dt_prediction_threshold=state.get("dt_prediction_threshold"),
            mlp_params=state.get("mlp_params"),
            adaboost_stumps=state.get("adaboost_stumps"),
        )

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _build_feature_vector(self, payload: dict):
        age = payload["age"]
        bmi = payload["bmi"]
        ht = payload["hypertension"]
        hd = payload["heart_disease"]
        smoking = payload["smoking_status"]

        # 1. Riesgo por edad avanzada
        is_senior = 1.0 if age > 65 else 0.0
        
        # 2. Índice de comorbilidad
        comorbidity = float(ht + hd)
        
        # 3. Riesgo combinado tabaquismo
        smoking_risk = 1.0 if (age > 50 and smoking in ["smokes", "formerly smoked"]) else 0.0
        
        # 4. BMI Categorization (Dummy columns)
        # Normal is dropped (first alphabetically: Normal, Obese, Overweight, Underweight)
        bmi_cat_Obese = 1.0 if bmi >= 30 else 0.0
        bmi_cat_Overweight = 1.0 if 25 <= bmi < 30 else 0.0
        bmi_cat_Underweight = 1.0 if bmi < 18.5 else 0.0

        base_features = [
            age,
            ht,
            hd,
            payload["avg_glucose_level"],
            bmi,
            is_senior,
            comorbidity,
            smoking_risk,
            1.0 if payload["gender"] == "Male" else 0.0,
            1.0 if payload["gender"] == "Other" else 0.0,
            1.0 if payload["ever_married"] == "Yes" else 0.0,
            1.0 if payload["work_type"] == "Never_worked" else 0.0,
            1.0 if payload["work_type"] == "Private" else 0.0,
            1.0 if payload["work_type"] == "Self-employed" else 0.0,
            1.0 if payload["work_type"] == "children" else 0.0,
            1.0 if payload["Residence_type"] == "Urban" else 0.0,
            1.0 if smoking == "formerly smoked" else 0.0,
            1.0 if smoking == "never smoked" else 0.0,
            1.0 if smoking == "smokes" else 0.0,
            bmi_cat_Obese,
            bmi_cat_Overweight,
            bmi_cat_Underweight
        ]

        if self.mu_x.shape[0] == len(base_features) + 1:
            synthetic_id = float(self.mu_x[0])
            return np.array([synthetic_id] + base_features, dtype=np.float32)

        if self.mu_x.shape[0] == len(base_features):
            return np.array(base_features, dtype=np.float32)

        raise ValueError(
            f"Incompatibilidad de features: API genera {len(base_features)}, modelo espera {self.mu_x.shape[0]}."
        )

    def _predict_tree_proba(self, x_b):
        if self.dt_tree is None:
            return None

        node = self.dt_tree
        while isinstance(node, dict) and node.get("type") == "node":
            feature_idx = int(node["feature"])
            threshold = float(node["threshold"])
            node = node["left"] if x_b[feature_idx] <= threshold else node["right"]

        if isinstance(node, dict) and node.get("type") == "leaf":
            return float(node["probability"])

        raise ValueError("Formato de arbol invalido para inferencia (se esperaba nodo/hoja enhanced).")

    def predict(self, payload: dict):
        features = self._build_feature_vector(payload)
        features_norm = (features - self.mu_x) / self.sigma_x
        x_b = np.append(features_norm, 1.0)

        resultados = []

        if self.log_theta is not None:
            z_log = np.dot(x_b, self.log_theta)
            prob_log = float(np.squeeze(self._sigmoid(z_log)))
            resultados.append({
                "modelo": "Clasificador Logistico (Ponderado)",
                "probabilidad_stroke": prob_log,
                "prediccion_binaria": int(prob_log >= self.umbral),
                "umbral_utilizado": self.umbral,
            })

        if self.lin_theta is not None:
            umbral_lin = 0.5
            score_lin = float(np.squeeze(np.dot(x_b, self.lin_theta)))
            resultados.append({
                "modelo": "Regresion Lineal",
                "probabilidad_stroke": score_lin,
                "prediccion_binaria": int(score_lin >= umbral_lin),
                "umbral_utilizado": umbral_lin,
            })

        prob_tree = self._predict_tree_proba(x_b)
        if prob_tree is not None:
            resultados.append({
                "modelo": "Decision Tree",
                "probabilidad_stroke": prob_tree,
                "prediccion_binaria": int(prob_tree >= self.dt_prediction_threshold),
                "umbral_utilizado": self.dt_prediction_threshold,
            })

        if self.mlp_params is not None:
            W1, b1, W2, b2 = self.mlp_params
            Z1 = np.dot(x_b, W1) + b1
            A1 = np.maximum(0, Z1)
            Z2 = np.dot(A1, W2) + b2
            prob_mlp = float(np.squeeze(self._sigmoid(Z2)))
            resultados.append({
                "modelo": "Multilayer Perceptron",
                "probabilidad_stroke": prob_mlp,
                "prediccion_binaria": int(prob_mlp >= self.umbral),
                "umbral_utilizado": self.umbral,
            })

        # --- AdaBoost: voto ponderado de stumps ---
        if self.adaboost_stumps is not None:
            score = 0.0
            for stump in self.adaboost_stumps:
                feat_idx = int(stump["feature_idx"])
                threshold = float(stump["threshold"])
                polarity = int(stump["polarity"])
                alpha = float(stump["alpha"])
                # Cada stump emite +1 o -1 según su polaridad y umbral
                if polarity == 1:
                    pred = 1.0 if x_b[feat_idx] <= threshold else -1.0
                else:
                    pred = -1.0 if x_b[feat_idx] <= threshold else 1.0
                score += alpha * pred
            # Convertimos el score acumulado a probabilidad con sigmoid
            prob_ada = float(self._sigmoid(score))
            resultados.append({
                "modelo": "AdaBoost",
                "probabilidad_stroke": prob_ada,
                "prediccion_binaria": int(prob_ada >= self.umbral),
                "umbral_utilizado": self.umbral,
            })

        return {"resultados": resultados}


def load_runtime_predictor(model_file: str):
    with open(model_file, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        return StrokeRuntimePredictor.from_state(obj)

    raise TypeError("El archivo de modelo debe contener un diccionario model_state exportado desde el notebook.")


