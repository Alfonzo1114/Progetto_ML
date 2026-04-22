import os
import jax
import jax.numpy as jnp
import kagglehub
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# =====================================================================
# CLASES DE MODELOS Y PROCESAMIENTO
# =====================================================================

class DataPreprocessor:
    def __init__(self, umbral=0.35):
        self.umbral = umbral
        self.mu_x, self.sigma_x = None, None

    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        df_clean = df.copy()
        if 'id' in df_clean.columns:
            df_clean = df_clean.drop(columns=['id'])

        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        df_clean['is_senior'] = (df_clean['age'] > 65).astype(float)
        def categorize_bmi(bmi):
            if bmi < 18.5: return 'Underweight'
            if bmi < 25: return 'Normal'
            if bmi < 30: return 'Overweight'
            return 'Obese'
        df_clean['bmi_cat'] = df_clean['bmi'].apply(categorize_bmi)
        df_clean['comorbidity_index'] = df_clean['hypertension'] + df_clean['heart_disease']
        df_clean['smoking_risk'] = ((df_clean['age'] > 50) & 
                                   (df_clean['smoking_status'].isin(['smokes', 'formerly smoked']))).astype(float)

        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col].values.reshape(-1, 1)

        X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)

        self.mu_x, self.sigma_x = X_encoded.mean().values, X_encoded.std().values
        self.sigma_x = np.where(self.sigma_x == 0, 1.0, self.sigma_x)
        X_norm = (X_encoded.values - self.mu_x) / self.sigma_x

        y_array = np.array(y, dtype=np.float32)
        ones = np.ones((X_norm.shape[0], 1), dtype=np.float32)
        X_b = np.hstack([X_norm, ones])

        return train_test_split(X_b, y_array, test_size=test_size, stratify=y_array, random_state=random_state)

class LinearRegressionJAX:
    def __init__(self, preprocessor):
        self.theta = None
        self.preprocessor = preprocessor

    def train(self, X_train, y_train):
        XT = X_train.T
        self.theta = jnp.linalg.solve(jnp.dot(XT, X_train), jnp.dot(XT, y_train))

    def predict(self, X_data):
        return jnp.dot(X_data, self.theta)

@jax.jit
def sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))

class LogisticRegressionJAX:
    def __init__(self, preprocessor, learning_rate=0.1, epochs=1000):
        self.preprocessor = preprocessor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def train(self, X_train, y_train):
        n_features = X_train.shape[1]
        self.theta = jnp.zeros((n_features, 1))
        for _ in range(self.epochs):
            loss, grad = jax.value_and_grad(lambda t: -jnp.mean(y_train * jnp.log(sigmoid(jnp.dot(X_train, t)) + 1e-15) + (1-y_train)*jnp.log(1-sigmoid(jnp.dot(X_train, t)) + 1e-15)))(self.theta)
            self.theta = self.theta - self.learning_rate * grad

    def predict_proba(self, X_data):
        return sigmoid(jnp.dot(X_data, self.theta))

    def predict(self, X_data):
        return (self.predict_proba(X_data) >= self.preprocessor.umbral).astype(int)

class MultilayerPerceptronJAX:
    def __init__(self, preprocessor, hidden_size=32, learning_rate=0.005, epochs=3000):
        self.preprocessor = preprocessor
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.params = None

    def train(self, X_train, y_train):
        n_features = X_train.shape[1]
        key = jax.random.PRNGKey(42)
        limit1 = jnp.sqrt(6.0 / (n_features + self.hidden_size))
        W1 = jax.random.uniform(key, (n_features, self.hidden_size), minval=-limit1, maxval=limit1)
        b1 = jnp.zeros((self.hidden_size,))
        limit2 = jnp.sqrt(6.0 / (self.hidden_size + 1))
        W2 = jax.random.uniform(key, (self.hidden_size, 1), minval=-limit2, maxval=limit2)
        b2 = jnp.zeros((1,))
        self.params = (W1, b1, W2, b2)

        @jax.jit
        def update(p, X, y):
            def loss_fn(params):
                w1, bb1, w2, bb2 = params
                a1 = jnp.maximum(0, jnp.dot(X, w1) + bb1)
                preds = sigmoid(jnp.dot(a1, w2) + bb2)
                return -jnp.mean(y * jnp.log(preds + 1e-7) + (1-y)*jnp.log(1-preds + 1e-7))
            l, grads = jax.value_and_grad(loss_fn)(p)
            return [param - self.learning_rate * jnp.clip(g, -1.0, 1.0) for param, g in zip(p, grads)], l

        for _ in range(self.epochs):
            self.params, _ = update(self.params, X_train, y_train)

    def predict_proba(self, X_data):
        w1, b1, w2, b2 = self.params
        a1 = jnp.maximum(0, jnp.dot(X_data, w1) + b1)
        return sigmoid(jnp.dot(a1, w2) + b2)

class SimpleFastDecisionTree:
    def __init__(self, max_depth=6, min_samples_split=10, min_samples_leaf=5, prediction_threshold=0.35):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.prediction_threshold = prediction_threshold
        self.sk_tree = None

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        self.sk_tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
        self.sk_tree.fit(X, y.flatten())

    def predict_proba(self, X):
        return self.sk_tree.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= self.prediction_threshold).astype(int)

class AdaBoostJAX:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.sk_ada = None

    def fit(self, X, y):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        self.sk_ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=self.n_estimators)
        self.sk_ada.fit(X, y.flatten())

    def predict_proba(self, X):
        return self.sk_ada.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.35):
        return (self.predict_proba(X) >= threshold).astype(int)

# =====================================================================
# MLFLOW PIPELINE
# =====================================================================

def main():
    print("Iniciando MLflow Pipeline...")
    # Configurar el experimento de MLflow
    mlflow.set_experiment("Stroke_Prediction_Models")
    
    # 1. Cargar Datos
    print("Cargando dataset...")
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    df = pd.read_csv(os.path.join(path, "healthcare-dataset-stroke-data.csv"))
    
    # 2. Preprocesamiento y SMOTE
    print("Preprocesando y aplicando SMOTE...")
    processor = DataPreprocessor(umbral=0.35)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = processor.prepare_data(df, target_col='stroke')
    
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw.flatten())
    
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train.reshape(-1, 1))
    X_test = jnp.array(X_test_raw)
    y_test = np.array(y_test_raw.reshape(-1, 1)).flatten()

    models_to_train = {
        "Linear Regression": (LinearRegressionJAX(processor), {}),
        "Logistic Regression": (LogisticRegressionJAX(processor, learning_rate=0.1, epochs=1000), 
                                {"learning_rate": 0.1, "epochs": 1000}),
        "Decision Tree": (SimpleFastDecisionTree(max_depth=6), 
                          {"max_depth": 6, "min_samples_split": 10, "min_samples_leaf": 5}),
        "Multilayer Perceptron": (MultilayerPerceptronJAX(processor, hidden_size=32, learning_rate=0.005, epochs=3000), 
                                  {"hidden_size": 32, "learning_rate": 0.005, "epochs": 3000}),
        "AdaBoost": (AdaBoostJAX(n_estimators=100), 
                     {"n_estimators": 100})
    }

    # 3. Entrenamiento y Tracking con MLflow
    for model_name, (model, params) in models_to_train.items():
        with mlflow.start_run(run_name=model_name):
            print(f"Entrenando {model_name}...")
            
            # Log Parameters
            mlflow.log_param("use_smote", True)
            mlflow.log_param("umbral_decision", 0.35)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Entrenamiento
            if hasattr(model, 'train'):
                model.train(X_train, y_train)
            elif hasattr(model, 'fit'):
                model.fit(np.array(X_train), np.array(y_train))
                
            # Predicción
            if model_name == 'Linear Regression':
                y_pred = (np.array(model.predict(X_test)) >= 0.5).astype(int).flatten()
            elif hasattr(model, 'predict_proba'):
                y_pred = (model.predict_proba(np.array(X_test)) >= 0.35).astype(int).flatten()
            else:
                y_pred = model.predict(np.array(X_test)).flatten()
            
            # Métricas
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Matriz de Confusión
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            ax.set_title(f'Matriz de Confusión - {model_name}')
            
            # Log Metrics y Artifacts
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_figure(fig, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
            plt.close(fig)
            
            # Log Model (Si es sklearn model podemos usar mlflow.sklearn)
            if hasattr(model, 'sk_tree'):
                mlflow.sklearn.log_model(model.sk_tree, "model")
            elif hasattr(model, 'sk_ada'):
                mlflow.sklearn.log_model(model.sk_ada, "model")
            
            print(f"[{model_name}] Recall: {rec:.3f} | F1: {f1:.3f}")

    print("\nPipeline ejecutado. Puedes ver los resultados iniciando la UI de MLflow:")
    print("👉 mlflow ui --host 0.0.0.0 --port 5000")

if __name__ == "__main__":
    main()
