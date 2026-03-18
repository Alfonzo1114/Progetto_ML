import os
import jax
import kagglehub
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from metaflow import FlowSpec, step



# Clases tomadas del jupyter notebook para que Metaflow pueda acceder a ellas en cualquier paso del flujo.

class DataPreprocessor:
    def __init__(self, umbral=0.5):
        self.umbral = umbral
        self.mu_x, self.sigma_x = None, None

    def prepare_data(self, df, target_col, test_size=0.25, random_state=42):
        X = df.drop(columns=[target_col])
        y = df[target_col].values.reshape(-1, 1)

        X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)

        self.mu_x, self.sigma_x = X_encoded.mean().values, X_encoded.std().values
        self.sigma_x = jnp.where(self.sigma_x == 0, 1.0, self.sigma_x)
        X_norm = (X_encoded.values - self.mu_x) / self.sigma_x

        y_array = jnp.array(y, dtype=jnp.float32)
        ones = jnp.ones((X_norm.shape[0], 1), dtype=jnp.float32)
        X_b = jnp.hstack([X_norm, ones])

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


def compute_weighted_bce_loss(theta, X, y, pos_weight):
    predictions = sigmoid(jnp.dot(X, theta))
    epsilon = 1e-15
    loss = -jnp.mean(
        pos_weight * y * jnp.log(predictions + epsilon) + 1.0 * (1 - y) * jnp.log(1 - predictions + epsilon))
    return loss


@jax.jit # Se compila esta función con JAX para acelerar su ejecución, especialmente durante el entrenamiento del modelo de regresión logística.
def update_step_weighted(theta, X, y, learning_rate, pos_weight):
    loss, gradients = jax.value_and_grad(compute_weighted_bce_loss)(theta, X, y, pos_weight) # Calcula el valor de la función de pérdida y sus gradientes con respecto a los parámetros theta.
    new_theta = theta - learning_rate * gradients
    return new_theta, loss


class LogisticRegressionJAX:
    def __init__(self, preprocessor, learning_rate=0.1, epochs=1000):
        self.preprocessor = preprocessor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.pos_weight = 1.0

    def train(self, X_train, y_train):
        n_features = X_train.shape[1]
        self.theta = jnp.zeros((n_features, 1))

        num_zeros = jnp.sum(y_train == 0)
        num_ones = jnp.sum(y_train == 1)
        self.pos_weight = num_zeros / num_ones
        print(f"Regresión Logistica valor del peso: {self.pos_weight:.2f}")

        for epoch in range(self.epochs):
            self.theta, loss = update_step_weighted(self.theta, X_train, y_train, self.learning_rate, self.pos_weight)

    def predict_proba(self, X_data):
        return sigmoid(jnp.dot(X_data, self.theta))

    def predict(self, X_data):
        probabilities = self.predict_proba(X_data)
        return (probabilities >= self.preprocessor.umbral).astype(int)


# flujo de Metaflow

class StrokePredictionFlow(FlowSpec):

    @step # con el decorador step se define cada paso del flujo, y el método next() se usa para indicar el siguiente paso a ejecutar.
    def start(self):
        """Inicia el flujo"""
        print("Inicia el flujo")
        self.next(self.load_data)

    @step
    def load_data(self):
        """Descarga del dataset."""
        path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
        csv_path = os.path.join(path, "healthcare-dataset-stroke-data.csv")
        self.df = pd.read_csv(csv_path)
        print(f"Dataset cargado. Tamaño: {self.df.shape}")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Limpieza de datos y separación en entrenamiento y validación"""
        df_clean = self.df.drop(columns=['id'])
        df_clean['bmi'] = df_clean['bmi'].fillna(df_clean['bmi'].mean())

        self.processor = DataPreprocessor(umbral=0.25)
        self.X_train, self.X_test, self.y_train, self.y_test_norm = self.processor.prepare_data(df_clean,
                                                                                                target_col='stroke')

        # Se ejecutan ambos entrenamientos en paralelo para aprovechar la capacidad de Metaflow de manejar ramas paralelas.
        self.next(self.train_linear, self.train_logistic)

    @step
    def train_linear(self):
        """Entrenamiento de la regresión lineal"""
        self.lin_model = LinearRegressionJAX(preprocessor=self.processor)
        self.lin_model.train(self.X_train, self.y_train)
        print("Entrenamiento de la regresión lineal.")
        self.next(self.evaluate)

    @step
    def train_logistic(self):
        """Entrenar el modelo de regresión Logística."""
        self.log_model = LogisticRegressionJAX(preprocessor=self.processor, learning_rate=0.1, epochs=1000)
        self.log_model.train(self.X_train, self.y_train)
        print("Entrenamiento de la regresión Logistica finalizada")
        self.next(self.evaluate)

    @step # con step se define el paso de evaluación, donde se calculan las métricas de rendimiento para ambos modelos y se preparan los datos para la visualización final.
    def evaluate(self, inputs):
        """paso de evaluación de los modelos y generación de gráficas"""
        # Metaflow pasa los estados de las ramas paralelas en el objeto inputs, lo que permite acceder a los modelos entrenados y a los datos de prueba para realizar la evaluación.
        self.processor = inputs.train_linear.processor
        self.X_test = inputs.train_linear.X_test
        self.y_test_norm = inputs.train_linear.y_test_norm

        self.lin_model = inputs.train_linear.lin_model
        self.log_model = inputs.train_logistic.log_model

        # cálculo de métricas para ambos modelos
        y_true = self.y_test_norm.astype(int).flatten()

        y_pred_lin_cont = self.lin_model.predict(self.X_test)
        y_pred_lin_bin = (y_pred_lin_cont >= self.processor.umbral).astype(int).flatten()
        y_pred_log_bin = self.log_model.predict(self.X_test).flatten()

        def get_metrics(y_true, y_pred, model_name):
            return {
                'Model': model_name,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1-Score': f1_score(y_true, y_pred, zero_division=0)
            }

        metrics_df = pd.DataFrame([
            get_metrics(y_true, y_pred_lin_bin, 'Clasificador lineal'),
            get_metrics(y_true, y_pred_log_bin, 'Clasificador Logístico (Ponderado)')
        ])

        print("\nEvaluación Métrica")
        print(metrics_df.to_string(index=False))


        self.y_true = y_true
        self.y_pred_lin_cont = y_pred_lin_cont
        self.next(self.end)

    @step
    def end(self):
        """Generación de gráficas"""
        z_lin = self.y_pred_lin_cont.flatten()
        z_log = jnp.dot(self.X_test, self.log_model.theta).flatten()

        sort_idx_lin = jnp.argsort(z_lin)
        z_lin_sorted = z_lin[sort_idx_lin]
        y_pred_lin_sorted = z_lin_sorted

        sort_idx_log = jnp.argsort(z_log)
        z_log_sorted = z_log[sort_idx_log]
        y_pred_log_sorted = 1.0 / (1.0 + jnp.exp(-z_log_sorted))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Linear Subplot
        axes[0].scatter(z_lin, self.y_true, color='blue', alpha=0.1, label='valores actuales')
        axes[0].plot(z_lin_sorted, y_pred_lin_sorted, color='red', linewidth=3, label='ajuste lineal')
        axes[0].axhline(self.processor.umbral, color='green', linestyle='--',
                        label=f'umbral ({self.processor.umbral})')
        axes[0].set_title('clasificador Lineal', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predictor (Z)', fontsize=12)
        axes[0].set_ylabel('Valor de salida', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Logistic Subplot
        axes[1].scatter(z_log, self.y_true, color='blue', alpha=0.1, label='valores actuales')
        axes[1].plot(z_log_sorted, y_pred_log_sorted, color='red', linewidth=3, label='Función Sigmoide')
        axes[1].axhline(self.processor.umbral, color='green', linestyle='--',
                        label=f'umbral ({self.processor.umbral})')
        axes[1].set_title('Clasificador Logístico ponderado', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predictor (Z)', fontsize=12)
        axes[1].set_ylabel('Probabilidad', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        # Guardar la gráfica en un archivo PNG para que pueda ser accedida después de la ejecución del flujo.
        plot_path = "classification_results.png"
        plt.savefig(plot_path)
        print(f"\nFin de flujo. Gráfica guardade en {plot_path}")


if __name__ == '__main__':
    StrokePredictionFlow()