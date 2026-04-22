import nbformat

with open('ML_v1.5.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

new_adaboost_source = """class AdaBoostJAX:
    \"\"\"
    Implementación de AdaBoost para clasificación binaria.
    Utiliza clasificadores heterogéneos (Decision Tree, Logistic, Linear, MLP).
    
    Parámetros:
    -----------
    n_estimators : int
        Número de clasificadores débiles a entrenar.
    \"\"\"
    def __init__(self, preprocessor=None, n_estimators=12):
        self.n_estimators = n_estimators
        self.preprocessor = preprocessor
        self.models = []

    def fit(self, X, y):
        \"\"\"
        Entrena el modelo AdaBoost con clasificadores heterogéneos mediante resampling.
        \"\"\"
        X_np = np.array(X)
        y_np = np.array(y).flatten()
        n_samples = X_np.shape[0]
        
        y_converted = np.where(y_np == 0, -1, 1)
        
        n_pos = np.sum(y_np == 1)
        n_neg = np.sum(y_np == 0)
        weights = np.where(y_np == 1, 1.0 / (2 * n_pos), 1.0 / (2 * n_neg))
        weights = weights / np.sum(weights)
        
        self.models = []
        
        # Si preprocessor es None, intentamos usar el global 'processor'
        proc = self.preprocessor if self.preprocessor is not None else processor
        
        for t in range(self.n_estimators):
            model_type = t % 4
            
            # Resampling basado en los pesos
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights)
            X_sample = X_np[indices]
            y_sample = y_np[indices]
            
            if model_type == 0:
                model = SimpleFastDecisionTree(max_depth=3, n_thresholds=10)
                model.fit(X_sample, y_sample)
            elif model_type == 1:
                model = LogisticRegressionJAX(preprocessor=proc, learning_rate=0.01, epochs=200)
                model.train(X_sample, y_sample)
            elif model_type == 2:
                model = LinearRegressionJAX(preprocessor=proc)
                model.train(X_sample, y_sample)
            elif model_type == 3:
                model = MultilayerPerceptronJAX(preprocessor=proc, hidden_size=8, learning_rate=0.01, epochs=200)
                model.train(X_sample, y_sample)
            
            # Generar predicciones sobre el dataset original
            if model_type == 2:
                preds = model.predict(X_np).flatten()
                predictions = np.where(preds >= 0.5, 1, -1)
            elif model_type == 0:
                preds = model.predict(X_np).flatten()
                predictions = np.where(preds == 1, 1, -1)
            else:
                preds = model.predict_proba(X_np).flatten()
                predictions = np.where(preds >= 0.5, 1, -1)
            
            misclassified = predictions != y_converted
            weighted_error = np.dot(weights, misclassified)
            
            if weighted_error >= 0.5:
                weighted_error = 0.5 - 1e-10

            epsilon = 1e-10
            alpha = 0.5 * np.log((1.0 - weighted_error + epsilon) / (weighted_error + epsilon))
            
            weights = weights * np.exp(-alpha * y_converted * predictions)
            weights = weights / np.sum(weights)
            
            self.models.append((model, alpha, model_type))
            
        print(f"AdaBoost entrenado con {self.n_estimators} modelos heterogéneos.")

    def predict_score(self, X):
        X_np = np.array(X)
        n_samples = X_np.shape[0]
        
        scores = np.zeros(n_samples)
        for model, alpha, model_type in self.models:
            if model_type == 2:
                preds = model.predict(X_np).flatten()
                predictions = np.where(preds >= 0.5, 1, -1)
            elif model_type == 0:
                preds = model.predict(X_np).flatten()
                predictions = np.where(preds == 1, 1, -1)
            else:
                preds = model.predict_proba(X_np).flatten()
                predictions = np.where(preds >= 0.5, 1, -1)
                
            scores += alpha * predictions
            
        return scores

    def predict_proba(self, X):
        scores = self.predict_score(X)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_stump_data(self):
        return []
"""

for cell in nb.cells:
    if cell.cell_type == 'code' and 'class AdaBoostJAX:' in cell.source:
        cell.source = new_adaboost_source
        break

# Update n_estimators in instantiations so it runs faster
for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'ada_model = AdaBoostJAX(n_estimators=50)' in cell.source:
            cell.source = cell.source.replace('ada_model = AdaBoostJAX(n_estimators=50)', 'ada_model = AdaBoostJAX(preprocessor=processor, n_estimators=12)')
        if 'ada_model_s = AdaBoostJAX(n_estimators=100)' in cell.source:
            cell.source = cell.source.replace('ada_model_s = AdaBoostJAX(n_estimators=100)', 'ada_model_s = AdaBoostJAX(preprocessor=processor, n_estimators=16)')

with open('ML_v1.5.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")
