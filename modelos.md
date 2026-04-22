# Modelos de Machine Learning Implementados

Este proyecto no utiliza librerías tradicionales (como Scikit-Learn) para el núcleo de los algoritmos predictivos, sino que implementa cada modelo desde cero utilizando **Python puro y JAX** (`jax.numpy`, `jax.grad`, `jax.jit`). 
Esto tiene un propósito educativo y otorga un control total sobre las matemáticas de optimización, lo que es especialmente útil para gestionar el desbalance de clases (95% sanos, 5% stroke) integrando penalizaciones directamente en las funciones de pérdida.

---

## 1. Regresión Lineal con JAX (`LinearRegressionJAX`)

- **Implementación en el Proyecto:**
  - En lugar de optimizar iterativamente con descenso de gradiente, la solución se encontró analíticamente utilizando la **Ecuación Normal** (`jnp.linalg.solve`).
  - Dado que el conjunto de datos tras el preprocesamiento tiene una dimensionalidad moderada, la inversión de matrices resulta computacionalmente eficiente y exacta.
  - Este modelo se incluyó puramente como una **línea base (baseline)** algorítmica. No es idóneo para clasificación binaria ya que sus predicciones no están acotadas probabilísticamente al rango $[0, 1]$.

---

## 2. Regresión Logística con JAX (`LogisticRegressionJAX`)

- **Implementación en el Proyecto:**
  - Totalmente optimizado mediante **Descenso de Gradiente**, usando `jax.value_and_grad` para el cálculo automático de derivadas (autodiff) y compilado con `@jax.jit` para acelerar drásticamente los bucles de entrenamiento.
  - **Manejo del Desbalance:** Se implementó una función de **Entropía Cruzada Binaria Ponderada (Weighted BCE)**. El modelo calcula de forma dinámica un peso (`pos_weight = num_zeros / num_ones`) aplicado únicamente a la clase positiva. Esto significa que si el modelo falla en predecir un accidente cerebrovascular (falso negativo), el error se magnifica y el gradiente penaliza los pesos de forma mucho más agresiva.

---

## 3. Árbol de Decisión Rápido (`SimpleFastDecisionTree`)

- **Implementación en el Proyecto:**
  - Una implementación estructurada desde cero en Python puro. Para mejorar la eficiencia, en lugar de evaluar todos los valores únicos como posibles cortes en cada nodo, evalúa **umbrales basados en cuantiles** (ej. percentiles del 5% al 95%).
  - **Criterio de Impureza Modificado:** Se codificó una **Impureza de Gini Ponderada**, lo que permite pasarle un parámetro `pos_weight`. Así, un nodo se evalúa como muy impuro si clasifica erróneamente instancias de la clase minoritaria (stroke).
  - A diferencia de los árboles convencionales que aplican una regla de mayoría simple en sus hojas, este retorna la proporción cruda de clases como una probabilidad (`predict_proba`). Esto permite aplicar un umbral de decisión más conservador (ej. $0.35$) para sacrificar algo de Precisión a cambio de ganar fuertemente en Recall.

---

## 4. Perceptrón Multicapa con JAX (`MultilayerPerceptronJAX`)

- **Implementación en el Proyecto:**
  - Red neuronal artificial construida sobre las primitivas de matrices de JAX, constando de una capa oculta de 32 neuronas.
  - Utiliza inicialización escalar `He` para los pesos, activación no lineal **ReLU** en la capa oculta y activación final **Sigmoide**.
  - El proceso de *Backpropagation* no está hardcodeado, sino que se deriva automáticamente a nivel de red neuronal utilizando `jax.value_and_grad`. 
  - Al igual que la regresión logística, la función de pérdida que optimiza la red está modificada (Weighted BCE) para balancear la asimetría clínica de las etiquetas.

---

## 5. AdaBoost con JAX (`AdaBoostJAX`)

- **Implementación en el Proyecto:**
  - **Ensamble Heterogéneo:** En lugar de usar exclusivamente *Decision Stumps* como dicta el algoritmo tradicional, esta versión fue reescrita para utilizar todos los clasificadores desarrollados en el proyecto (**Regresión Lineal, Regresión Logística, Árbol de Decisión y Perceptrón Multicapa**) iterando a través de ellos como modelos base en cada estimación.
  - **Resampling por Pesos:** Dado que no todos los modelos base admiten de forma natural ponderaciones por instancia durante el entrenamiento, se implementó un mecanismo dinámico de muestreo con reemplazo (resampling) gobernado por la distribución de pesos de AdaBoost en cada iteración.
  - **Manejo de Desbalance Nativo:** Las muestras se inicializan asimétricamente: la clase positiva recibe un peso inicial de `1 / (2 * n_pos)` y la negativa `1 / (2 * n_neg)`. 
  - Extrae el voto ponderado acumulado de todos los clasificadores iterados y lo transforma a una pseudo-probabilidad usando una función sigmoide (`predict_proba`). Esto permite trazar su desempeño mediante umbrales variables, útil en las Curvas Precision-Recall.

---

## Estrategia de Evaluación Clínica

Debido al enfoque de este proyecto (salud humana y detección de strokes), **la métrica de mayor impacto es el Recall (Sensibilidad)**. Los Falsos Negativos pueden implicar que una persona en alto riesgo no sea advertida, lo cual es éticamente inadmisible en un pipeline clínico.

Por lo tanto:
1. Las arquitecturas JAX penalizan internamente los falsos negativos durante la propia derivación matemática (Weighted BCE / Weighted Gini).
2. Durante la inferencia sobre el set de prueba, los umbrales de decisión (`umbral_smote`) han sido recalibrados (ej. bajando el punto de corte a $0.35$).
3. La evaluación del pipeline se enfoca en las **Curvas Precision-Recall** en detrimento de las Curvas ROC, ya que las Curvas ROC otorgan mucho peso a los Verdaderos Negativos (que son el 95% de este dataset) lo cual da un diagnóstico excesivamente optimista del desempeño real del clasificador.
4. Todos los entrenamientos, pesos e hiperparámetros se guardan versionados dentro del servidor de registro **MLflow**.
