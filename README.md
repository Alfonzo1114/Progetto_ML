# Proyecto de Machine Learning para predicción de Embolias cerebrovasculares

Este proyecto implementa un sistema completo de predicción de accidentes cerebrovasculares utilizando modelos de Machine Learning avanzados desarrollados desde cero en **JAX**, un pipeline de datos robusto con **Metaflow** y una interfaz web para predicciones en tiempo real.

## Características Principales

*   **Modelos Multi-Algoritmo**: Implementación desde cero (sin bibliotecas de alto nivel como Scikit-Learn para el core) de:
    *   Regresión Logística Ponderada.
    *   Perceptrón Multicapa (MLP) robusto con JAX.
    *   AdaBoost con Decision Stumps.
    *   Árboles de Decisión (Classification Tree).
*   **Feature Engineering Avanzado**: Creación de variables clínicas críticas como:
    *   Riesgo por edad avanzada (is_senior).
    *   Índice de comorbilidad cardiovascular.
    *   Riesgo combinado por tabaquismo.
    *   Categorización médica de BMI.
*   **Balanceo de Datos**: Uso de **SMOTE** para manejar el desbalance severo del dataset (~95% sano vs 5% stroke).
*   **Pipeline de Producción**: Orquestación con **Metaflow** para entrenamiento paralelo y reproducible.
*   **Despliegue (API & UI)**: Backend en **FastAPI** y frontend interactivo.

---

## 📂 Estructura del Repositorio (Archivos Relevantes)

*   `ML_v1.5.ipynb`: Notebook principal con todo el análisis, entrenamiento y métricas.
*   `metaflow_v1.2.py`: Pipeline de producción para orquestar el flujo de datos y entrenamiento.
*   `schema.py`: Lógica central de inferencia y procesamiento de características para la API.
*   `app.py` / `main.py`: Servidor FastAPI para exponer los modelos vía Uvicorn.
*   `index.html`: Interfaz web interactiva para realizar predicciones.
*   `modelo_entregado.pkl`: Pesos y parámetros del modelo final entrenado.
*   `requirements.txt`: Dependencias del proyecto.
*   `Project_First_Phase.pdf`: Documentación del proyecto.

---

## Instalación y Configuración

1. **Clonar el repositorio**:
   ```bash
   git clone <url-del-repo>
   cd Progetto_ML-1
   ```

2. **Crear y activar entorno virtual**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Cómo Ejecutar

### 1. Ejecutar el Pipeline (Entrenamiento)
Para re-entrenar todos los modelos y generar el reporte de métricas:
```bash
python metaflow_v1.2.py run
```

### 2. Levantar el Backend (API)
Inicia el servidor de predicción:
```bash
python main.py
# O directamente con uvicorn:
# uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 3. Levantar la Interfaz Web
En una nueva terminal, sirve el archivo HTML:
```bash
python3 -m http.server 5500
```
Luego abre [http://127.0.0.1:5500/index.html](http://127.0.0.1:5500/index.html) en tu navegador.

---

## 📊 Métricas y Selección de Modelos

En este proyecto se prioriza el **Recall (Sensibilidad)** debido a la naturaleza crítica de los accidentes cerebrovasculares (un Falso Negativo puede ser fatal). Sin embargo, se utiliza el **F1-Score** para seleccionar el mejor modelo (MLP/AdaBoost) por su capacidad de equilibrar la detección con la precisión clínica.

---
*Desarrollado como parte del Proyecto de Machine Learning - Fase 1.*
