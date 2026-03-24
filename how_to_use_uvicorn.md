# How to use Uvicorn with this Project

This project uses **FastAPI** to serve a machine learning model. To run a FastAPI application, we use an ASGI server called **Uvicorn**.

## Prerequisites

Before running the server, you should create and activate a Python virtual environment to install the dependencies:

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
```

Additionally, ensure that the machine learning model has been trained and saved as `modelo_entrenado.pkl`. If not, run your Jupyter Notebook (`ML_v1.4.ipynb`) to generate it.

## Running the Application

There are two main files that can be run with Uvicorn in this project: `main.py` and `app.py`.

### 1. Running `app.py` (The ML Prediction API)

The `app.py` file contains the actual machine learning prediction endpoints (e.g., `/predict`). To run this API, execute the following command in your terminal:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- `app:app`: The first `app` refers to the file `app.py`. The second `app` refers to the FastAPI instance (`app = FastAPI(...)`) inside that file.
- `--reload`: Enables hot-reloading so the server automatically restarts when you make code changes. (Useful for development).
- `--host 0.0.0.0`: Makes the server accessible on your local network.
- `--port 8000`: Sets the port where the server will listen (default is 8000).

Once running, you can access the interactive API documentation (Swagger UI) by navigating to:
[http://127.0.0.0:8000/docs](http://127.0.0.0:8000/docs)

### 2. Running `main.py` (The explicitly hosted entrypoint)

`main.py` is written to be executed directly as a Python script, since it contains the following block:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="[IP_ADDRESS]", port=8000, reload=True)
```

To run it, simply execute:

```bash
python main.py
```

*Note: You may need to replace `[IP_ADDRESS]` with `127.0.0.1` or `0.0.0.0` in `main.py` for it to work locally without issues.*

Alternatively, you can run `main.py` directly through the uvicorn CLI in the same way as `app.py`:

```bash
uvicorn main:app --reload
```
