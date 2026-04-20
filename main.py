from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app import app

BASE_DIR = Path(__file__).resolve().parent

# Mount the static directory for serving images
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(BASE_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
