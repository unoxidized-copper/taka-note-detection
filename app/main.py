"""
Endpoints:
    GET  /         -> service metadata + class list
    GET  /health   -> health check
    POST /predict  -> main prediction api
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from app.inference import get_model, predict

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
        print("[INFO] Model loaded successfully.")
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
    yield


app = FastAPI(
    title="Bangladeshi Taka Note Detection API",
    description="Detects Bangladeshi taka using a YOLOv11 model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    try:
        classes = list(get_model().names.values())
    except FileNotFoundError:
        classes = []
    return {
        "service": "Bangladeshi Taka Note Detection API",
        "status": "ok",
        "model_classes": classes,
        "usage": "POST an image file to /predict.",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   "Please upload a JPEG or PNG image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        result = predict(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422,
                            detail=f"Could not process image: {exc}")

    result["filename"] = file.filename
    return JSONResponse(status_code=200, content=result)