import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = "incident-classifier"
MODEL_VERSION = "1"

FEATURES = [
    "cpu_usage",
    "memory_usage",
    "request_rate",
    "latency_ms",
    "error_rate",
    "db_connections",
    "http_5xx",
]


app = FastAPI(
    title="AI Incident Intelligence API",
    description="API for ML-based incident detection",
    version="1.0.0",
)


class TelemetryRequest(BaseModel):
    cpu_usage: float = Field(ge=0, le=100)
    memory_usage: float = Field(ge=0, le=100)
    request_rate: float = Field(ge=0)
    latency_ms: float = Field(gt=0)
    error_rate: float = Field(ge=0)
    db_connections: float = Field(ge=0)
    http_5xx: float = Field(ge=0)


class PredictionResponse(BaseModel):
    incident: int
    status: str


model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

model = None

try:
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("ML model loaded successfully: %s", model_uri)
except Exception:
    logger.exception("Failed to load ML model: %s", model_uri)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "incident-detection-api",
    }


@app.get("/ready")
def readiness_check():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not ready",
        )

    return {
        "status": "ready",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_incident(telemetry: TelemetryRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not ready",
        )

    telemetry_data = pd.DataFrame(
        [telemetry.model_dump()],
        columns=FEATURES,
    )

    try:
        prediction = model.predict(telemetry_data)[0]
    except Exception:
        logger.exception("Model inference failed")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed",
        )

    return {
        "incident": int(prediction),
        "status": (
            "incident_detected"
            if prediction == 1
            else "normal"
        ),
    }