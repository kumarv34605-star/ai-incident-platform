import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


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


model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "incident-detection-api",
    }


@app.post("/predict")
def predict_incident(telemetry: TelemetryRequest):
    telemetry_data = pd.DataFrame(
        [telemetry.model_dump()],
        columns=FEATURES,
    )

    prediction = model.predict(telemetry_data)[0]

    return {
        "incident": int(prediction),
        "status": (
            "incident_detected"
            if prediction == 1
            else "normal"
        ),
    }