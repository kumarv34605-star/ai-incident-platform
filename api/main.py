from fastapi import FastAPI
from pydantic import BaseModel, Field


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


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "incident-detection-api",
    }


@app.post("/predict")
def predict_incident(telemetry: TelemetryRequest):
    return {
        "message": "Telemetry received successfully",
        "telemetry": telemetry.model_dump(),
    }