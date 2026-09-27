import mlflow
import mlflow.sklearn
import pandas as pd

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


def main():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

    print(f"Loading model: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    print("Model loaded successfully!")

    sample_telemetry = pd.DataFrame(
        [[
            95,
            75,
            900,
            700,
            4,
            90,
            35,
        ]],
        columns=FEATURES,
    )

    prediction = model.predict(sample_telemetry)

    print(f"Prediction: {prediction[0]}")

    if prediction[0] == 1:
        print("🚨 INCIDENT DETECTED")
    else:
        print("✅ NORMAL")


if __name__ == "__main__":
    main()