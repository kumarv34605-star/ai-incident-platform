from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/telemetry.csv")

EXPECTED_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "request_rate",
    "latency_ms",
    "error_rate",
    "db_connections",
    "http_5xx",
    "incident",
]


def validate_data(df: pd.DataFrame) -> None:
    """Validate telemetry dataset before it enters the ML pipeline."""

    # 1. Check schema
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # 2. Check missing values
    if df[EXPECTED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    # 3. Check CPU range
    if not df["cpu_usage"].between(0, 100).all():
        raise ValueError("CPU usage contains invalid values.")

    # 4. Check memory range
    if not df["memory_usage"].between(0, 100).all():
        raise ValueError("Memory usage contains invalid values.")

    # 5. Check latency
    if not (df["latency_ms"] > 0).all():
        raise ValueError("Latency must be greater than zero.")

    # 6. Check target
    if not df["incident"].isin([0, 1]).all():
        raise ValueError("Incident target must contain only 0 or 1.")

    print("Data validation passed.")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded dataset: {DATA_PATH}")
    print(f"Shape: {df.shape}")

    validate_data(df)


if __name__ == "__main__":
    main()