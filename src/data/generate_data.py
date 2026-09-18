import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_SAMPLES = 10_000


def generate_telemetry(num_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """Generate synthetic production telemetry."""

    rng = np.random.default_rng(RANDOM_SEED)

    data = pd.DataFrame(
        {
            "cpu_usage": rng.normal(50, 10, num_samples),
            "memory_usage": rng.normal(60, 8, num_samples),
            "request_rate": rng.normal(500, 80, num_samples),
            "latency_ms": rng.normal(150, 30, num_samples),
            "error_rate": rng.normal(0.5, 0.2, num_samples),
            "db_connections": rng.normal(40, 8, num_samples),
            "http_5xx": rng.normal(5, 2, num_samples),
        }
    )

    # Keep values within realistic ranges
    data["cpu_usage"] = data["cpu_usage"].clip(0, 100)
    data["memory_usage"] = data["memory_usage"].clip(0, 100)
    data["request_rate"] = data["request_rate"].clip(0)
    data["latency_ms"] = data["latency_ms"].clip(1)
    data["error_rate"] = data["error_rate"].clip(0)
    data["db_connections"] = data["db_connections"].clip(0)
    data["http_5xx"] = data["http_5xx"].clip(0)

    # Initially, everything is considered normal
    data["incident"] = 0

    return data

def inject_incidents(
    data: pd.DataFrame,
    incident_ratio: float = 0.10,
) -> pd.DataFrame:
    """Inject realistic production incidents into telemetry."""

    rng = np.random.default_rng(RANDOM_SEED)

    num_incidents = int(len(data) * incident_ratio)

    incident_indices = rng.choice(
        data.index,
        size=num_incidents,
        replace=False,
    )

    for index in incident_indices:

        incident_type = rng.choice(
            [
                "cpu_overload",
                "memory_pressure",
                "database_saturation",
                "error_spike",
            ]
        )

        if incident_type == "cpu_overload":
            data.loc[index, "cpu_usage"] = rng.uniform(85, 100)
            data.loc[index, "latency_ms"] *= rng.uniform(2, 4)

        elif incident_type == "memory_pressure":
            data.loc[index, "memory_usage"] = rng.uniform(85, 100)
            data.loc[index, "latency_ms"] *= rng.uniform(1.5, 3)

        elif incident_type == "database_saturation":
            data.loc[index, "db_connections"] = rng.uniform(85, 100)
            data.loc[index, "latency_ms"] *= rng.uniform(2, 5)
            data.loc[index, "error_rate"] *= rng.uniform(3, 8)

        elif incident_type == "error_spike":
            data.loc[index, "error_rate"] = rng.uniform(10, 30)
            data.loc[index, "http_5xx"] = rng.uniform(100, 500)
            data.loc[index, "latency_ms"] *= rng.uniform(2, 4)

        data.loc[index, "incident"] = 1

    return data

if __name__ == "__main__":
    df = generate_telemetry()
    df = inject_incidents(df)

    output_path = "data/raw/telemetry.csv"

    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nIncident distribution:")
    print(df["incident"].value_counts())