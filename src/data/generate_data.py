from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_SAMPLES = 10_000
INCIDENT_RATIO = 0.10

OUTPUT_PATH = Path("data/raw/telemetry.csv")


def generate_telemetry(num_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """Generate realistic synthetic production telemetry."""

    rng = np.random.default_rng(RANDOM_SEED)

    cpu = rng.normal(50, 12, num_samples)
    memory = rng.normal(60, 10, num_samples)
    request_rate = rng.normal(500, 100, num_samples)

    # Latency has some relationship with CPU and traffic.
    latency = (
        100
        + cpu * 0.8
        + request_rate * 0.05
        + rng.normal(0, 25, num_samples)
    )

    error_rate = (
        0.3
        + (latency / 1000) * 0.5
        + rng.normal(0, 0.15, num_samples)
    )

    db_connections = (
        30
        + request_rate * 0.04
        + rng.normal(0, 6, num_samples)
    )

    http_5xx = (
        error_rate * 8
        + rng.normal(2, 1, num_samples)
    )

    data = pd.DataFrame(
        {
            "cpu_usage": cpu,
            "memory_usage": memory,
            "request_rate": request_rate,
            "latency_ms": latency,
            "error_rate": error_rate,
            "db_connections": db_connections,
            "http_5xx": http_5xx,
        }
    )

    # Keep values realistic.
    data["cpu_usage"] = data["cpu_usage"].clip(0, 100)
    data["memory_usage"] = data["memory_usage"].clip(0, 100)
    data["request_rate"] = data["request_rate"].clip(10)
    data["latency_ms"] = data["latency_ms"].clip(10)
    data["error_rate"] = data["error_rate"].clip(0)
    data["db_connections"] = data["db_connections"].clip(1)
    data["http_5xx"] = data["http_5xx"].clip(0)

    data["incident"] = 0
    data["incident_type"] = "normal"
    data["severity"] = "none"

    return data


def inject_incidents(
    data: pd.DataFrame,
    incident_ratio: float = INCIDENT_RATIO,
) -> pd.DataFrame:
    """Inject realistic production incidents."""

    rng = np.random.default_rng(RANDOM_SEED)

    num_incidents = int(len(data) * incident_ratio)

    incident_indices = rng.choice(
        data.index,
        size=num_incidents,
        replace=False,
    )

    incident_types = [
        "cpu_overload",
        "memory_pressure",
        "database_saturation",
        "error_spike",
        "latency_degradation",
    ]

    severities = ["low", "medium", "high"]

    for index in incident_indices:

        incident_type = rng.choice(incident_types)
        severity = rng.choice(severities, p=[0.5, 0.35, 0.15])

        if severity == "low":
            multiplier = 1.2
        elif severity == "medium":
            multiplier = 1.5
        else:
            multiplier = 2.0

        if incident_type == "cpu_overload":
            data.loc[index, "cpu_usage"] = min(
                data.loc[index, "cpu_usage"] * multiplier,
                100,
            )
            data.loc[index, "latency_ms"] *= multiplier

        elif incident_type == "memory_pressure":
            data.loc[index, "memory_usage"] = min(
                data.loc[index, "memory_usage"] * multiplier,
                100,
            )
            data.loc[index, "latency_ms"] *= 1.3

        elif incident_type == "database_saturation":
            data.loc[index, "db_connections"] *= multiplier
            data.loc[index, "latency_ms"] *= 1.5
            data.loc[index, "error_rate"] *= 1.5

        elif incident_type == "error_spike":
            data.loc[index, "error_rate"] *= 2.5
            data.loc[index, "http_5xx"] *= 3

        elif incident_type == "latency_degradation":
            data.loc[index, "latency_ms"] *= 2
            data.loc[index, "error_rate"] *= 1.3

        data.loc[index, "incident"] = 1
        data.loc[index, "incident_type"] = incident_type
        data.loc[index, "severity"] = severity

    return data


def main():
    df = generate_telemetry()
    df = inject_incidents(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")

    print("\nIncident distribution:")
    print(df["incident"].value_counts())

    print("\nIncident types:")
    print(df["incident_type"].value_counts())

    print("\nSeverity distribution:")
    print(df["severity"].value_counts())


if __name__ == "__main__":
    main()