from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/telemetry.csv")


FEATURES = [
    "cpu_usage",
    "memory_usage",
    "request_rate",
    "latency_ms",
    "error_rate",
    "db_connections",
    "http_5xx",
]

TARGET = "incident"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def prepare_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


def main():
    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()