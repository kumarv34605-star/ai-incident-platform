import yaml
from pathlib import Path
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/telemetry.csv")
CONFIG_PATH = Path("configs/training.yaml")


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


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def prepare_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def train_model(X_train, y_train, config):
    model_config = config["model"]

    model = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        random_state=model_config["random_state"],
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def save_confusion_matrix(model, X_test, y_test):
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title("Incident Classification Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()

    output_path = Path("confusion_matrix.png")
    fig.savefig(output_path)
    plt.close(fig)

    return output_path

def main():
    config = load_config()

    mlflow.set_experiment("incident-classification")

    with mlflow.start_run():

        # Load data
        df = load_data()

        # Prepare features and target
        X, y = prepare_data(df)

        # Training configuration
        training_config = config["training"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=training_config["test_size"],
            random_state=training_config["random_state"],
            stratify=y,
        )

        # Log configuration to MLflow
        mlflow.log_params(
            {
                "model_type": config["model"]["type"],
                "n_estimators": config["model"]["n_estimators"],
                "model_random_state": config["model"]["random_state"],
                "test_size": config["training"]["test_size"],
                "split_random_state": config["training"]["random_state"],
            }
        )

        # Train model
        model = train_model(
            X_train,
            y_train,
            config,
        )

        # Evaluate model
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )
        # Log metrics
        mlflow.log_metrics(metrics)

        # Create and log confusion matrix
        confusion_matrix_path = save_confusion_matrix(
            model,
            X_test,
            y_test,
        )

        mlflow.log_artifact(
            confusion_matrix_path,
            artifact_path="evaluation",
        )

        # Log trained model
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="incident-classifier",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        print("\nMLflow run completed.")
        print(
            f"Run ID: {mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    main()