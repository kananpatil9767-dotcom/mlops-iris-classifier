
from importlib import import_module

try:
    mlflow = import_module("mlflow")
    MlflowClient = import_module("mlflow.tracking").MlflowClient
except ModuleNotFoundError as exc:
    if exc.name == "mlflow":
        raise SystemExit(
            "MLflow is not installed. Install it with: pip install mlflow"
        ) from exc
    raise

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiment = client.get_experiment_by_name(
    "iris-hyperparameter-tuning"
)

if experiment is None:
    print("Experiment not found. Run the model scripts first.")
    raise SystemExit(1)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id]
)

print("\nHYPERPARAMETER TUNING COMPARISON")
print("-" * 75)
print(f"{'Run Name':<35}{'CV F1 Score':<20}{'Test Accuracy':<15}")
print("-" * 75)

for run in runs:
    name = run.data.tags.get("mlflow.runName", "unknown")
    metrics = run.data.metrics

    cv_score = metrics.get(
        "cv_f1_macro_mean",
        metrics.get("best_cv_f1_macro", 0)
    )
    accuracy = metrics.get("test_accuracy", 0)

    print(f"{name:<35}{cv_score:<20.4f}{accuracy:<15.4f}")

print("-" * 75)
