
import os
import importlib

mlflow = importlib.import_module("mlflow")
importlib.import_module("mlflow.sklearn")

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

MODEL_URI = "models:/iris-classifier-prod/Staging"

print("Loading registered model...")
model = mlflow.sklearn.load_model(MODEL_URI)

print("Model loaded successfully!")
print("Model type:", type(model))
print(model)
