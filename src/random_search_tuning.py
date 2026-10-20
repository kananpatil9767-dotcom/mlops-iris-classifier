
import importlib
import pandas as pd

LabelEncoder = importlib.import_module("sklearn.preprocessing").LabelEncoder

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

PARAM_DIST = {
    "n_estimators": list(range(50, 300)),
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": list(range(2, 15)),
    "max_features": ["sqrt", "log2"],
}


def run_random_search():
    model_selection = importlib.import_module("sklearn.model_selection")
    RandomizedSearchCV = model_selection.RandomizedSearchCV
    train_test_split = model_selection.train_test_split

    RandomForestClassifier = importlib.import_module(
        "sklearn.ensemble"
    ).RandomForestClassifier

    try:
        mlflow = importlib.import_module("mlflow")
    except ModuleNotFoundError as exc:
        if exc.name == "mlflow":
            raise RuntimeError(
                "MLflow is required; install it with 'pip install mlflow'."
            ) from exc
        raise

    df = pd.read_csv("data/processed/iris_features.csv")

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = LabelEncoder().fit_transform(df["species"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    with mlflow.start_run(run_name="random_search_random_forest"):
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=PARAM_DIST,
            n_iter=30,
            cv=5,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)
        accuracy = search.best_estimator_.score(X_test, y_test)

        mlflow.log_param("search_type", "RandomizedSearchCV")
        mlflow.log_param("n_iter", 30)
        mlflow.log_metric("best_cv_f1_macro", search.best_score_)
        mlflow.log_metric("test_accuracy", accuracy)

        for name, value in search.best_params_.items():
            mlflow.log_param(f"best_{name}", str(value))

        print("Random Search Results")
        print("Best Parameters:", search.best_params_)
        print(f"Best CV F1 Score: {search.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    run_random_search()
