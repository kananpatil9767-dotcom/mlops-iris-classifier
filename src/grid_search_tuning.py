
from importlib import import_module

mlflow = import_module("mlflow")
pd = import_module("pandas")
RandomForestClassifier = import_module("sklearn.ensemble").RandomForestClassifier
LabelEncoder = import_module("sklearn.preprocessing").LabelEncoder

_model_selection = import_module("sklearn.model_selection")
GridSearchCV = _model_selection.GridSearchCV
train_test_split = _model_selection.train_test_split

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}


def run_grid_search():
    df = pd.read_csv("data/processed/iris_features.csv")

    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = LabelEncoder().fit_transform(df["species"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    with mlflow.start_run(run_name="grid_search_random_forest"):
        model = RandomForestClassifier(random_state=42)

        grid = GridSearchCV(
            model,
            PARAM_GRID,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
        )

        grid.fit(X_train, y_train)

        accuracy = grid.best_estimator_.score(X_test, y_test)

        mlflow.log_param("search_type", "GridSearchCV")
        mlflow.log_metric("best_cv_f1_macro", grid.best_score_)
        mlflow.log_metric("test_accuracy", accuracy)

        for name, value in grid.best_params_.items():
            mlflow.log_param(f"best_{name}", str(value))

        print("Grid Search Results")
        print("Best Parameters:", grid.best_params_)
        print(f"Best CV F1 Score: {grid.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    run_grid_search()
