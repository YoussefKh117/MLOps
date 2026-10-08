import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import numpy as np


def execute_run(
    run_name,
    max_depth,
    learning_rate,
    X_train,
    X_val,
    y_train,
    y_val
):
    with mlflow.start_run(run_name=run_name):

        # Parameters
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        }

        mlflow.log_params(params)

        # Convert data to XGBoost DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train model
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[
                (dtrain, "train"),
                (dval, "val")
            ],
            verbose_eval=False
        )

        # Predictions
        y_pred = model.predict(dval)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Log metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost-model"
        )

        print(f"\n{run_name}")
        print(f"max_depth     = {max_depth}")
        print(f"learning_rate = {learning_rate}")
        print(f"RMSE          = {rmse:.4f}")
        print(f"MAE           = {mae:.4f}")
        print(f"R2            = {r2:.4f}")


def main():

    # MLflow tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # MLflow experiment
    mlflow.set_experiment("California_Housing_Optimization")

    # Load California Housing dataset
    data = fetch_california_housing()

    X = data.data
    y = data.target

    # 80% training / 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Three required experiments
    configs = [
        ("Run 1", 3, 0.1),
        ("Run 2", 5, 0.05),
        ("Run 3", 7, 0.01)
    ]

    # Execute experiments
    for run_name, max_depth, learning_rate in configs:
        execute_run(
            run_name,
            max_depth,
            learning_rate,
            X_train,
            X_val,
            y_train,
            y_val
        )


if __name__ == "__main__":
    main()