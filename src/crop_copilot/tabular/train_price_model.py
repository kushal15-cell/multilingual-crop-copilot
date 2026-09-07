from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crop_copilot.tabular.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FORECAST_HORIZON_DAYS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_features,
)


def temporal_split(frame: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = frame["date"].max() - pd.Timedelta(days=test_days)
    purged_train_end = cutoff - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
    labeled = frame.dropna(subset=[TARGET_COLUMN])
    train = labeled[labeled["date"] <= purged_train_end].copy()
    test = labeled[labeled["date"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Not enough chronological data for the requested holdout")
    return train, test


def train(input_path: Path, output_path: Path, test_days: int = 30) -> dict[str, float]:
    frame = build_features(pd.read_csv(input_path))
    train_frame, test_frame = temporal_split(frame, test_days)
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.06,
                    max_iter=300,
                    max_leaf_nodes=24,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    predictions = pipeline.predict(test_frame[FEATURE_COLUMNS])
    metrics = {
        "mae": float(mean_absolute_error(test_frame[TARGET_COLUMN], predictions)),
        "rmse": float(np.sqrt(mean_squared_error(test_frame[TARGET_COLUMN], predictions))),
        "r2": float(r2_score(test_frame[TARGET_COLUMN], predictions)),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "train_feature_cutoff": str(train_frame["date"].max().date()),
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "trained_until": str(train_frame["date"].max()),
            "metrics": metrics,
        },
        output_path,
    )
    (output_path.parent / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError('Install MLflow with: pip install -e ".[mlops]"') from exc
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "crop-copilot-price"))
        with mlflow.start_run():
            mlflow.log_params(
                {"test_days": test_days, "forecast_horizon_days": FORECAST_HORIZON_DAYS}
            )
            mlflow.log_metrics(
                {key: value for key, value in metrics.items() if isinstance(value, int | float)}
            )
            mlflow.log_artifact(str(output_path), artifact_path="price")
            mlflow.log_artifact(str(output_path.parent / "metrics.json"), artifact_path="price")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-days", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output, args.test_days), indent=2))


if __name__ == "__main__":
    main()
