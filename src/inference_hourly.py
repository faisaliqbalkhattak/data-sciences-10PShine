"""Inference for the hourly 30-output AQI forecast contract.

The inference path intentionally uses the same feature builder as training and
loads only an artifact whose manifest declares the same feature schema. It
returns 24 point forecasts followed by four six-hour means and two twelve-hour
means. Inputs must end at the forecast origin and contain no missing current
features; future observations are never read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from . import config
from .train_hourly import (
    BLOCK_TARGET_COLUMNS,
    HOURLY_TARGET_COLUMNS,
    TARGET_COLUMNS,
    build_hourly_training_frame,
)

MANIFEST_PATH = config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_models.json"
RIDGE_ARTIFACT = config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_ridge.joblib"
MODEL_ARTIFACTS = {
    "ridge": RIDGE_ARTIFACT,
    "random_forest": config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_rf.joblib",
    "xgboost": config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_xgb.joblib",
}


def _load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Hourly model manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_feature_schema(features: pd.DataFrame, manifest: dict) -> list[str]:
    columns = manifest.get("_meta", {}).get("feature_columns")
    if not columns:
        raise ValueError("Hourly manifest has no feature-column contract.")
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ValueError(f"Inference features are missing trained columns: {missing}")
    latest = features[columns].iloc[-1]
    if latest.isna().any() or not np.isfinite(latest.to_numpy(dtype=float)).all():
        raise ValueError("The latest hourly feature row contains missing or non-finite values.")
    return columns


def forecast_rows(
    origin: pd.Timestamp, predictions: np.ndarray
) -> pd.DataFrame:
    """Label a 30-value vector with timestamps and interval semantics."""
    values = np.asarray(predictions, dtype=float).reshape(-1)
    if len(values) != len(TARGET_COLUMNS):
        raise ValueError(f"Expected {len(TARGET_COLUMNS)} predictions, got {len(values)}.")

    rows = []
    for offset, column in enumerate(HOURLY_TARGET_COLUMNS, start=1):
        target_time = origin + pd.Timedelta(hours=offset)
        rows.append(
            {
                "forecast_origin": origin,
                "output": column,
                "kind": "point",
                "start_time": target_time,
                "end_time": target_time,
                "value": values[offset - 1],
            }
        )
    block_specs = [
        (25, 30, BLOCK_TARGET_COLUMNS[0], "six_hour_mean"),
        (31, 36, BLOCK_TARGET_COLUMNS[1], "six_hour_mean"),
        (37, 42, BLOCK_TARGET_COLUMNS[2], "six_hour_mean"),
        (43, 48, BLOCK_TARGET_COLUMNS[3], "six_hour_mean"),
        (49, 60, BLOCK_TARGET_COLUMNS[4], "twelve_hour_mean"),
        (61, 72, BLOCK_TARGET_COLUMNS[5], "twelve_hour_mean"),
    ]
    for index, (start, end, column, kind) in enumerate(block_specs, start=24):
        rows.append(
            {
                "forecast_origin": origin,
                "output": column,
                "kind": kind,
                "start_time": origin + pd.Timedelta(hours=start),
                "end_time": origin + pd.Timedelta(hours=end),
                "value": values[index],
            }
        )
    return pd.DataFrame(rows)


def predict_latest(
    hourly: pd.DataFrame,
    manifest_path: Path = MANIFEST_PATH,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Predict from the last completed hourly row using the selected model.

    If model_path is not provided, loads the champion model from MLflow registry
    (falling back to the local artifact specified in the manifest).
    """
    manifest = _load_manifest(manifest_path)
    selected = manifest.get("_meta", {}).get("selected_model_by_group", {})
    best_model = selected.get("hourly_points", "ridge")

    if model_path is None:
        # Try MLflow registry first, then local artifact
        try:
            from .model_registry import load_hourly_model

            model = load_hourly_model()
        except Exception:
            # Fallback to local artifact
            local_path = MODEL_ARTIFACTS.get(best_model, RIDGE_ARTIFACT)
            if not local_path.exists():
                raise FileNotFoundError(
                    f"No model found: MLflow registry empty and {local_path} missing. "
                    f"Run `python -m src.train_hourly --store` first."
                )
            model = joblib.load(local_path)
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Hourly model artifact not found: {model_path}")
        model = joblib.load(model_path)

    features = build_hourly_training_frame(hourly, include_targets=False)
    columns = _validate_feature_schema(features, manifest)
    prediction = model.predict(features[columns].iloc[[-1]])
    origin = pd.Timestamp(features.index[-1])
    return forecast_rows(origin, prediction[0])


def predict_latest_from_csv(path: Path) -> pd.DataFrame:
    """Load a raw hourly feature CSV and run :func:`predict_latest`."""
    hourly = pd.read_csv(path, parse_dates=["time"])
    return predict_latest(hourly)


if __name__ == "__main__":
    source = config.DATA_PROCESSED_DIR / "karak_aqi_open_meteo_hourly_features.csv"
    print(predict_latest_from_csv(source).to_string(index=False))
