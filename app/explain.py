"""SHAP explanation of the hourly Ridge forecast.

The selected hourly model is a ``StandardScaler + Ridge`` pipeline. Because it
is linear in the scaled space, SHAP attribution is exact and cheap via
``shap.LinearExplainer`` (interventional expectations over the observed
feature distribution). The dashboard and API both call
:func:`explain_latest_origin` so the explanation never re-implements the
feature contract.

If LinearExplainer is unavailable for any reason, a coefficient-based
attribution (``scaled row * ridge coefficient``) is returned with an explicit
``method`` marker so the UI can label it honestly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

RIDGE_ARTIFACT = config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_ridge.joblib"
HOURLY_MANIFEST = config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_models.json"


def _feature_columns() -> list[str]:
    import json

    manifest = json.loads(HOURLY_MANIFEST.read_text(encoding="utf-8"))
    columns = manifest.get("_meta", {}).get("feature_columns", [])
    if not columns:
        raise ValueError("Hourly manifest has no feature-column contract.")
    return columns


def _coefficient_attribution(pipeline, X_scaled_row: np.ndarray) -> np.ndarray:
    ridge = pipeline.named_steps["model"]
    coef = np.asarray(ridge.coef_)
    # coef shape: (n_outputs, n_features) for multi-output Ridge.
    attribution = X_scaled_row * coef
    return attribution


def explain_latest_origin(
    features: pd.DataFrame, output_index: int = 0, reference_rows: int = 500
) -> dict:
    """Explain the latest forecast origin for one output of the Ridge model.

    ``features`` is the processed hourly frame (from
    ``train_hourly.build_hourly_training_frame(..., include_targets=False)``).
    ``output_index`` selects the output column (0 = t+1h, 23 = t+24h, 28 = the
    first twelve-hour mean).
    """
    import shap

    if not RIDGE_ARTIFACT.exists():
        raise FileNotFoundError(
            "Hourly Ridge artifact not found; run `python -m src.train_hourly` first."
        )
    pipeline = joblib.load(RIDGE_ARTIFACT)
    scaler = pipeline.named_steps["scale"]
    ridge = pipeline.named_steps["model"]

    columns = _feature_columns()
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ValueError(f"Explanation features missing trained columns: {missing}")
    X = features[columns].tail(reference_rows)
    X_scaled = scaler.transform(X)

    try:
        explainer = shap.LinearExplainer(
            ridge, masker=shap.maskers.Independent(X_scaled)
        )
        raw_values = np.asarray(explainer.shap_values(X_scaled[-1:]))
        expected = np.asarray(explainer.expected_value)
        method = "linear_shap"
    except Exception as exc:  # noqa: BLE001 - fall back to coefficient attribution
        raw_values = _coefficient_attribution(pipeline, X_scaled[-1:])
        expected = np.asarray(ridge.intercept_)
        method = f"coefficient_fallback ({type(exc).__name__})"

    # raw_values shape: (1, n_features) for single-output, (1, n_outputs, n_features)
    # for multi-output Ridge.
    if raw_values.ndim == 3:
        row_values = raw_values[0, output_index, :]
        expected_value = float(np.ravel(expected)[output_index]) if expected.ndim else float(expected)
    elif raw_values.ndim == 2 and raw_values.shape[0] == 1:
        row_values = raw_values[0]
        expected_value = float(expected)
    else:
        row_values = raw_values.reshape(-1)
        expected_value = float(np.ravel(expected)[0])

    contribution = float(np.sum(row_values))
    rows = []
    for name, value, shap_value in zip(
        columns, X[columns].iloc[-1].to_numpy(), row_values
    ):
        rows.append(
            {
                "feature": str(name),
                "value": float(value),
                "shap": float(shap_value),
            }
        )
    rows.sort(key=lambda row: abs(row["shap"]), reverse=True)
    return {
        "method": method,
        "output_index": int(output_index),
        "output_column": _output_name(output_index),
        "expected_value": expected_value,
        "shap_sum": contribution,
        "prediction_base_plus_shap": expected_value + contribution,
        "features": rows,
    }


def _output_name(output_index: int) -> str:
    from src.train_hourly import TARGET_COLUMNS

    if 0 <= output_index < len(TARGET_COLUMNS):
        return TARGET_COLUMNS[output_index]
    return f"output_{output_index}"
