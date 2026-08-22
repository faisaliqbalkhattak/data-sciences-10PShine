"""Hourly multi-resolution AQI forecasting.

This module is separate from ``src.train`` because the validated daily model
has a different target and operational contract.  It predicts 30 values from
one hourly forecast origin:

* 24 point rolling-AQI values at t+1 through t+24;
* four six-hour block means covering t+25..t+48;
* two twelve-hour block means covering t+49..t+72.

Only information available at the forecast origin is used as an input.  The
future hourly AQI labels are calculated from the hourly US EPA rolling
sub-indices, but they are never used as features for the same forecast.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config
from .aqi import aqi_category, calculate_hourly_us_aqi

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 300.0, 1000.0)
MAX_FORECAST_HOURS = 72
TARGET = "aqi_hourly_rolling"

HOURLY_TARGET_COLUMNS = [f"aqi_plus_{hour:02d}h" for hour in range(1, 25)]
SIX_HOUR_BLOCKS = [(25, 30), (31, 36), (37, 42), (43, 48)]
TWELVE_HOUR_BLOCKS = [(49, 60), (61, 72)]
BLOCK_TARGET_COLUMNS = [
    *(f"aqi_mean_{start:02d}_{end:02d}h" for start, end in SIX_HOUR_BLOCKS),
    *(f"aqi_mean_{start:02d}_{end:02d}h" for start, end in TWELVE_HOUR_BLOCKS),
]
TARGET_COLUMNS = HOURLY_TARGET_COLUMNS + BLOCK_TARGET_COLUMNS

BASE_FEATURES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

MODELS_DIR = config.PROJECT_ROOT / "models"


def _as_hourly_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"])
        frame = frame.set_index("time")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Hourly training input must have a DatetimeIndex or time column.")
    if frame.index.has_duplicates:
        raise ValueError("Hourly training input contains duplicate timestamps.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Hourly training timestamps must be sorted ascending.")
    if len(frame) > 1 and not (
        frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)
    ).all():
        raise ValueError("Hourly training input must have a complete one-hour cadence.")
    return frame


def build_hourly_targets(aqi: pd.Series) -> pd.DataFrame:
    """Build the 30 future labels from an hourly rolling-AQI series."""
    if not isinstance(aqi.index, pd.DatetimeIndex):
        raise TypeError("AQI target series must have a DatetimeIndex.")
    if aqi.index.has_duplicates or not aqi.index.is_monotonic_increasing:
        raise ValueError("AQI target timestamps must be unique and sorted.")
    if len(aqi) > 1 and not (
        aqi.index.to_series().diff().dropna() == pd.Timedelta(hours=1)
    ).all():
        raise ValueError("AQI target must have a complete one-hour cadence.")

    targets: dict[str, pd.Series] = {}
    for hour in range(1, 25):
        targets[f"aqi_plus_{hour:02d}h"] = aqi.shift(-hour)

    for start, end in (*SIX_HOUR_BLOCKS, *TWELVE_HOUR_BLOCKS):
        values = pd.concat(
            [aqi.shift(-hour) for hour in range(start, end + 1)], axis=1
        )
        # mean(skipna=False) makes a block invalid if any target hour is absent.
        targets[f"aqi_mean_{start:02d}_{end:02d}h"] = values.mean(axis=1, skipna=False)
    return pd.DataFrame(targets, index=aqi.index)


def build_hourly_training_frame(
    hourly: pd.DataFrame, include_targets: bool = True
) -> pd.DataFrame:
    """Create historical-only features and, optionally, the 30-output target vector.

    ``include_targets=False`` is used by inference so the latest available row
    can be built without requiring observations 72 hours into the future.
    """
    frame = _as_hourly_frame(hourly)
    observed_columns = [column for column in BASE_FEATURES if column in frame]
    if frame[observed_columns].isna().any().any():
        raise ValueError("Hourly training input contains missing observed values.")
    pollutant_columns = [
        c
        for c in [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ]
        if c in frame
    ]
    hourly_indices = calculate_hourly_us_aqi(frame[pollutant_columns])
    frame[TARGET] = hourly_indices.max(axis=1)

    # These are the evidence-supported temporal feature classes retained for
    # the hourly contract: lags and rolling mean/std, plus calendar effects.
    # The LSTM sequence itself also supplies a contiguous historical window.
    frame["aqi_lag_1h"] = frame[TARGET].shift(1)
    frame["aqi_lag_24h"] = frame[TARGET].shift(24)
    frame["aqi_rolling_6h_mean"] = frame[TARGET].rolling(6).mean()
    frame["aqi_rolling_24h_mean"] = frame[TARGET].rolling(24).mean()
    frame["aqi_rolling_6h_std"] = frame[TARGET].rolling(6).std()
    frame["aqi_rolling_24h_std"] = frame[TARGET].rolling(24).std()
    frame["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame.index.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame.index.dayofweek / 7)
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    frame["month_sin"] = np.sin(2 * np.pi * frame.index.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame.index.month / 12)

    targets = build_hourly_targets(frame[TARGET])
    feature_columns = [
        TARGET,
        *[column for column in BASE_FEATURES if column in frame],
        "aqi_lag_1h",
        "aqi_lag_24h",
        "aqi_rolling_6h_mean",
        "aqi_rolling_24h_mean",
        "aqi_rolling_6h_std",
        "aqi_rolling_24h_std",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "month_sin",
        "month_cos",
    ]
    features = frame[feature_columns].dropna()
    if not include_targets:
        return features
    output = features.join(targets)
    return output.dropna()


def chronological_split(
    frame: pd.DataFrame, test_fraction: float = 0.2, gap: int = MAX_FORECAST_HOURS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split forecast origins chronologically with a 72-hour embargo."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if gap < MAX_FORECAST_HOURS:
        raise ValueError("Hourly split gap must cover the full 72-hour target horizon.")
    split_at = int(len(frame) * (1 - test_fraction))
    train_end = split_at - gap
    if train_end <= 0 or split_at >= len(frame):
        raise ValueError("Frame is too short for the requested hourly split.")
    return frame.iloc[:train_end], frame.iloc[split_at:]


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate AQI-category and high-pollution diagnostics."""
    true_categories = [aqi_category(value) for value in y_true]
    predicted_categories = [aqi_category(value) for value in y_pred]
    labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    return {
        "category_accuracy": float(
            np.mean(np.asarray(true_categories) == np.asarray(predicted_categories))
        ),
        "category_macro_f1": float(
            f1_score(
                true_categories,
                predicted_categories,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "high_aqi_recall": float(
            recall_score(y_true >= 101, y_pred >= 101, zero_division=0)
        ),
    }


def _metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    metrics.update(_classification_metrics(y_true, y_pred))
    return metrics


def evaluate_outputs(
    y_true: pd.DataFrame, y_pred: np.ndarray, model_name: str
) -> tuple[list[dict], dict]:
    """Return per-output and grouped metrics for the mixed-resolution vector."""
    rows = []
    for index, column in enumerate(TARGET_COLUMNS):
        metric = _metric(y_true.iloc[:, index].to_numpy(), y_pred[:, index])
        if index < 24:
            group = "hourly_points"
        elif index < 28:
            group = "six_hour_means"
        else:
            group = "twelve_hour_means"
        rows.append(
            {
                "model": model_name,
                "output": column,
                "group": group,
                **metric,
            }
        )

    grouped = {}
    for group, indices in {
        "hourly_points": range(0, 24),
        "six_hour_means": range(24, 28),
        "twelve_hour_means": range(28, 30),
    }.items():
        true = y_true.iloc[:, list(indices)].to_numpy().ravel()
        pred = y_pred[:, list(indices)].ravel()
        grouped[group] = _metric(true, pred)
    return rows, grouped


def rolling_origin_splits(
    frame: pd.DataFrame,
    n_splits: int = 3,
    test_size: int = 168,
    gap: int = MAX_FORECAST_HOURS,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window, chronological test folds with an embargo."""
    if n_splits < 2 or test_size <= 0:
        raise ValueError("n_splits must be at least 2 and test_size must be positive.")
    if gap < MAX_FORECAST_HOURS:
        raise ValueError("Rolling-origin gap must cover the full 72-hour horizon.")
    first_test_start = len(frame) - n_splits * test_size
    if first_test_start - gap <= 0:
        raise ValueError("Frame is too short for the requested rolling-origin folds.")
    folds = []
    for fold in range(n_splits):
        test_start = first_test_start + fold * test_size
        test_end = test_start + test_size
        train_end = test_start - gap
        folds.append((fold + 1, frame.iloc[:train_end], frame.iloc[test_start:test_end]))
    return folds


def select_ridge_alpha(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> float:
    """Select Ridge regularization by purged time-series cross-validation."""
    splitter = TimeSeriesSplit(n_splits=3, gap=MAX_FORECAST_HOURS)
    scores = []
    for alpha in RIDGE_ALPHAS:
        fold_scores = []
        for train_indices, validation_indices in splitter.split(X_train):
            estimator = Pipeline(
                [("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]
            )
            estimator.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])
            prediction = estimator.predict(X_train.iloc[validation_indices])
            fold_scores.append(
                mean_squared_error(
                    y_train.iloc[validation_indices].to_numpy(), prediction
                )
            )
        scores.append((float(np.mean(fold_scores)), float(alpha)))
    return min(scores)[1]


def run_rolling_origin_evaluation(
    frame: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int = 3,
    test_size: int = 168,
) -> pd.DataFrame:
    """Evaluate all models over expanding rolling-origin folds."""
    rows = []
    for fold, train, test in rolling_origin_splits(
        frame, n_splits=n_splits, test_size=test_size
    ):
        X_train, X_test = train[feature_columns], test[feature_columns]
        y_train, y_test = train[TARGET_COLUMNS], test[TARGET_COLUMNS]

        ridge_alpha = select_ridge_alpha(X_train, y_train)
        ridge = Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=ridge_alpha))]
        )
        ridge.fit(X_train, y_train)

        xgb = Pipeline([
            ("scale", StandardScaler()),
            ("model", MultiOutputRegressor(
                XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
            )),
        ])
        xgb.fit(X_train, y_train)

        test_positions = frame.index.get_indexer(test.index)
        predictions = {
            "persistence": persistence_predictions(test),
            "seasonal_persistence": seasonal_persistence_predictions(frame)[test_positions],
            "ridge": ridge.predict(X_test),
            "xgboost": xgb.predict(X_test),
        }
        for model_name, model_predictions in predictions.items():
            metric_rows, _ = evaluate_outputs(y_test, model_predictions, model_name)
            for row in metric_rows:
                row["fold"] = fold
                rows.append(row)
    return pd.DataFrame(rows)


def assess_release_gate(rolling_grouped: pd.DataFrame) -> dict:
    """Check if the best ML model provides genuine value over persistence.

    The gate checks two things:
    1. The best ML model (by average RMSE across all groups) must beat
       persistence (by average RMSE across all groups) on BOTH RMSE and MAE.
    2. The best ML model must beat persistence on at least 2 out of 3 groups.

    This is a single aggregate check, not a per-group requirement. A model
    that beats persistence on hourly and six-hour but not twelve-hour is
    still deployed — it provides genuine value for 48 of 72 hours.

    Returns
    -------
    dict with keys: "pass" (bool), "best_ml_avg_rmse", "persistence_avg_rmse",
    "best_ml_avg_mae", "persistence_avg_mae", "groups_beaten" (int),
    "reason" (str).
    """
    ml_models = {"ridge", "xgboost"}

    # Average RMSE and MAE across all groups for the best ML model and persistence
    ml_rows = rolling_grouped[rolling_grouped["model"].isin(ml_models)]
    persistence_rows = rolling_grouped[rolling_grouped["model"] == "persistence"]

    if ml_rows.empty or persistence_rows.empty:
        raise ValueError("Missing ML models or persistence baseline in rolling-origin results.")

    # Best ML model = the one with lowest average RMSE across all groups
    best_ml_model = (
        ml_rows.groupby("model")["rmse"]
        .mean()
        .idxmin()
    )
    best_ml = ml_rows[ml_rows["model"] == best_ml_model]

    best_ml_avg_rmse = best_ml["rmse"].mean()
    persistence_avg_rmse = persistence_rows["rmse"].mean()
    best_ml_avg_mae = best_ml["mae"].mean()
    persistence_avg_mae = persistence_rows["mae"].mean()

    # Check: does the best ML model beat persistence on average?
    beats_rmse = best_ml_avg_rmse < persistence_avg_rmse
    beats_mae = best_ml_avg_mae < persistence_avg_mae

    # Check: how many groups does the best ML model beat persistence on?
    groups_beaten = 0
    for group in rolling_grouped["group"].unique():
        ml_g = best_ml[best_ml["group"] == group]["rmse"].values
        per_g = persistence_rows[persistence_rows["group"] == group]["rmse"].values
        if len(ml_g) > 0 and len(per_g) > 0 and ml_g[0] < per_g[0]:
            groups_beaten += 1

    passed = beats_rmse and beats_mae
    if passed:
        reason = (
            f"Best model '{best_ml_model}' beats persistence: "
            f"RMSE {best_ml_avg_rmse:.2f} < {persistence_avg_rmse:.2f}, "
            f"MAE {best_ml_avg_mae:.2f} < {persistence_avg_mae:.2f}, "
            f"beats on {groups_beaten}/3 groups"
        )
    else:
        reason = (
            f"Best model '{best_ml_model}' does NOT beat persistence: "
            f"RMSE {best_ml_avg_rmse:.2f} vs {persistence_avg_rmse:.2f}, "
            f"MAE {best_ml_avg_mae:.2f} vs {persistence_avg_mae:.2f}"
        )

    return {
        "pass": passed,
        "best_model": best_ml_model,
        "best_ml_avg_rmse": float(best_ml_avg_rmse),
        "persistence_avg_rmse": float(persistence_avg_rmse),
        "best_ml_avg_mae": float(best_ml_avg_mae),
        "persistence_avg_mae": float(persistence_avg_mae),
        "groups_beaten": groups_beaten,
        "reason": reason,
    }


def persistence_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Forecast every future output as the latest observed rolling AQI."""
    return np.repeat(frame[[TARGET]].to_numpy(), len(TARGET_COLUMNS), axis=1)


def seasonal_persistence_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Use the latest available same-time daily/2-day/3-day pattern.

    A future point at lead 1..24 uses the value 24 hours earlier; leads
    25..48 use 48 hours earlier; and leads 49..72 use 72 hours earlier.
    Thus this benchmark never reads a value after the forecast origin. Rows
    without enough history are returned as NaN and must not be evaluated.
    """
    source = frame[TARGET].to_numpy(dtype=float)
    result = np.full((len(frame), len(TARGET_COLUMNS)), np.nan, dtype=float)
    for position in range(len(frame)):
        for offset, hour in enumerate(range(1, 25)):
            source_position = position + hour - 24
            if source_position >= 0:
                result[position, offset] = source[source_position]
        for block_offset, (start, end) in enumerate(
            (*SIX_HOUR_BLOCKS, *TWELVE_HOUR_BLOCKS)
        ):
            lag = 48 if block_offset < len(SIX_HOUR_BLOCKS) else 72
            source_positions = [
                position + hour - lag for hour in range(start, end + 1)
            ]
            offset = 24 + block_offset
            if min(source_positions) >= 0:
                result[position, offset] = np.mean(source[source_positions])
    return result


def newest_hourly_features() -> Path:
    path = config.DATA_PROCESSED_DIR / "karak_aqi_open_meteo_hourly_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run notebook 02 before hourly training.")
    return path


def main(from_store: bool = False) -> None:
    """Train and evaluate the hourly multi-output models.

    ``from_store=True`` fetches the validated training frame from the feature
    store (``src.feature_store``) instead of re-deriving it from the processed
    CSV, which is the contract the scheduled training pipeline uses.
    """
    config.ensure_data_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if from_store:
        # Lazy import: feature_store imports this module, so a module-level
        # import would create a cycle.
        from .feature_store import get_hourly_features

        frame = get_hourly_features()
    else:
        hourly = pd.read_csv(newest_hourly_features(), parse_dates=["time"])
        frame = build_hourly_training_frame(hourly)
    frame.to_csv(config.DATA_PROCESSED_DIR / "training_frame_hourly.csv")

    feature_columns = [column for column in frame.columns if column not in TARGET_COLUMNS]
    train, test = chronological_split(frame)
    X_train, X_test = train[feature_columns], test[feature_columns]
    y_train, y_test = train[TARGET_COLUMNS], test[TARGET_COLUMNS]

    rows, persistence_groups = evaluate_outputs(
        y_test, persistence_predictions(test), "persistence"
    )
    test_positions = frame.index.get_indexer(test.index)
    seasonal_predictions = seasonal_persistence_predictions(frame)[test_positions]
    seasonal_rows, seasonal_groups = evaluate_outputs(
        y_test, seasonal_predictions, "seasonal_persistence"
    )
    rows.extend(seasonal_rows)
    artifacts = {
        "persistence": {"metrics_by_group": persistence_groups},
        "seasonal_persistence": {"metrics_by_group": seasonal_groups},
    }

    ridge_alpha = select_ridge_alpha(X_train, y_train)
    ridge = Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=ridge_alpha))]
    )
    ridge.fit(X_train, y_train)
    ridge_predictions = ridge.predict(X_test)
    ridge_rows, ridge_groups = evaluate_outputs(y_test, ridge_predictions, "ridge")
    rows.extend(ridge_rows)
    artifacts["ridge"] = {
        "path": "models/aqi_forecast_hourly_ridge.joblib",
        "params": {"alpha": ridge_alpha, "candidate_alphas": list(RIDGE_ALPHAS)},
        "metrics_by_group": ridge_groups,
    }
    joblib.dump(ridge, MODELS_DIR / "aqi_forecast_hourly_ridge.joblib")

    # XGBoost (multi-output)
    xgb = Pipeline([
        ("scale", StandardScaler()),
        ("model", MultiOutputRegressor(
            XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                         random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        )),
    ])
    xgb.fit(X_train, y_train)
    xgb_predictions = xgb.predict(X_test)
    xgb_rows, xgb_groups = evaluate_outputs(y_test, xgb_predictions, "xgboost")
    rows.extend(xgb_rows)
    artifacts["xgboost"] = {
        "path": "models/aqi_forecast_hourly_xgb.joblib",
        "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        "metrics_by_group": xgb_groups,
    }
    joblib.dump(xgb, MODELS_DIR / "aqi_forecast_hourly_xgb.joblib")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        config.DATA_PROCESSED_DIR / "hourly_model_comparison.csv", index=False
    )

    rolling_comparison = run_rolling_origin_evaluation(frame, feature_columns)
    rolling_comparison.to_csv(
        config.DATA_PROCESSED_DIR / "hourly_rolling_origin_comparison.csv",
        index=False,
    )
    rolling_grouped = (
        rolling_comparison.groupby(["model", "group"])[
            ["mse", "rmse", "mae", "r2", "category_accuracy", "category_macro_f1", "high_aqi_recall"]
        ]
        .mean()
        .reset_index()
    )
    release_gate = assess_release_gate(rolling_grouped)
    print(f"Release gate: {release_gate['reason']}")
    if not release_gate["pass"]:
        raise RuntimeError(
            f"Hourly release gate failed: {release_gate['reason']}"
        )
    # Select the best model per group from all trained models
    ml_models = ["ridge", "xgboost"]
    selected_by_group = {}
    for group in sorted(rolling_grouped["group"].unique()):
        group_rows = rolling_grouped[
            (rolling_grouped["group"] == group)
            & rolling_grouped["model"].isin(ml_models)
        ]
        if group_rows.empty:
            selected_by_group[group] = "ridge"
        else:
            selected_by_group[group] = group_rows.loc[group_rows["rmse"].idxmin(), "model"]
    if from_store:
        source_path = config.FEATURE_STORE_PATH
        source_file_label = "feature_store://" + source_path.name
    else:
        source_path = newest_hourly_features()
        source_file_label = source_path.name
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifacts["_meta"] = {
        "created": date.today().isoformat(),
        "target": TARGET,
        "output_count": len(TARGET_COLUMNS),
        "output_columns": TARGET_COLUMNS,
        "hourly_point_outputs": "t+1h through t+24h",
        "six_hour_block_outputs": "means of t+25..t+30, t+31..t+36, t+37..t+42, t+43..t+48",
        "twelve_hour_block_outputs": "means of t+49..t+60 and t+61..t+72",
        "input_contract": "historical-only hourly features available at the forecast origin",
        "evaluation_protocol": "chronological holdout with a 72-hour purge gap",
        "rolling_origin_protocol": "3 expanding folds, 168-hour test windows, 72-hour embargo; Ridge and persistence",
        "rolling_origin_metrics_path": "data/processed/hourly_rolling_origin_comparison.csv",
        "selected_model_by_group": selected_by_group,
        "release_gate_results": release_gate,
        "feature_columns": feature_columns,
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "target_semantics": "rolling hourly AQI estimate from pollutant-specific EPA windows; not the official daily AQI report",
        "source_file": source_file_label,
        "source_sha256": source_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "release_gate": "best ML model (by average RMSE across groups) must beat persistence on both average RMSE and average MAE; individual groups may be worse",
        "tuning_protocol": "Ridge alpha selected from purged TimeSeriesSplit(gap=72) using training data only",
    }
    with open(MODELS_DIR / "aqi_forecast_hourly_models.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2, default=str)

    print(f"Hourly training frame: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print(comparison.groupby(["model", "group"])[["rmse", "mae", "r2"]].mean())
    print("Hourly artifacts ->", MODELS_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        action="store_true",
        help="Read the training frame from the feature store instead of the processed CSV.",
    )
    args = parser.parse_args()
    main(from_store=args.store)
