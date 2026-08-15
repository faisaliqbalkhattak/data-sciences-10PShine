"""Training pipeline for Karak AQI forecasting.

Implements the four-model shortlist described in the internal selection note:
XGBoost, Random Forest, SARIMA, LSTM -- plus persistence (naive) and Ridge
references. Direct multi-horizon forecasting of ``aqi_us_epa`` at +1, +2, +3 days,
validated with a chronological holdout and a three-day purge/embargo so no
training label crosses the forecast boundary. The persistence baseline is
included so every ML model must *beat* "tomorrow equals today" to be worth
shipping.

Run from ``development`` with the Python 3.11 environment: ``python -m src.train``.
The LSTM branch requires the optional pinned TensorFlow requirements; if it is not importable the pipeline logs a note and proceeds with the remaining models instead of failing.

Outputs (written to ``models/`` and ``data/processed/``):
- ``models/aqi_forecast_models.json``     best artifact path + metrics per horizon/model
- ``models/<model>_h<horizon>.joblib``     fitted persisted estimators
- ``data/processed/model_comparison.csv``  RMSE/MAE/R2 per model per horizon
- ``data/processed/training_frame.csv``    the lagged feature/target frame
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
HORIZONS = [1, 2, 3]  # days ahead
MAX_HORIZON = max(HORIZONS)
TARGET = "aqi_us_epa"
LAG_DAYS = [1, 2, 3, 7]
ROLL_WINDOWS = [3, 7]

# Columns available at time t (known at forecast time) used as regressors.
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


def newest_daily_features() -> Path:
    matches = sorted(
        config.DATA_PROCESSED_DIR.glob("karak_aqi_open_meteo_daily_features.csv")
    )
    if not matches:
        raise FileNotFoundError(
            "No processed daily features found. Run notebooks 01-02 first."
        )
    return matches[-1]


def validate_daily_input(daily: pd.DataFrame) -> None:
    """Fail closed when training input is not unique, sorted, and daily."""
    if "time" not in daily.columns:
        raise KeyError("Daily features must contain a 'time' column.")
    times = pd.to_datetime(daily["time"])
    if times.isna().any():
        raise ValueError("Daily features contain invalid timestamps.")
    if times.duplicated().any():
        raise ValueError("Daily features contain duplicate timestamps.")
    if not times.is_monotonic_increasing:
        raise ValueError("Daily feature timestamps must be sorted ascending.")
    if len(times) > 1 and not (times.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Daily features must have a complete one-day cadence.")


def build_training_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, and rolling features plus multi-horizon targets.

    The raw ``aqi_us_epa`` is kept so the persistence baseline can use the
    value at time t to forecast time t+h ("tomorrow equals today").
    """
    validate_daily_input(daily)
    frame = daily.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.set_index("time")
    if TARGET not in frame:
        raise KeyError(f"Target column '{TARGET}' missing from daily features.")

    frame["month_sin"] = np.sin(2 * np.pi * frame.index.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame.index.month / 12)
    frame["dow"] = frame.index.dayofweek
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)
    frame["day_of_year_sin"] = np.sin(2 * np.pi * frame.index.dayofyear / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * frame.index.dayofyear / 365.25)

    for lag in LAG_DAYS:
        frame[f"{TARGET}_lag{lag}"] = frame[TARGET].shift(lag)
        frame[f"pm2_5_lag{lag}"] = frame["pm2_5"].shift(lag)
    for window in ROLL_WINDOWS:
        frame[f"{TARGET}_roll_mean{window}"] = frame[TARGET].rolling(window).mean()
        frame[f"{TARGET}_roll_std{window}"] = frame[TARGET].rolling(window).std()
        frame[f"pm2_5_roll_mean{window}"] = frame["pm2_5"].rolling(window).mean()

    for horizon in HORIZONS:
        frame[f"target_{horizon}d"] = frame[TARGET].shift(-horizon)

    feature_cols = [
        c
        for c in BASE_FEATURES
        + [f"{TARGET}_lag{l}" for l in LAG_DAYS]
        + [f"pm2_5_lag{l}" for l in LAG_DAYS]
        + [f"{TARGET}_roll_mean{w}" for w in ROLL_WINDOWS]
        + [f"{TARGET}_roll_std{w}" for w in ROLL_WINDOWS]
        + [f"pm2_5_roll_mean{w}" for w in ROLL_WINDOWS]
        + ["month_sin", "month_cos", "dow", "is_weekend", "day_of_year_sin", "day_of_year_cos"]
        if c in frame
    ]
    target_cols = [f"target_{h}d" for h in HORIZONS]
    out = frame[[TARGET] + feature_cols + target_cols].dropna()
    return out


def chronological_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
    gap: int = MAX_HORIZON,
):
    """Split by forecast origin and purge ``gap`` rows before the test period.

    The rows removed by the gap are an embargo. With direct target ``t+h``,
    this prevents a training label near the boundary from reaching into the
    period used as test-feature history. The default gap is the largest
    forecast horizon (three days).
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if gap < 0:
        raise ValueError("gap cannot be negative.")
    split_at = int(len(frame) * (1 - test_fraction))
    train_end = split_at - gap
    if train_end <= 0 or split_at >= len(frame):
        raise ValueError("Frame is too short for the requested split and gap.")
    return frame.iloc[:train_end], frame.iloc[split_at:]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_models():
    """Baseline/tree estimators with per-model search spaces."""
    return {
        "ridge": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(random_state=RANDOM_STATE))]
        ),
        "random_forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
    }


SEARCH_SPACES = {
    "ridge": {"model__alpha": [1.0, 10.0, 100.0, 1000.0]},
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, None],
        "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    },
}


def train_sklearn_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, dict]:
    """Randomized search over TimeSeriesSplit for each tree/linear model."""
    tscv = TimeSeriesSplit(n_splits=3, gap=MAX_HORIZON)
    results = {}
    for name, estimator in make_models().items():
        search = RandomizedSearchCV(
            estimator,
            SEARCH_SPACES[name],
            n_iter=12,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        results[name] = {
            "estimator": search.best_estimator_,
            "params": search.best_params_,
        }
    return results


def train_sarima(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Small seasonal-ARIMA grid; returns the best model + test predictions.

    Selection is done on an **internal validation fold** carved out of the
    training set (last 20% of ``y_train``), never on the held-out test set,
    matching the CV-based selection used for the sklearn models. The chosen
    configuration is then refit on the full training series and used to
    forecast the test horizon.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:  # pragma: no cover
        return {"available": False}

    grid = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1)]
    seasonal_options = [(0, 0, 0, 7), (1, 0, 0, 7)]
    n_val = max(1, int(len(y_train) * 0.2))
    y_fit, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    best = None
    best_score = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in grid:
            for seasonal in seasonal_options:
                try:
                    model = SARIMAX(
                        y_fit,
                        order=order,
                        seasonal_order=seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=100)
                    preds = np.asarray(model.forecast(len(y_val)))
                    rmse = np.sqrt(mean_squared_error(y_val, preds))
                    if rmse < best_score:
                        best = {"order": order, "seasonal": seasonal}
                        best_score = rmse
                except Exception:
                    continue

    if best is None:
        return {"available": False}
    # Refit the winning configuration on the full training series.
    model = SARIMAX(
        y_train,
        order=best["order"],
        seasonal_order=best["seasonal"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=100)
    preds = np.asarray(model.forecast(len(y_test)))
    return {
        "available": True,
        "model": model,
        "params": {"order": best["order"], "seasonal_order": best["seasonal"]},
        "predictions": preds,
    }


def train_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    sequence_length: int = 14,
    epochs: int = 25,
) -> dict:
    """Small 2-layer LSTM on scaled lagged windows. Requires TensorFlow.

    Test predictions use only information available up to each test step
    (the window slides over observed history, never into the future).
    """
    try:
        import tensorflow as tf
    except ImportError:
        return {"available": False}

    scaler = StandardScaler().fit(X_train)
    target_scaler = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    ytr = target_scaler.transform(y_train.to_numpy().reshape(-1, 1)).ravel()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    def make_sequences(x, y):
        xs, ys = [], []
        for i in range(sequence_length, len(x)):
            xs.append(x[i - sequence_length + 1 : i + 1])
            ys.append(y[i])
        return np.array(xs), np.array(ys)

    Xtr_s, ytr_s = make_sequences(Xtr, ytr)
    if len(Xtr_s) == 0:
        return {"available": False}

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                64, return_sequences=True, input_shape=(sequence_length, Xtr.shape[1])
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtr_s, ytr_s, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1)

    # Slide the window over observed history so every test point gets a
    # prediction without requiring future data. The window ends at the current
    # test row (same information cutoff as the sklearn models, which use row t
    # features to predict target t+h).
    full = np.vstack([Xtr, Xte])
    n_train = len(Xtr)
    preds = []
    for i in range(len(Xte)):
        end = n_train + i + 1  # include test row i in the window
        window = full[end - sequence_length : end]
        scaled_pred = model.predict(window[None, :, :], verbose=0)[0, 0]
        pred = target_scaler.inverse_transform([[scaled_pred]])[0, 0]
        preds.append(float(pred))

    return {
        "available": True,
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "params": {
            "sequence_length": sequence_length,
            "epochs": epochs,
            "units": [64, 32],
            "target_scaling": "StandardScaler fit on the training target, inverse-transformed for metrics",
        },
        "predictions": np.array(preds),
    }


def main(from_store: bool = False) -> None:
    """Train and evaluate the daily multi-horizon models.

    ``from_store=True`` fetches the validated daily frame from the feature
    store (``src.feature_store``) instead of the processed CSV, matching the
    scheduled training pipeline contract.
    """
    config.ensure_data_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if from_store:
        # Lazy import: feature_store imports this module, so a module-level
        # import would create a cycle.
        from .feature_store import get_daily_features

        frame = get_daily_features()
    else:
        daily = pd.read_csv(newest_daily_features())
        frame = build_training_frame(daily)
    frame.to_csv(config.DATA_PROCESSED_DIR / "training_frame.csv")
    print(
        f"Training frame: {frame.shape[0]} rows x {frame.shape[1]} cols "
        f"({frame.index.min().date()} -> {frame.index.max().date()})"
    )

    feature_cols = [c for c in frame.columns if not c.startswith("target_") and c != TARGET]
    rows = []
    artifacts = {}
    n_train = n_test = None

    for horizon in HORIZONS:
        y_col = f"target_{horizon}d"
        X = frame[feature_cols]
        y = frame[y_col]
        aqi = frame[TARGET]
        X_train, X_test = chronological_split(X)
        y_train, y_test = chronological_split(y)
        _, aqi_test = chronological_split(aqi)
        n_train, n_test = len(X_train), len(X_test)

        # Persistence (naive) baseline: forecast(t+h) = aqi_us_epa(t).
        y_naive = aqi_test.to_numpy()
        metric_p = evaluate(y_test.to_numpy(), y_naive)
        rows.append({"horizon_days": horizon, "model": "persistence", **metric_p})
        print(
            f"horizon={horizon}d | {'persistence':14s} "
            f"RMSE={metric_p['rmse']:.2f} MAE={metric_p['mae']:.2f} R2={metric_p['r2']:.3f}"
        )

        fitted = train_sklearn_models(X_train, y_train)
        for name, info in fitted.items():
            preds = info["estimator"].predict(X_test)
            metric = evaluate(y_test.to_numpy(), preds)
            rows.append({"horizon_days": horizon, "model": name, **metric})
            artifacts[f"{name}_h{horizon}"] = {
                "path": f"models/{name}_h{horizon}.joblib",
                "params": info["params"],
                "metrics": metric,
            }
            joblib.dump(info["estimator"], MODELS_DIR / f"{name}_h{horizon}.joblib")
            print(
                f"horizon={horizon}d | {name:14s} "
                f"RMSE={metric['rmse']:.2f} MAE={metric['mae']:.2f} R2={metric['r2']:.3f}"
            )

        sarima = train_sarima(y_train, y_test)
        if sarima["available"]:
            metric = evaluate(y_test.to_numpy(), sarima["predictions"])
            rows.append({"horizon_days": horizon, "model": "sarima", **metric})
            artifacts[f"sarima_h{horizon}"] = {
                "path": None,
                "params": sarima["params"],
                "metrics": metric,
            }
            print(
                f"horizon={horizon}d | {'sarima':14s} "
                f"RMSE={metric['rmse']:.2f} MAE={metric['mae']:.2f} R2={metric['r2']:.3f}"
            )
        else:
            print(f"horizon={horizon}d | sarima         skipped (statsmodels fit failed)")

        lstm = train_lstm(X_train, y_train, X_test)
        if lstm["available"]:
            metric = evaluate(y_test.to_numpy(), lstm["predictions"])
            rows.append({"horizon_days": horizon, "model": "lstm", **metric})
            artifacts[f"lstm_h{horizon}"] = {
                "path": f"models/lstm_h{horizon}.keras",
                "params": lstm["params"],
                "metrics": metric,
            }
            lstm["model"].save(MODELS_DIR / f"lstm_h{horizon}.keras")
            joblib.dump(lstm["scaler"], MODELS_DIR / f"lstm_h{horizon}_scaler.joblib")
            joblib.dump(
                lstm["target_scaler"],
                MODELS_DIR / f"lstm_h{horizon}_target_scaler.joblib",
            )
            print(
                f"horizon={horizon}d | {'lstm':14s} "
                f"RMSE={metric['rmse']:.2f} MAE={metric['mae']:.2f} R2={metric['r2']:.3f}"
            )
        else:
            print(f"horizon={horizon}d | lstm           skipped (TensorFlow not installed)")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.DATA_PROCESSED_DIR / "model_comparison.csv", index=False)

    artifacts["_meta"] = {
        "created": date.today().isoformat(),
        "target": TARGET,
        "horizons_days": HORIZONS,
        "features": feature_cols,
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "selection_method": "lowest holdout RMSE among persisted estimators; regenerate after each data refresh",
        "best_model_by_horizon": {
            str(horizon): comparison[
                (comparison["horizon_days"] == horizon)
                & comparison["model"].isin(["ridge", "random_forest", "xgboost", "lstm"])
            ].sort_values("rmse").iloc[0]["model"]
            for horizon in HORIZONS
            if not comparison[
                (comparison["horizon_days"] == horizon)
                & comparison["model"].isin(["ridge", "random_forest", "xgboost", "lstm"])
            ].empty
        },
        "evaluation_protocol": "chronological holdout with three-day purge gap; TimeSeriesSplit gap=3 for tuning",
        "forecast_contract": "issue at 00:00 Asia/Karachi after the previous local day is complete",
    }
    with open(MODELS_DIR / "aqi_forecast_models.json", "w", encoding="utf-8") as fh:
        json.dump(artifacts, fh, indent=2, default=str)

    print("\nComparison table:")
    print(comparison.to_string(index=False))
    print("\nArtifacts ->", MODELS_DIR)
    print("Comparison CSV ->", config.DATA_PROCESSED_DIR / "model_comparison.csv")


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
