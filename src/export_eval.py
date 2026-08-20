"""Export model evaluation stats and SHAP explanations as a static JSON file.

The training pipeline runs this after training to pre-compute:
- Model registry status
- Hourly holdout stats (RMSE/MAE/R2 per model per group)
- Daily holdout stats (RMSE/MAE/R2 per model per horizon)
- Rolling-origin evaluation stats
- SHAP explanations for the latest prediction (top features)

The dashboard reads this JSON instead of trying to load local CSVs
that don't exist on Streamlit Cloud.

Usage (from ``development``)::

    python -m src.export_eval
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

logger = logging.getLogger(__name__)

#: Where the dashboard reads the pre-computed evaluation data.
EVAL_PATH = config.PROJECT_ROOT / "data" / "model_eval.json"


def _export_eval() -> Path:
    """Generate the evaluation JSON and write it to EVAL_PATH."""
    result = {}

    # --- Model registry ---
    try:
        from src.model_registry import list_registered

        registered = list_registered()
        reg_rows = []
        for model in registered[:6]:
            versions = model.get("latest_versions", [])
            latest = versions[-1] if versions else {}
            aliases = ",".join(latest.get("alias") or []) or "none"
            reg_rows.append({
                "name": model["name"],
                "version": latest.get("version"),
                "alias": aliases,
            })
        result["registry"] = reg_rows
    except Exception as exc:
        logger.warning("Registry unavailable: %s", exc)
        result["registry"] = []

    # --- Hourly holdout ---
    hourly_path = config.DATA_PROCESSED_DIR / "hourly_model_comparison.csv"
    if hourly_path.exists():
        try:
            df = pd.read_csv(hourly_path)
            grouped = (
                df.groupby(["model", "group"])[["rmse", "mae", "r2"]]
                .mean()
                .reset_index()
                .round(3)
            )
            result["hourly_holdout"] = grouped.to_dict(orient="records")
        except Exception as exc:
            logger.warning("Hourly holdout read failed: %s", exc)
            result["hourly_holdout"] = []
    else:
        result["hourly_holdout"] = []

    # --- Daily holdout ---
    daily_path = config.DATA_PROCESSED_DIR / "model_comparison.csv"
    if daily_path.exists():
        try:
            df = pd.read_csv(daily_path)
            pivot = df.pivot_table(
                index="model", columns="horizon_days", values="rmse"
            ).round(2)
            # Flatten into rows like {"model": "persistence", "1d_rmse": 18.5, "2d_rmse": 22.9, ...}
            records = []
            for model, row in pivot.iterrows():
                rec = {"model": model}
                for col in pivot.columns:
                    rec[f"{col}d_rmse"] = row[col]
                records.append(rec)
            result["daily_holdout"] = {"rows": records}
        except Exception as exc:
            logger.warning("Daily holdout read failed: %s", exc)
            result["daily_holdout"] = {}
    else:
        result["daily_holdout"] = {}

    # --- Rolling-origin evaluation ---
    rolling_path = config.DATA_PROCESSED_DIR / "hourly_rolling_origin_comparison.csv"
    if rolling_path.exists():
        try:
            df = pd.read_csv(rolling_path)
            grouped = (
                df.groupby(["model", "group"])[
                    ["mse", "rmse", "mae", "r2", "category_accuracy", "high_aqi_recall"]
                ]
                .mean()
                .round(3)
                .reset_index()
            )
            result["rolling_origin"] = grouped.to_dict(orient="records")
        except Exception as exc:
            logger.warning("Rolling origin read failed: %s", exc)
            result["rolling_origin"] = []
    else:
        result["rolling_origin"] = []

    # --- EDA: training frame summaries ---
    hourly_frame_path = config.DATA_PROCESSED_DIR / "training_frame_hourly.csv"
    if hourly_frame_path.exists():
        try:
            frame = pd.read_csv(hourly_frame_path, parse_dates=["time"]).set_index("time")
            frame = frame.sort_index().tail(90 * 24)  # last 90 days
            from src.aqi import aqi_category

            counts = frame["aqi_hourly_rolling"].map(aqi_category).value_counts()
            result["eda_hourly_dist"] = [
                {"category": cat, "hours": int(count)} for cat, count in counts.items()
            ]
            # Downsample for chart: take every 6th hour
            sampled = frame[["aqi_hourly_rolling"]].dropna()
            sampled = sampled.iloc[::6]
            result["eda_hourly_ts"] = [
                {"time": str(t), "aqi": round(float(v), 1)}
                for t, v in sampled["aqi_hourly_rolling"].items()
            ]
        except Exception as exc:
            logger.warning("EDA hourly read failed: %s", exc)
            result["eda_hourly_dist"] = []
            result["eda_hourly_ts"] = []
    else:
        result["eda_hourly_dist"] = []
        result["eda_hourly_ts"] = []

    daily_frame_path = config.DATA_PROCESSED_DIR / "training_frame.csv"
    if daily_frame_path.exists():
        try:
            frame = pd.read_csv(daily_frame_path, parse_dates=["time"]).set_index("time")
            frame = frame.sort_index().tail(730)  # last 2 years
            sampled = frame[["aqi_us_epa"]].dropna().iloc[::7]  # weekly
            result["eda_daily_ts"] = [
                {"time": str(t), "aqi": round(float(v), 1)}
                for t, v in sampled["aqi_us_epa"].items()
            ]
        except Exception as exc:
            logger.warning("EDA daily read failed: %s", exc)
            result["eda_daily_ts"] = []
    else:
        result["eda_daily_ts"] = []

    # --- SHAP explanations ---
    try:
        from app.explain import explain_latest_origin
        from app.live_data import load_latest_hourly
        from src.train_hourly import build_hourly_training_frame

        hourly = load_latest_hourly("live")
        if hourly is not None and len(hourly) > 0:
            features = build_hourly_training_frame(hourly, include_targets=False)
            shap_result = explain_latest_origin(features, output_index=0)
            # Keep only top 15 features for the JSON
            shap_result["features"] = shap_result["features"][:15]
            result["shap"] = shap_result
        else:
            result["shap"] = None
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        result["shap"] = None

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Evaluation written to %s (registry=%d, hourly=%d, daily=%s, rolling=%d, shap=%s)",
        EVAL_PATH,
        len(result.get("registry", [])),
        len(result.get("hourly_holdout", [])),
        type(result.get("daily_holdout")).__name__,
        len(result.get("rolling_origin", [])),
        "yes" if result.get("shap") else "no",
    )
    return EVAL_PATH


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Export model evaluation as static JSON")
    args = parser.parse_args()
    path = _export_eval()
    print(f"Evaluation written to {path}")
