"""Export the 72-hour forecast as a static JSON file for the dashboard.

This script is called by the CI pipelines (feature pipeline hourly, training
pipeline daily) to pre-compute predictions.  The dashboard reads the resulting
JSON file instead of running inference at runtime, giving every visitor a
near-instant page load.

The JSON file is committed back to the repo so Streamlit Cloud (and any other
static host) can serve it without a backend.

Usage (from ``development``)::

    python -m src.export_forecast --source live
    python -m src.export_forecast --source store
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests as _requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.aqi import aqi_category, calculate_hourly_us_aqi  # noqa: E402
from src.inference_hourly import forecast_rows, predict_latest  # noqa: E402

logger = logging.getLogger(__name__)

#: Where the dashboard reads the pre-computed forecast.
FORECAST_PATH = config.PROJECT_ROOT / "data" / "static_forecast.json"


def _fetch_open_meteo_aq_forecast() -> list[dict]:
    """Fetch 72h hourly US AQI forecast from Open-Meteo (free, keyless).

    Returns a list of dicts with 'time' and 'aqi' keys.
    """
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": 33.1255,
            "longitude": 71.5372,
            "hourly": "us_aqi",
            "forecast_days": 3,
            "timezone": "Asia/Karachi",
        }
        resp = _requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        times = data.get("hourly", {}).get("time", [])
        aqis = data.get("hourly", {}).get("us_aqi", [])
        result = []
        for t, a in zip(times, aqis):
            if a is not None:
                result.append({"time": t, "aqi": round(float(a), 1)})
        return result
    except Exception as exc:
        logger.warning("Open-Meteo AQ forecast fetch failed: %s", exc)
        return []


def _current_hour_local() -> pd.Timestamp:
    """Current hour in the project timezone (naive, aligned to the data grid)."""
    return pd.Timestamp.now(tz=config.TIMEZONE).floor("h").tz_localize(None)


def _current_aqi_from_data(hourly: pd.DataFrame) -> dict:
    """Calculate the current hour's AQI from observed data using the US EPA formula.

    Returns a dict with ``aqi`` (float), ``category`` (str), ``main_pollutant``
    (str) and ``concentration`` (float, ug/m3).
    """
    if hourly is None or hourly.empty:
        return {"aqi": None, "category": None, "main_pollutant": None, "concentration": None}

    # Calculate pollutant sub-indices from the latest observations
    aqi_cols = [c for c in config.AIR_QUALITY_HOURLY_VARS if c in hourly.columns]
    if not aqi_cols:
        return {"aqi": None, "category": None, "main_pollutant": None, "concentration": None}

    try:
        subindices = calculate_hourly_us_aqi(hourly[aqi_cols])
        latest = subindices.iloc[-1]
        if latest.notna().any():
            main_pollutant = str(latest.idxmax()).removeprefix("aqi_")
            aqi_value = float(latest.max())
        else:
            main_pollutant = "pm2_5"
            aqi_value = None
    except Exception:
        main_pollutant = "pm2_5"
        aqi_value = None

    # Get the raw concentration of the main pollutant
    raw = hourly.iloc[-1].get(main_pollutant)
    concentration = round(float(raw), 1) if pd.notna(raw) else None

    return {
        "aqi": round(aqi_value, 1) if aqi_value is not None else None,
        "category": aqi_category(aqi_value) if aqi_value is not None else None,
        "main_pollutant": main_pollutant,
        "concentration": concentration,
    }


def export_forecast(source: str = "live") -> Path:
    """Generate the forecast JSON and write it to ``FORECAST_PATH``.

    Parameters
    ----------
    source : str
        ``"live"`` pulls fresh data from Open-Meteo; ``"store"`` reads the
        DuckDB feature store (falls back to live if empty).

    Returns
    -------
    Path
        The path to the written JSON file.
    """
    from app.live_data import current_conditions, load_latest_hourly

    logger.info("Exporting forecast (source=%s)", source)

    # Load the latest hourly observations
    try:
        hourly = load_latest_hourly(source)
    except Exception:
        if source == "store":
            logger.warning("Store empty, falling back to live")
            hourly = load_latest_hourly("live")
        else:
            raise

    # Run inference
    forecast = predict_latest(hourly)
    origin = pd.Timestamp(forecast["forecast_origin"].iloc[0])

    # Current hour AQI from observed data
    current_aqi = _current_aqi_from_data(hourly)

    # Open-Meteo AQ forecast reference: 72h of hourly US AQI, free & keyless.
    ref_data = _fetch_open_meteo_aq_forecast()
    logger.info("Open-Meteo AQ ref (raw): %d values", len(ref_data))

    # Trim ref_forecast to start from the origin hour (skip past hours).
    origin_iso = origin.isoformat()
    ref_data = [r for r in ref_data if r["time"] >= origin_iso]
    logger.info("Open-Meteo AQ ref (trimmed to origin %s): %d values", origin_iso, len(ref_data))

    # Build the outputs array
    outputs = []
    for _, row in forecast.iterrows():
        outputs.append({
            "output": row["output"],
            "kind": row["kind"],
            "start_time": row["start_time"].isoformat(),
            "end_time": row["end_time"].isoformat(),
            "value": round(float(row["value"]), 1),
            "category": aqi_category(row["value"]),
        })

    # ref_now: find the Open-Meteo AQI for the current hour (closest to now).
    now_iso = _current_hour_local().isoformat()
    ref_now_val = None
    for r in ref_data:
        if r["time"] >= now_iso:
            ref_now_val = r["aqi"]
            break
    # Fallback: if trimmed ref_data starts after now, use the first value.
    if ref_now_val is None and ref_data:
        ref_now_val = ref_data[0]["aqi"]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "origin": origin.isoformat(),
        "model": "aqi-hourly-ridge",
        "current_aqi": current_aqi,
        "ref_now": ref_now_val,
        "outputs": outputs,
        "ref_forecast": ref_data,
    }

    FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORECAST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Forecast written to %s (%d outputs, %d ref refs)", FORECAST_PATH, len(outputs), len(ref_data))
    return FORECAST_PATH


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Export forecast as static JSON")
    parser.add_argument("--source", choices=["store", "live"], default="live")
    args = parser.parse_args()
    path = export_forecast(args.source)
    print(f"Forecast written to {path}")
