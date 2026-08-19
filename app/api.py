"""FastAPI backend for the Karak AQI dashboard.

Serves the same forecast contract as ``src.inference_hourly.predict_latest``
(30 outputs: 24 hourly points + four six-hour means + two twelve-hour means)
plus SHAP explanations and hazardous-AQI alerts. The Streamlit dashboard can
call these endpoints or import the functions directly.

Run from ``development``::

    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.explain import explain_latest_origin  # noqa: E402
from app.live_data import (  # noqa: E402
    current_conditions,
    load_latest_hourly,
    latest_origin,
)
from src import config  # noqa: E402
from src.aqi import aqi_category  # noqa: E402
from src.inference_hourly import predict_latest  # noqa: E402
from src.train_hourly import build_hourly_training_frame  # noqa: E402

app = FastAPI(
    title="Karak AQI Forecast API",
    description="30-output hourly AQI forecast (t+1h..t+72h) for Karak, "
    "Pakistan from the selected Ridge model, with SHAP explanations and "
    "hazardous-AQI alerts.",
    version="1.0.0",
)

ALERT_CATEGORIES = {"Very Unhealthy", "Hazardous"}
WARNING_CATEGORIES = {"Unhealthy", "Unhealthy for Sensitive Groups"}


def _alerts(forecast_rows: list[dict]) -> list[dict]:
    alerts = []
    for row in forecast_rows:
        category = aqi_category(row["value"])
        if category in ALERT_CATEGORIES:
            alerts.append(
                {
                    "start_time": str(row["start_time"]),
                    "end_time": str(row["end_time"]),
                    "kind": row["kind"],
                    "value": round(float(row["value"]), 1),
                    "category": category,
                    "severity": "alert" if category == "Hazardous" else "warning",
                }
            )
    return alerts


def _champion_version() -> Optional[str]:
    """Version of the registered hourly champion (best-effort)."""
    try:
        from src.model_registry import list_registered

        for model in list_registered():
            if model["name"] == "aqi-hourly-ridge":
                versions = model.get("latest_versions") or []
                if versions:
                    return f"v{versions[-1].get('version')}"
    except Exception:  # noqa: BLE001 - cosmetic
        pass
    return None


def _iqair_forecast_reference() -> list[dict]:
    """IQAir's own hourly AQI forecast for Karak (best-effort; empty on failure)."""
    try:
        from app.live_data import iqair_forecast_aqi

        reference = iqair_forecast_aqi()
        return [
            {"time": str(ts), "aqi": round(float(value), 1)}
            for ts, value in reference.items()
        ]
    except Exception:  # noqa: BLE001 - reference line is optional
        return []


def _forecast_payload(source: str) -> dict:
    hourly = load_latest_hourly(source)
    forecast = predict_latest(hourly)
    rows = [
        {
            "output": row.output,
            "kind": row.kind,
            "start_time": str(row.start_time),
            "end_time": str(row.end_time),
            "value": round(float(row.value), 1),
            "category": aqi_category(row.value),
        }
        for row in forecast.itertuples()
    ]
    return {
        "origin": str(forecast["forecast_origin"].iloc[0]),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "model": "aqi-hourly-ridge",
        "model_version": _champion_version(),
        "location": f"{config.CITY_NAME} ({config.LOCATION_LABEL})",
        "outputs": rows,
        "alerts": _alerts(rows),
        "iqair_forecast": _iqair_forecast_reference(),
        "current_conditions": current_conditions(hourly),
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "karak-aqi-api",
        "location": f"{config.CITY_NAME} ({config.LOCATION_LABEL})",
        "timezone": config.TIMEZONE,
    }


@app.get("/forecast")
def forecast(
    source: str = Query("store", pattern="^(store|live)$"),
) -> dict:
    """Return the 30-output 72-hour forecast from the latest observations."""
    try:
        return _forecast_payload(source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface a readable API error
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/explain")
def explain(
    output: int = Query(0, ge=0, le=29, description="Target output index (0=t+1h, 23=t+24h, 28=first 12h mean)"),
    source: str = Query("store", pattern="^(store|live)$"),
) -> dict:
    """SHAP attribution for the latest origin's prediction of one output."""
    try:
        hourly = load_latest_hourly(source)
        features = build_hourly_training_frame(hourly, include_targets=False)
        return explain_latest_origin(features, output_index=output)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/registry")
def registry() -> dict:
    """List registered models in the MLflow registry (file-backed)."""
    try:
        from src.model_registry import list_registered

        return {"registered": list_registered()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/latest-origin")
def origin(source: str = Query("store", pattern="^(store|live)$")) -> dict:
    return {
        "origin": str(latest_origin(source)) if latest_origin(source) else None,
        "source": source,
    }
