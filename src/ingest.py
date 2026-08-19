"""Open-Meteo-only ingestion for Karak.

The filename is part of the data contract:
karak_<purpose>_<provider>_<frequency>_<start>_to_<end>_<pull>.csv
"""

from __future__ import annotations
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
from . import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _get_with_retry(
    url: str, params: dict, retries: int = 3, backoff: float = 2.0
) -> dict:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning("Request failed (%d/%d): %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError("retry loop unexpectedly exited")


def _frame_from_hourly_response(data: dict, source_name: str) -> pd.DataFrame:
    if "hourly" not in data or "time" not in data["hourly"]:
        raise ValueError(
            f"Open-Meteo response did not contain hourly time data: {data.keys()}"
        )
    frame = pd.DataFrame(data["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame["source"] = source_name
    return frame


def fetch_open_meteo_air_quality(
    start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    end_date = end_date or config.AS_OF_DATE
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(config.AIR_QUALITY_HOURLY_VARS),
        "timezone": config.TIMEZONE,
    }
    logger.info("Fetching Open-Meteo air quality: %s to %s", start_date, end_date)
    return _frame_from_hourly_response(
        _get_with_retry(config.OPEN_METEO_AIR_QUALITY_URL, params),
        "open_meteo_air_quality",
    )


def fetch_open_meteo_weather_history(
    start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    end_date = end_date or config.AS_OF_DATE
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(config.WEATHER_HOURLY_VARS),
        "timezone": config.TIMEZONE,
    }
    logger.info("Fetching Open-Meteo weather features: %s to %s", start_date, end_date)
    return _frame_from_hourly_response(
        _get_with_retry(config.OPEN_METEO_WEATHER_ARCHIVE_URL, params),
        "open_meteo_weather_features",
    )


def fetch_open_meteo_weather_trend(
    start_date: str, end_date: Optional[str] = None
) -> pd.DataFrame:
    end_date = end_date or config.AS_OF_DATE
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(config.WEATHER_TREND_DAILY_VARS),
        "timezone": config.TIMEZONE,
    }
    logger.info(
        "Fetching Open-Meteo daily weather trend: %s to %s", start_date, end_date
    )
    data = _get_with_retry(config.OPEN_METEO_WEATHER_ARCHIVE_URL, params)
    if "daily" not in data or "time" not in data["daily"]:
        raise ValueError(
            f"Open-Meteo response did not contain daily time data: {data.keys()}"
        )
    frame = pd.DataFrame(data["daily"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame["source"] = "open_meteo_weather_trend"
    return frame


def save_raw(
    df: pd.DataFrame,
    purpose: str,
    provider: str,
    frequency: str,
    start_date: str,
    end_date: str,
) -> Path:
    if provider != "open_meteo":
        raise ValueError("The active pipeline accepts only provider='open_meteo'.")
    config.ensure_data_directories()
    output = config.DATA_RAW_DIR / (
        f"karak_{purpose}_{provider}_{frequency}_{start_date}_to_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(output, index=False)
    logger.info("Saved %d rows to %s", len(df), output)
    return output


def run_open_meteo_backfill(
    air_quality_start: str = config.AIR_QUALITY_START_DATE,
    weather_trend_start: str = config.WEATHER_TREND_START_DATE,
    end_date: str = config.AS_OF_DATE,
) -> list[Path]:
    outputs = []
    air = fetch_open_meteo_air_quality(air_quality_start, end_date)
    outputs.append(
        save_raw(
            air, "aqi_training", "open_meteo", "hourly", air_quality_start, end_date
        )
    )
    weather = fetch_open_meteo_weather_history(air_quality_start, end_date)
    outputs.append(
        save_raw(
            weather,
            "weather_features",
            "open_meteo",
            "hourly",
            air_quality_start,
            end_date,
        )
    )
    trend = fetch_open_meteo_weather_trend(weather_trend_start, end_date)
    outputs.append(
        save_raw(
            trend, "weather_trend", "open_meteo", "daily", weather_trend_start, end_date
        )
    )
    return outputs


if __name__ == "__main__":
    for output in run_open_meteo_backfill():
        print(output)
