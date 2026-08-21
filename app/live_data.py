"""Latest-observation loading for the dashboard and API.

Two sources are supported:

* ``store`` -- the last N hours of raw observations from the DuckDB feature
  store (the data the scheduled feature pipeline maintains).
* ``live``  -- a fresh keyless Open-Meteo pull of the last seven days, merged
  and validated with the same feature contract (used by the dashboard's live
  mode and as the API's default when the store is empty).

Both return the same raw observed hourly frame (``time`` + base pollutant and
weather columns) so ``inference_hourly.predict_latest`` and the feature
builders see an identical contract.

Open-Meteo is the project's observation provider (history fetch only). The
independent forecast reference is Open-Meteo's AQ forecast (free, keyless,
same US EPA AQI scale).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from src import config  # noqa: E402
from src.aqi import calculate_hourly_us_aqi  # noqa: E402
from src.build_features import merge_raw_frames  # noqa: E402
from src.ingest import (  # noqa: E402
    fetch_open_meteo_air_quality,
    fetch_open_meteo_weather_history,
)

#: Open-Meteo AQ forecast endpoint.
AQ_FORECAST_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

#: Cache the parsed forecast briefly so dashboard refreshes don't hammer the API.
_FORECAST_CACHE_TTL_SECONDS = 30 * 60
_forecast_cache: dict = {"ts": None, "series": None}

#: Rows of history to serve as inference input (7 days is enough for the
#: 24-hour EPA windows and all derived lag/rolling features).
DEFAULT_LOOKBACK_HOURS = 168


def _current_hour_local() -> pd.Timestamp:
    """Current hour in the project timezone (naive, aligned to the data grid)."""
    return pd.Timestamp.now(tz=config.TIMEZONE).floor("h").tz_localize(None)


def load_latest_hourly(
    source: str = "store", hours: int = DEFAULT_LOOKBACK_HOURS
) -> pd.DataFrame:
    """Return the raw observed hourly frame ending at the latest origin.

    The returned frame always ends at the most recent completed hour (the
    "current hour"): the store is read when it is fresh, and topped up with a
    live pull when it lags behind the clock, so a refresh after an hour passes
    advances the forecast origin to the next hour. The model and its feature
    logic are untouched -- only the input data is kept current.
    """
    source = (source or "store").lower()
    if source == "store":
        from src.feature_store import get_hourly_raw

        frame = get_hourly_raw()
        if frame.empty:
            raise FileNotFoundError(
                "Feature store is empty. Run `python -m src.feature_store "
                "backfill-hourly --replace` or use source='live'."
            )
        latest = pd.Timestamp(frame.index[-1])
        current_hour = _current_hour_local()
        if latest < current_hour - pd.Timedelta(hours=1):
            # The store lags the clock (e.g. the hourly CI pipeline has not run
            # yet). Pull the last seven days fresh so the forecast starts from
            # the current hour; all lag/rolling features need at most 24 hours.
            logger.info(
                "Store origin %s is stale vs %s; topping up with a live pull",
                latest,
                current_hour,
            )
            end = config.AS_OF_DATE
            start = (pd.Timestamp.today() - pd.Timedelta(days=7)).date().isoformat()
            air = fetch_open_meteo_air_quality(start, end)
            weather = fetch_open_meteo_weather_history(start, end)
            frame = merge_raw_frames(air, weather)
        # The upstream analysis can include hours later in the current day that
        # have not happened yet. Anchor the origin to the current hour so the
        # forecast always starts from now and rolls forward on refresh.
        frame = frame[frame.index <= current_hour]
        return frame.tail(hours)

    if source == "live":
        end = config.AS_OF_DATE
        start = (pd.Timestamp.today() - pd.Timedelta(days=7)).date().isoformat()
        air = fetch_open_meteo_air_quality(start, end)
        weather = fetch_open_meteo_weather_history(start, end)
        master = merge_raw_frames(air, weather)
        # A trailing in-progress hour may contain upstream nulls; drop it so
        # the inference contract sees only complete observations.
        observed = [
            c
            for c in master.columns
            if c in config.AIR_QUALITY_HOURLY_VARS + config.WEATHER_HOURLY_VARS
        ]
        master = master.dropna(subset=observed)
        if master.empty:
            raise RuntimeError("Live Open-Meteo pull returned no complete hours.")
        # Anchor to the current hour (see the store branch).
        master = master[master.index <= _current_hour_local()]
        return master.tail(hours)

    raise ValueError(f"Unknown data source: {source!r} (expected 'store' or 'live').")


def latest_origin(source: str = "store", hours: int = DEFAULT_LOOKBACK_HOURS) -> Optional[pd.Timestamp]:
    """Timestamp of the newest observation available for the given source."""
    frame = load_latest_hourly(source, hours=hours)
    return pd.Timestamp(frame.index[-1]) if len(frame) else None


def current_conditions(hourly: pd.DataFrame) -> dict:
    """Latest observed hour: weather readings plus the dominant pollutant.

    Returns a dict with ``time``, the three weather readings used in the hero
    strip (``temperature_2m``, ``relative_humidity_2m``, ``wind_speed_10m``),
    ``main_pollutant`` (the pollutant with the highest EPA sub-index at that
    hour) and its ``concentration``. Missing values become
    ``None``; a failure degrades gracefully to PM2.5.
    """
    if hourly is None or hourly.empty:
        return {}
    latest = hourly.iloc[-1]
    out = {"time": str(hourly.index[-1])}
    for col in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m"):
        value = latest.get(col)
        out[col] = round(float(value), 1) if pd.notna(value) else None

    pollutant = "pm2_5"
    try:
        cols = [c for c in config.AIR_QUALITY_HOURLY_VARS if c in hourly.columns]
        sub = calculate_hourly_us_aqi(hourly[cols])
        row = sub.iloc[-1]
        if row.notna().any():
            pollutant = str(row.idxmax()).removeprefix("aqi_")
    except Exception:  # noqa: BLE001 - best-effort fallback to PM2.5
        pass
    out["main_pollutant"] = pollutant
    raw = latest.get(pollutant)
    out["concentration"] = round(float(raw), 1) if pd.notna(raw) else None
    return out


def _fetch_open_meteo_aq_forecast() -> pd.Series:
    """Fetch 72h hourly US AQI forecast from Open-Meteo (free, keyless).

    Returns a ``pd.Series`` of hourly AQI values indexed by local time.
    This fetches from the Open-Meteo AQ forecast API (free, keyless).
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
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        times = data.get("hourly", {}).get("time", [])
        aqis = data.get("hourly", {}).get("us_aqi", [])
        values = [float(a) for a in aqis if a is not None]
        if not values:
            raise RuntimeError("No AQI values in Open-Meteo response")
        index = pd.to_datetime(times[:len(values)])
        return pd.Series(values, index=index, name="aqi")
    except Exception as exc:
        raise RuntimeError(f"Open-Meteo AQ forecast fetch failed: {exc}") from exc


def reference_forecast_aqi() -> pd.Series:
    """Reference hourly AQI forecast for Karak from Open-Meteo.

    Returns a ``pd.Series`` of hourly AQI values indexed by local time.
    Cached for 30 minutes to avoid hammering the API on dashboard refreshes.
    """
    now_ts = time.time()
    series = _forecast_cache.get("series")
    if series is None or now_ts - (_forecast_cache.get("ts") or 0) > _FORECAST_CACHE_TTL_SECONDS:
        series = _fetch_open_meteo_aq_forecast()
        _forecast_cache.update({"ts": now_ts, "series": series})
        return series
    # Re-anchor the cached values to the current hour.
    origin = _current_hour_local()
    return pd.Series(series.values, index=pd.date_range(origin, periods=len(series), freq="h"), name="aqi")
