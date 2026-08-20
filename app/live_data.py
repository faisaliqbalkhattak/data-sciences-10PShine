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
independent forecast reference is IQAir's hourly forecast for Karak.
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

#: IQAir city page for Karak (server-rendered hourly forecast table).
IQAIR_URL = (
    "https://www.iqair.com/sg/air-quality/pakistan/khyber-pakhtunkhwa/karak"
)
IQAIR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

#: IQAir rate-limits aggressively (HTTP 429); cache the parsed table briefly
#: and re-anchor it to the current hour so refreshes do not hammer the site.
IQAIR_CACHE_TTL_SECONDS = 30 * 60
_iqair_cache: dict = {"ts": None, "series": None}

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
    hour, like IQAir's widget) and its ``concentration``. Missing values become
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


def _fetch_iqair_hourly_table() -> pd.Series:
    """Fetch and parse IQAir's server-rendered hourly forecast for Karak.

    The city page renders its hourly AQI forecast as an HTML table whose first
    column is ``Now`` (the current hour) followed by 71 labeled hours. The
    values are IQAir's ``US AQI+`` -- the US EPA AQI scale (same categories,
    colors and breakpoints as this project's target) computed on hourly
    averages -- so the line is directly comparable to the model's forecast.

    Returns a ``pd.Series`` of hourly AQI values indexed by local time
    starting at the current hour (the same origin convention as the model
    forecast). Retries with backoff on IQAir's aggressive HTTP 429s.
    """
    import re

    last_error: Optional[Exception] = None
    for attempt in range(4):
        try:
            response = requests.get(IQAIR_URL, headers=IQAIR_HEADERS, timeout=30)
            if response.status_code == 429:
                time.sleep(2**attempt * 3)
                continue
            response.raise_for_status()
            html = response.text
            table_start = html.find("Hourly forecast")
            if table_start == -1:
                raise RuntimeError("IQAir page did not include the hourly forecast table.")
            table_end = html.find("</table>", table_start)
            if table_end == -1:
                raise RuntimeError("IQAir hourly forecast table not found.")
            section = html[table_start:table_end]
            # The AQI values live inside aqi-bg-* divs with a <p> containing the number.
            # The HTML has nested divs between the aqi-bg class and the <p>, so use
            # a dotall match across the intermediate tags.
            values: list[float] = []
            for m in re.finditer(r"aqi-bg-[a-z-]+.*?<p[^>]*>\s*(\d+)\s*</p>", section, re.S):
                values.append(float(m.group(1)))
            if not values:
                raise RuntimeError("No AQI values found in the IQAir hourly forecast table.")
            origin = _current_hour_local()
            index = pd.date_range(origin, periods=len(values), freq="h")
            return pd.Series(values, index=index, name="aqi")
        except Exception as exc:  # noqa: BLE001 - retry transient failures
            last_error = exc
            time.sleep(2**attempt * 3)
    raise RuntimeError(f"IQAir hourly forecast fetch failed: {last_error}")


def iqair_forecast_aqi() -> pd.Series:
    """IQAir's own hourly AQI forecast for Karak (cached, re-anchored to now).

    Returns a ``pd.Series`` of hourly AQI values indexed by local time from the
    current hour. The parsed table is cached for :data:`IQAIR_CACHE_TTL_SECONDS`
    and re-anchored to the current hour on each call, so the reference line
    starts from the same origin as the model forecast without hammering IQAir.
    """
    now = time.time()
    series = _iqair_cache.get("series")
    if series is None or now - (_iqair_cache.get("ts") or 0) > IQAIR_CACHE_TTL_SECONDS:
        series = _fetch_iqair_hourly_table()
        _iqair_cache.update({"ts": now, "series": series})
        return series
    # Re-anchor the cached values to the current hour (AQI is persistent, so a
    # forecast fetched up to 30 min ago still represents the near future well).
    origin = _current_hour_local()
    return pd.Series(series.values, index=pd.date_range(origin, periods=len(series), freq="h"), name="aqi")
