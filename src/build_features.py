"""Scripted feature building for the Karak Open-Meteo pipeline.

This module reproduces notebook 02 as a runnable command so the automated
feature pipeline (GitHub Actions) and the dashboard's live path can build the
processed feature frames without executing a notebook:

1. Load (or optionally fetch) the two active Open-Meteo raw products.
2. Merge them on the local hourly grid (``Asia/Karachi``).
3. Compute the US EPA daily AQI target (``aqi_us_epa``) with pollutant-specific
   averaging windows via ``src.aqi``.
4. Add the calendar columns consumed by the training feature builders.
5. Save ``karak_aqi_open_meteo_hourly_features.csv`` and
   ``karak_aqi_open_meteo_daily_features.csv`` to ``data/processed/``.

The saved files are exactly what ``src.feature_store`` backfills and what the
training pipelines read, so the notebook, the CLI, and the dashboard all share
one feature contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .aqi import calculate_daily_us_aqi

POLLUTANT_COLUMNS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
]
WEATHER_COLUMNS = [
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


def _newest_raw(pattern: str) -> Path:
    matches = sorted(config.DATA_RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No raw file matched {pattern!r}. Run `python -m src.ingest` or use --fetch."
        )
    return matches[-1]


def _prepare_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Set a validated hourly DatetimeIndex and drop ingestion metadata."""
    prepared = frame.copy()
    prepared["time"] = pd.to_datetime(prepared["time"])
    prepared = prepared.set_index("time")
    if "source" in prepared.columns:
        prepared = prepared.drop(columns=["source"])
    return prepared


def merge_raw_frames(
    air_quality: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Join the two Open-Meteo products on the hourly local-time grid."""
    aq = _prepare_raw(air_quality)
    wx = _prepare_raw(weather)
    master = aq.join(wx, how="inner", rsuffix="_weather").sort_index()
    if master.index.has_duplicates:
        raise ValueError("Merged hourly master contains duplicate timestamps.")
    return master


def add_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the hour/month sin-cos calendar columns used by feature builders."""
    output = frame.copy()
    output["hour_sin"] = np.sin(2 * np.pi * output.index.hour / 24)
    output["hour_cos"] = np.cos(2 * np.pi * output.index.hour / 24)
    output["month_sin"] = np.sin(2 * np.pi * output.index.month / 12)
    output["month_cos"] = np.cos(2 * np.pi * output.index.month / 12)
    return output


def build_feature_frames(
    air_quality: pd.DataFrame, weather: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(hourly_master, daily_features)`` matching notebook 02."""
    master = add_calendar_columns(merge_raw_frames(air_quality, weather))

    pollutant_cols = [c for c in POLLUTANT_COLUMNS if c in master]
    weather_cols = [c for c in WEATHER_COLUMNS if c in master]
    agg = {c: "mean" for c in pollutant_cols + weather_cols}
    for c in ("precipitation", "rain"):
        if c in agg:
            agg[c] = "sum"
    daily = master[pollutant_cols + weather_cols].resample("D").agg(agg)
    daily = daily.dropna(subset=["pm2_5"])
    daily["hour_count"] = master["pm2_5"].resample("D").count().reindex(daily.index)

    aqi_daily = calculate_daily_us_aqi(master[pollutant_cols])
    daily = daily.join(aqi_daily, how="left").dropna(subset=["aqi_us_epa"])
    return master, daily


def build_from_raw(
    air_quality_path: Optional[Path] = None, weather_path: Optional[Path] = None,
    incremental: bool = False,
) -> tuple[Path, Path]:
    """Build both processed feature CSVs from the newest raw files.

    When incremental=True and processed CSVs already exist, only the new rows
    are appended (after merging the raw frames and computing features).  The
    feature pipeline runs hourly so only ~1 row is appended per run.
    """
    aq_path = air_quality_path or _newest_raw(
        "karak_aqi_training_open_meteo_hourly_*.csv"
    )
    wx_path = weather_path or _newest_raw("karak_weather_features_open_meteo_hourly_*.csv")
    aq = pd.read_csv(aq_path, parse_dates=["time"])
    wx = pd.read_csv(wx_path, parse_dates=["time"])

    master, daily = build_feature_frames(aq, wx)
    config.ensure_data_directories()
    hourly_out = config.DATA_PROCESSED_DIR / "karak_aqi_open_meteo_hourly_features.csv"
    daily_out = config.DATA_PROCESSED_DIR / "karak_aqi_open_meteo_daily_features.csv"

    if incremental and hourly_out.exists() and daily_out.exists():
        # Append only new rows to existing processed CSVs
        existing_hourly = pd.read_csv(hourly_out, parse_dates=["time"], index_col="time")
        existing_daily = pd.read_csv(daily_out, parse_dates=["time"], index_col="time")
        new_hourly = master.loc[~master.index.isin(existing_hourly.index)]
        new_daily = daily.loc[~daily.index.isin(existing_daily.index)]
        if len(new_hourly) == 0 and len(new_daily) == 0:
            print("No new rows to append — processed CSVs are up to date.")
            return hourly_out, daily_out
        combined_hourly = pd.concat([existing_hourly, new_hourly]).sort_index()
        combined_daily = pd.concat([existing_daily, new_daily]).sort_index()
        combined_hourly.to_csv(hourly_out)
        combined_daily.to_csv(daily_out)
        print(f"Appended {len(new_hourly)} hourly, {len(new_daily)} daily rows -> {hourly_out}")
    else:
        master.to_csv(hourly_out)
        daily.to_csv(daily_out)
        print(f"Saved hourly features ({len(master)} rows) -> {hourly_out}")
        print(f"Saved daily features ({len(daily)} rows) -> {daily_out}")
    return hourly_out, daily_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch fresh raw Open-Meteo data before building features (feature pipeline mode).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: only fetch new data since the last pull, and append to processed CSVs.",
    )
    parser.add_argument("--start", default=config.AIR_QUALITY_START_DATE)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    if args.fetch:
        from .ingest import run_open_meteo_backfill

        run_open_meteo_backfill(
            air_quality_start=args.start, weather_trend_start=config.WEATHER_TREND_START_DATE,
            end_date=args.end or config.AS_OF_DATE,
            incremental=args.incremental,
        )
    build_from_raw(incremental=args.incremental)


if __name__ == "__main__":
    main()
