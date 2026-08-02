"""Central configuration for the Karak Open-Meteo-only pipeline."""

from __future__ import annotations
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
CITY_NAME = "Karak"
LOCATION_LABEL = "sabir_abad"
LATITUDE = 33.1383653
LONGITUDE = 71.1909136
TIMEZONE = "Asia/Karachi"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive" / "secondary_validation"
for _directory in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_ARCHIVE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_HOURLY_VARSF = [
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
WEATHER_HOURLY_VARS = [
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
WEATHER_TREND_DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_mean",
    "wind_gusts_10m_max",
    "relative_humidity_2m_mean",
]
# The first pull showed a contiguous upstream null block from 2022-08-01 00:00
# through 2022-08-04 04:00. Starting the active training file at the first
# complete date rather than imputing an unavailable model output.
AIR_QUALITY_START_DATE = "2022-08-05"
WEATHER_TREND_START_DATE = "2000-01-01"
AS_OF_DATE = date.today().isoformat()
HISTORICAL_START_DATE = AIR_QUALITY_START_DATE
