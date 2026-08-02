# API documentation — Karak Open-Meteo-only pipeline

**Location:** Karak, Pakistan — Sabir Abad area (`33.1383653, 71.1909136`)  
**Timezone:** `Asia/Karachi`  
**Active provider:** Open-Meteo only  
**Last updated:** 2026-07-31

## Active endpoints

### 1. Open-Meteo Air Quality API

`https://air-quality-api.open-meteo.com/v1/air-quality`

Used for the hourly AQI training/analysis file:
`karak_aqi_training_open_meteo_hourly_<start>_to_<end>_<pull>.csv`

Requested variables are configured in `src/config.py` and include PM₂.₅, PM₁₀, CO, NO₂, SO₂, ozone, aerosol optical depth, dust, and UV index. Requests explicitly set `timezone=Asia/Karachi`. The operational historical boundary is `2022-08-05` because the first pull exposed a contiguous upstream null block from `2022-08-01 00:00` through `2022-08-04 04:00`. Those 77 timestamps were not imputed; the diagnostic file is retained outside the active raw-data directory.

Open-Meteo returns modeled/reanalysis atmospheric values, not a local Karak monitor. The output is therefore an estimate and should not be described as ground truth.

### 2. Open-Meteo Historical Weather API

`https://archive-api.open-meteo.com/v1/archive`

Used for two deliberately separate files:

- `karak_weather_features_open_meteo_hourly_<start>_to_<end>_<pull>.csv` — weather features aligned with the AQI dataset from 2022 onward.
- `karak_weather_trend_open_meteo_daily_2000-01-01_to_<end>_<pull>.csv` — daily weather summaries for long-run Karak trend analysis.

Hourly variables include temperature, humidity, dew point, precipitation, rain, pressure, cloud cover, wind speed/direction, and gusts. Daily trend variables include mean/min/max temperature, precipitation, rain, mean wind, maximum gust, and mean humidity.

### 3. Open-Meteo Forecast API

`https://api.open-meteo.com/v1/forecast`

Notebook 03 uses this same-provider endpoint only to verify that current air-quality and weather responses have the expected fields, local timezone, and non-missing values. It is not merged into the historical training file.

## Standard data practices

- Keep each raw tabular pull unchanged apart from the explicit `source` ingestion metadata column in `data/raw/`.
- Include purpose, provider, frequency, requested date range, and pull timestamp in every filename.
- Never select raw data using a generic `*.csv` glob.
- Keep the 2000-present weather-trend file separate from the 2022-present AQI training data.
- Record actual row counts, missing values, duplicate timestamps, and date ranges in executed notebooks.
- Treat modeled values as estimates, especially in a location without a local ground monitor.

## Historical sanity-check decision

The project previously compared Open-Meteo with OpenWeather and consulted a distant WAQI/Peshawar station. That exercise is preserved as historical evidence in `learning.txt` and `DATA_SOURCES.md`; it is not part of the active code or notebook workflow. The recorded comparison showed timezone normalization materially improved agreement, while remaining differences were expected model-to-model variation. The distant station was not Karak ground truth.

**Operational decision:** use Open-Meteo only. Existing secondary raw files should be archived under `data/archive/secondary_validation/` for review before irreversible deletion; they must not be loaded by the active notebooks.
