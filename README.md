# Karak AQI and weather analysis

This project uses **Open-Meteo only** for the active pipeline. Open-Meteo air quality and weather are separate endpoints from one provider; OpenWeather and WAQI/AQICN are retained only as historical sanity-check (cross reference) evidence and are no longer active dependencies.

## Notebook order

1. `notebooks/01_raw_data_check.ipynb` — raw-file naming, completeness, duplicate/time-gap checks, physical-range checks, and an evidence-based QC verdict.
2. `notebooks/02_feature_eda.ipynb` — Open-Meteo-only hourly master data, daily aggregation, AQI proxy, seasonality, weather relationships, and model-ready outputs.
3. `notebooks/03_live_open_meteo_check.ipynb` — one-provider live air-quality/weather request and timestamp/range sanity check.
4. `notebooks/04_karak_weather_trends.ipynb` — separate daily weather trend analysis for Karak from 2000 through the current pull date.


## Data files

See `PRIMARY_DATA_SOURCES.md` for the complete naming contract. In brief:

- `karak_aqi_training_open_meteo_hourly_...csv` = pollutant training/analysis data, active from 2022-08-05 onward after excluding a documented 77-hour upstream null block at the start of the first pull.
- `karak_weather_features_open_meteo_hourly_...csv` = hourly weather features for the same operational range.
- `karak_weather_trend_open_meteo_daily_...csv` = daily weather trend data, 2000 onward, intentionally separate from AQI training.

`data/` is ignored because raw downloads can be large and may be refreshed. Keep a local copy while working and document its exact filename in notebook output.



## Interpretation limits

Open-Meteo provides model/reanalysis data, not a local Karak monitor. Treat the resulting AQI as an estimate. The 2000–present weather notebook is suitable for describing modeled weather trends, not for claiming station-measured climate change without uncertainty analysis.
