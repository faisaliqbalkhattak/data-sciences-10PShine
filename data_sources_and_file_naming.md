# Data sources and file naming contract

**Active provider: Open-Meteo only.** The project uses two Open-Meteo endpoints because air quality and weather are separate products, not because they are competing providers:

1. Air Quality API — CAMS pollutant data for the operational AQI dataset.
2. Historical Weather API — ERA5/IFS weather features and daily long-run trend data.
3. Forecast API — optional same-provider live weather/AQ preview in notebook 03.

The previous OpenWeather/AQICN work was a one-time sanity check. It is not part of training, feature generation, or current validation.

## Resolved source-data gap

The first Open-Meteo AQ pull (`2022-08-01` through `2026-07-31`) contained one contiguous upstream null block from `2022-08-01 00:00` through `2022-08-04 04:00` (77 timestamps; all nine AQ variables null together). Weather values existed for those timestamps. This was treated as source unavailability, not a parsing error. The original file is retained under `data/archive/diagnostic/` and is not selected by active notebook globs. The active AQ training range begins `2022-08-05`; no values are imputed.

## Dataset purposes and names

All raw files live in the ignored `data/raw/` directory. Never infer purpose from a generic name such as `data.csv`; use the filename contract:

```text
karak_<purpose>_<provider>_<frequency>_<start>_to_<end>_<pull_timestamp>.csv
```

Examples:

| Filename pattern | Purpose | Retention |
|---|---|---|
| `karak_aqi_training_open_meteo_hourly_2022-08-05_to_<end>_<pull>.csv` | Hourly pollutant inputs/targets for AQI training and 2022–present analysis | Keep as active source |
| `karak_weather_features_open_meteo_hourly_2022-08-05_to_<end>_<pull>.csv` | Hourly weather features aligned with the AQ dataset | Keep as active source |
| `karak_weather_trend_open_meteo_daily_2000-01-01_to_<end>_<pull>.csv` | Daily weather summaries used only for long-run Karak climate/weather trends | Keep separately from AQ training |
| `data/archive/secondary_validation/*` | Historical OpenWeather/AQICN sanity-check evidence, if recovered locally | Archive; never load automatically |

The final timestamp is the pull time, not a measurement time. The date range in the filename is the requested observation range. The `source` column and notebook output must agree with the filename.

## Why the secondary source can be removed from the active pipeline

The earlier documented check found that timezone normalization fixed the major apparent misalignment (historical PM2.5 correlation improved from about 0.4 to about 0.68; daily aggregation reached about 0.77), while the remaining difference was expected model-to-model disagreement. The checks also found that OpenWeather had gaps and that the nearest WAQI station was Peshawar, not Karak, with stale readings in the recorded run. These sources were useful for the exercise but do not provide Karak ground truth.

**Decision:** the sanity check supports removing OpenWeather and WAQI/AQICN from the active workflow. Keep any existing secondary CSVs only in `data/archive/secondary_validation/` until the executed notebooks and this decision record have been reviewed. Do not delete them irreversibly until that review is complete.

## Data-quality rules

- All timestamps are requested in and interpreted as `Asia/Karachi`.
- Raw tabular pulls are never overwritten by feature engineering; ingestion adds only the explicit `source` metadata column.
- Each pull is a new file with an ingestion timestamp.
- Notebook 01 checks duplicate timestamps, missing values, missing hourly periods, negative concentrations, and source/date-range consistency. The initial 77-hour AQ source gap is documented as diagnostic evidence; the active file starts after that gap.
- Notebook 02 uses only Open-Meteo air quality plus Open-Meteo weather features.
- Notebook 03 verify that the one active provider returns current Karak air-quality and weather observations with the expected timezone and fields. This is not a cross-provider comparison.
- Notebook 04 uses only the separately named daily weather-trend file; it must not be mixed into AQI training without an explicit methodological decision.
- A model/reanalysis value is not a local ground-station observation. Karak has no local ground monitor in this project.
