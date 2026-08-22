# karAQI — Karak AQI Predictor

This project forecasts a **US EPA AQI calculated from Open-Meteo modeled concentrations** for Karak, Pakistan. Open-Meteo is the active data provider; no local ground monitor exists in Karak.

## Architecture

The system uses a **static JSON serving architecture** — predictions are pre-computed by CI pipelines and stored in the [`karAQI-data`](https://github.com/faisaliqbalkhattak/karAQI-data) repository. The Streamlit dashboard fetches these JSON files via GitHub raw URLs, giving every visitor a near-instant page load with zero runtime inference.

```
karAQI (this repo)                    karAQI-data (data repo)
─────────────────────                 ─────────────────────────
CI workflows generate JSONs   ───►   data/static_forecast.json
  feature_pipeline (hourly)           data/model_eval.json
  training_pipeline (daily)
  forecast_pipeline (hourly)
                                     ◄── Dashboard reads via raw URLs
```

| Workflow | Schedule (UTC) | What it does | Pushes to |
|---|---|---|---|
| `feature_pipeline.yml` | Hourly at `:01` | Fetch Open-Meteo data, build features, run tests | artifact upload |
| `forecast_pipeline.yml` | Hourly at `:04` | Run Ridge inference, fetch Open-Meteo AQ forecast, export JSON | karAQI-data |
| `training_pipeline.yml` | Daily at `00:00` | Train all models, register in MLflow, export eval JSON | karAQI-data |

### GitHub Actions timing caveat

GitHub Actions cron triggers are **best-effort, not precise**. Scheduled workflows are frequently delayed by 5–30 minutes (and occasionally longer) due to platform load and runner availability — see [github/community#156282](https://github.com/orgs/community/discussions/156282).

**What this means for the dashboard:** If you see a previous hour's AQI as the hero section value, it is because the CI pipeline that generates the forecast JSON was delayed by GitHub Actions infrastructure. The data is still correct — it just reflects the most recent successful pipeline run, which may be 30–60 minutes behind clock time during peak periods.

The pipelines run in the correct order (features → forecast → training), so data consistency is maintained regardless of absolute timing.

### Dashboard (Streamlit Community Cloud)

- Deployed at [kaqindex.streamlit.app](https://kaqindex.streamlit.app/)
- Reads all data from `karAQI-data` repo — no local data files needed
- **My model tab**: Our current-hour AQI as primary, Open-Meteo as secondary, model's next-hour prediction as tertiary
- **Live tab**: Open-Meteo live AQI as primary, our current-hour AQI as secondary
- Reference comparison: Open-Meteo AQ forecast (free, keyless, same US EPA AQI scale)
- Font: Poppins/Inter (Google Material Design inspired)

## MLOps phase

- **Feature store** — `src/feature_store.py`: serverless DuckDB store, keyless
- **Model registry** — `src/model_registry.py`: MLflow with local file-backed store
- **Automated CI/CD** — three workflows: feature (hourly), forecast (hourly), training (daily)
- **Static serving** — pre-computed JSONs in karAQI-data, zero runtime inference

## Environment setup

Run all commands from this `development/` directory. Python 3.11.x is the only supported version.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt
```

On Linux/macOS, replace `.venv/Scripts/python` with `.venv/bin/python`.

### Per-workflow requirements

| File | Used by | Packages |
|---|---|---|
| `requirements.txt` | Full dev environment | All |
| `requirements-feature.txt` | Feature pipeline CI | 9 packages |
| `requirements-forecast.txt` | Forecast pipeline CI | 8 packages |
| `requirements-training.txt` | Training pipeline CI | 16 packages |
| `app/requirements.txt` | Streamlit Cloud | 11 packages |

## Workflow

1. `python -m src.build_features --fetch` — fetch Open-Meteo data
2. `python -m src.feature_store backfill-hourly --replace` — backfill feature store
3. `python -m src.train --store` — train daily models
4. `python -m src.train_hourly --store` — train hourly models
5. `python -m src.model_registry register-hourly` — register champion models
6. `python -m src.export_forecast --source live` — generate forecast JSON
7. `python -m src.export_eval` — generate model evaluation JSON
8. `pytest -q` — run automated checks

### Dashboard quickstart

```bash
streamlit run app/dashboard.py
```

## AQI target definition

`src/aqi.py` calculates US EPA AQI using the official breakpoint method, including the May 2024 PM2.5 breakpoint update. The target is a **modeled estimate** from Open-Meteo, not station-measured AQI (Karak has no local ground monitor).

## Hourly forecast contract

Each forecast origin produces 30 outputs:
- `aqi_plus_01h` through `aqi_plus_24h`: hourly AQI points
- `aqi_mean_25_30h` through `aqi_mean_43_48h`: six-hour block means
- `aqi_mean_49_60h` and `aqi_mean_61_72h`: twelve-hour block means

## Data files

- `karak_aqi_training_open_meteo_hourly_...csv` — modeled pollutant inputs
- `karak_weather_features_open_meteo_hourly_...csv` — aligned weather features
- `karak_weather_trend_open_meteo_daily_...csv` — long-run weather analysis

## Deployment

1. Push to GitHub (`faisaliqbalkhattak/karAQI`)
2. [Streamlit Cloud](https://share.streamlit.io) → New app → `karAQI` → `app/dashboard.py`
3. Ensure `karAQI-data` repo exists with the same owner for CI data commits
4. Set `DATA_REPO_TOKEN` secret in karAQI repo settings for CI push access

## Interpretation limits

Open-Meteo provides modeled/reanalysis data, not a local Karak monitor. Results describe model-to-model agreement, not ground-truth station accuracy.
