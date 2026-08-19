# Karak AQI and weather analysis

This project forecasts a **US EPA AQI calculated from Open-Meteo modeled concentrations** for the Sabir Abad area of Karak, Pakistan. Open-Meteo is the active provider; OpenWeather and WAQI/AQICN remain historical sanity-check evidence only and are not active dependencies.

## Current implementation status

The pre-infrastructure modeling gate is complete:

- official US EPA AQI calculation in `src/aqi.py`
- pollutant-specific rolling windows, unit conversions, EPA truncation, breakpoint interpolation, and AQI categories
- direct +1, +2, and +3 day targets
- hourly multi-resolution forecasting: 24 hourly points, four six-hour means, and two twelve-hour means
- persistence and seasonal-persistence baselines, tuned Ridge, comparison LSTM, Random Forest, XGBoost, and SARIMA evaluation
- chronological holdouts plus expanding rolling-origin evaluation with 72-hour purge gaps
- MSE, RMSE, MAE, R², category accuracy, macro F1, and high-AQI recall
- tested local inference contract in `src/inference_hourly.py`
- pinned dependencies and one supported interpreter: Python 3.11.9
- automated AQI, data-contract, split-gap, inference, and metadata tests

## MLOps phase (implemented 2026-08-15)

- **Feature store** — `src/feature_store.py`: a serverless, keyless DuckDB store (`data/feature_store/karak_feature_store.duckdb`) holding the exact validated feature frames plus raw observations. Historical backfill is idempotent (`backfill --replace`). See `Docs/mlops_architecture.md` for why this is the documented alternative to Hopsworks.
- **Feature building as code** — `src/build_features.py` reproduces notebook 02 so the automated pipeline and the dashboard's live path build features without executing a notebook.
- **Model registry** — `src/model_registry.py`: MLflow with a local file-backed store (`models/mlruns/`, no server, no API key). Registers `aqi-hourly-ridge` and `aqi-daily-h1..h3` with a `champion` alias.
- **Automated CI/CD** — `.github/workflows/feature_pipeline.yml` (hourly schedule) rebuilds features + backfills the store; `.github/workflows/training_pipeline.yml` (daily schedule) trains from the store and registers models in MLflow. Both need no secrets (Open-Meteo is keyless) and upload their artifacts for inspection.
- **Web dashboard** — `app/dashboard.py` (Streamlit): 72-hour forecast chart, 30-output table, hazardous-AQI alerts, SHAP explanations, model comparison, and EDA tabs.
- **API backend** — `app/api.py` (FastAPI): `/health`, `/forecast`, `/explain`, `/registry`, `/latest-origin`.

All components consume the validated contracts in `MODELING_READINESS.md` and make no station-measured accuracy claim.

## Environment setup

Run all commands from this `development/` directory. Python 3.11.x is the only supported version; `.python-version`, `pyproject.toml`, CI, and the notebooks use that same version.

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt
# Optional LSTM support, in the same environment:
.venv/Scripts/python -m pip install -r requirements-lstm.txt
```

On Linux/macOS, replace `.venv/Scripts/python` with `.venv/bin/python`.

## Workflow

1. `python -m src.ingest` — download Open-Meteo air-quality and weather inputs.
2. `python build_notebooks.py` — rebuild the executable notebooks after source changes.
3. `notebooks/01_raw_data_check.ipynb` — check file identity, cadence, duplicates, missing values, and physical ranges.
4. `notebooks/02_feature_eda.ipynb` — join the active files, calculate official US EPA AQI, and write daily model features. (Or run `python -m src.build_features`, the scripted equivalent.)
5. `python -m src.feature_store backfill-hourly --replace && python -m src.feature_store backfill-daily --replace` — backfill the feature store.
6. `python -m src.train` — train and evaluate daily models using the purge-aware split (`--store` reads from the feature store).
7. `python -m src.train_hourly` — train the 30-output hourly models and rolling-origin report (`--store` reads from the feature store).
8. `python -m src.model_registry register-hourly && python -m src.model_registry register-daily` — register models in MLflow.
9. `pytest -q` — run the automated checks.
10. Use `src.inference_hourly.predict_latest` for the local 30-output inference contract; do not duplicate feature engineering in an API/dashboard.

### Dashboard and API quickstart

```bash
# Backend (terminal 1)
uvicorn app.api:app --reload --port 8000

# Dashboard (terminal 2)
streamlit run app/dashboard.py
```

The dashboard defaults to the **feature store** data source and can switch to a fresh **live Open-Meteo** pull. It can also route through the FastAPI backend (checkbox in the sidebar). The API docs are at `http://127.0.0.1:8000/docs`.

Raw and generated data are ignored by Git. Keep the exact input filename and pull date in experiment notes when reporting results.

## AQI target definition

`src/aqi.py` calculates `aqi_us_epa` rather than the former `aqi_proxy`:

- PM₂.₅ and PM₁₀: complete calendar-day 24-hour averages
- ozone and CO: trailing 8-hour windows
- SO₂ and NO₂: 1-hour windows
- Open-Meteo µg/m³ values are converted to the EPA table units where required
- pollutant-specific truncation happens before interpolation
- daily AQI is the maximum valid pollutant sub-index observed during that local day
- `aqi_category` and `dominant_pollutant` are retained with the target

This follows the US EPA AQI breakpoint method, including the May 2024 PM₂.₅ breakpoint update. The target is still a **modeled estimate**, not station-measured AQI, because Karak has no local ground monitor in this project.

## Forecast contract and evaluation

A model row at local date `t` is available only after the complete local day `t` has ended. The operational contract is therefore: **issue forecasts at 00:00 Asia/Karachi using the previous completed day’s feature row**. A target at horizon `h` is `aqi_us_epa(t+h)`.

The final chronological holdout removes a three-day embargo before the test origins. Hyperparameter tuning uses `TimeSeriesSplit(gap=3)`. This is a single chronological holdout with purge-aware tuning, not a claim of a full rolling-origin backtest. Metrics must be regenerated after the AQI target change; old proxy metrics and artifacts are not valid evidence for the current target.

## Hourly multi-resolution forecast

`python -m src.train_hourly` trains the separate hourly contract from `karak_aqi_open_meteo_hourly_features.csv`. Each forecast origin produces 30 outputs:

- `aqi_plus_01h` through `aqi_plus_24h`: rolling hourly AQI points for the next 24 hours
- `aqi_mean_25_30h` through `aqi_mean_43_48h`: four six-hour block means
- `aqi_mean_49_60h` and `aqi_mean_61_72h`: two twelve-hour block means

The hourly targets are calculated from pollutant-specific rolling EPA sub-indices. They are a modeled rolling-hour AQI estimate and are not the official once-per-day AQI reporting value. Inputs are historical-only: current observed features, supported lag/rolling features, and calendar features. A 72-hour purge gap is used for evaluation. The selected Ridge model is tuned with purged time-series cross-validation and validated with three expanding rolling-origin folds. Artifacts and reports are written to `models/aqi_forecast_hourly_models.json`, `data/processed/hourly_model_comparison.csv`, and `data/processed/hourly_rolling_origin_comparison.csv`.

## Data files and source documentation

See [`data_sources_and_file_naming.md`](data_sources_and_file_naming.md) for the canonical data naming contract. The older `v1_multi_source_api_specs.md` is retained as a historical superseded specification; it is not the active source contract.

- `karak_aqi_training_open_meteo_hourly_...csv` — active modeled pollutant inputs.
- `karak_weather_features_open_meteo_hourly_...csv` — aligned weather features.
- `karak_weather_trend_open_meteo_daily_...csv` — separate long-run weather analysis, not an AQI target.

## Accuracy and release interpretation

The assignment requires RMSE, MAE, and R² but does not define an absolute MSE/RMSE threshold. Mentor guidance says that R² near 0.7 at 72 hours is difficult and recommends the highest realistic result. We therefore require the selected model to beat persistence and seasonal persistence on RMSE and MAE for every output group under rolling-origin evaluation. Ridge passes that relative gate. Its rolling-origin results are: RMSE 11.16 for the first 24 hourly points, 18.52 for the six-hour means, and 19.11 for the twelve-hour means; corresponding MSE values are 138.76, 378.43, and 426.35. The twelve-hour group has negative R² (-0.110), so the long-range result is useful but not highly explanatory. See `MODELING_READINESS.md` for the complete table and release checklist.

These are metrics against an Open-Meteo-derived modeled target. They are not ground-truth station accuracy.

## Deployment (Streamlit Community Cloud)

The dashboard deploys automatically to **Streamlit Community Cloud** (free, no server required):

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select this repo → set **Main file path** to `app/dashboard.py`.
4. Click **Deploy**. The app live-updates on every push to `main`.

The dashboard starts in **Live** mode by default (pulls fresh Open-Meteo data + IQAir comparison on each refresh). The **Store** mode reads from the local DuckDB feature store (populated by the training pipeline).

### GitHub Actions (auto-updates)

Two workflows run on a schedule with no secrets required (Open-Meteo is keyless):

| Workflow | Schedule | What it does |
|---|---|---|
| `feature_pipeline.yml` | Hourly (`:00`) | Fetches fresh Open-Meteo data, rebuilds features, backfills the DuckDB store |
| `training_pipeline.yml` | Daily (`01:15 UTC`) | Trains daily + hourly models from the store, registers champions in MLflow |

Both upload their artifacts (feature store, model registry) for inspection. The trained model binaries (`.joblib`) and manifests (`.json`) are committed to the repo so the deployed dashboard always has a working model.

### Running locally

```bash
# Dashboard (terminal 1)
streamlit run app/dashboard.py

# Optional API backend (terminal 2)
uvicorn app.api:app --reload --port 8000
```

## Interpretation limits

Open-Meteo provides modeled/reanalysis data, not a local Karak monitor. Results describe how well the model forecasts an Open-Meteo-derived US EPA AQI estimate. They must not be reported as ground-truth AQI accuracy or as “R² percent accuracy.”
