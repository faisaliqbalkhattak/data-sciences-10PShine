# Evaluation — The Full Project Journey

This document tells the complete story of how karAQI was built, from the initial research papers to the live dashboard. It is written so that anyone — a reviewer, a teammate, or a future maintainer — can understand every decision, why it was made, and what was learned along the way.

---

## Table of Contents

1. [Starting Point: The Assignment](#1-starting-point-the-assignment)
2. [Research Phase: Reading the Papers](#2-research-phase-reading-the-papers)
3. [Data Collection: Fetching 26 Years of Weather Data](#3-data-collection-fetching-26-years-of-weather-data)
4. [Data Quality: The Sanity Check That Changed Everything](#4-data-quality-the-sanity-check-that-changed-everything)
5. [Feature Engineering: From Raw Data to Model Inputs](#5-feature-engineering-from-raw-data-to-model-inputs)
6. [AQI Calculation: Implementing the US EPA Standard](#6-aqi-calculation-implementing-the-us-epa-standard)
7. [Model Selection: Choosing What to Train](#7-model-selection-choosing-what-to-train)
8. [Training and Evaluation: The Numbers](#8-training-and-evaluation-the-numbers)
9. [Validation: The Hardening Phase](#9-validation-the-hardening-phase)
10. [Infrastructure: Feature Store and Model Registry](#10-infrastructure-feature-store-and-model-registry)
11. [Dashboard: Building the Interface](#11-dashboard-building-the-interface)
12. [CI/CD: Automating Everything](#12-cicd-automating-everything)
13. [The Data Repo Split](#13-the-data-repo-split)
14. [Champion Comparison and Model Promotion](#14-champion-comparison-and-model-promotion)
15. [Current Architecture](#15-current-architecture)
16. [Limitations and Honest Assessment](#16-limitations-and-honest-assessment)
17. [What Was Learned](#17-what-was-learned)

---

## 1. Starting Point: The Assignment

The 10Pearls Shine internship project description asked for:

> "Let's predict the Air Quality Index (AQI) in your city in the next 3 days, using a 100% serverless stack."

Required technology stack: Python, Scikit-learn, TensorFlow, Hopsworks or Vertex AI, Apache Airflow or GitHub Actions, Streamlit, Flask, AQICN or OpenWeather APIs, SHAP, Git.

Key features required:
- Feature pipeline (hourly)
- Historical data backfill
- Training pipeline (daily)
- Model registry
- CI/CD automation
- Web dashboard
- EDA, SHAP, hazardous alerts
- Multiple forecasting models (statistical to deep learning)

---

## 2. Research Phase: Reading the Papers

Five research papers were reviewed to select models:

| Paper | Models | Key finding |
|---|---|---|
| Comparative Analysis of AQI Forecasting | LSTM, ARIMA/SARIMA, Prophet | "Best models are LSTM, ARIMA/SARIMA, Prophet" |
| ML Models for Daily AQI Prediction | Extra Trees, RF, LightGBM | Extra Trees most effective; O3/PM10/PM2.5 most important |
| Forecasting AQI with MLR | MLR, LSTM, CNN | MLR best scenario 99.96% accuracy |
| Mapping AQI Disparities (Sentinel-5P) | Satellite-derived AQI | Modeled data viable where no ground monitor exists |
| Real-Time AQI Estimation | DL and ML models | IQR outlier detection; sensor-based estimation |

A weighted scoring system selected 4 models: **XGBoost, Random Forest, SARIMA, LSTM** — covering boosting, bagged trees, statistical, and deep learning families.

---

## 3. Data Collection: Fetching 26 Years of Weather Data

Open-Meteo was chosen over AQICN/OpenWeather because:
- Free, no API key required
- No rate limits
- Provides both historical data and forecasts
- Includes US AQI directly

Data collected:
- **Weather:** 2000–present (26 years) from Open-Meteo Archive API
- **AQI/pollutants:** 2022-08-05–present (4 years) from Open-Meteo Air Quality API
- **AQI forecast reference:** 4-day rolling forecast for live comparison

---

## 4. Data Quality: The Sanity Check That Changed Everything

An initial sanity check comparing Open-Meteo with OpenWeather and WAQI revealed:
- OpenWeather historical data had timezone inconsistencies
- WAQI data was stale for Karak
- Open-Meteo was the only consistent, fresh source

This led to the decision to use Open-Meteo exclusively.

---

## 5. Feature Engineering: From Raw Data to Model Inputs

43 features engineered from raw hourly data:
- **Pollutant concentrations:** PM2.5, PM10, CO, NO₂, SO₂, O₃, AOD, dust, UV index
- **Weather:** temperature, humidity, dew point, precipitation, rain, pressure, cloud cover, wind speed/direction/gusts
- **Time-based:** hour_sin, hour_cos, day_of_week, month, is_weekend
- **Lag features:** 1h, 3h, 6h, 12h, 24h lags of key pollutants
- **Rolling statistics:** 6h, 12h, 24h rolling means and standard deviations

---

## 6. AQI Calculation: Implementing the US EPA Standard

`src/aqi.py` implements the US EPA AQI calculation:
- Pollutant-specific averaging windows (8h for O₃/CO, 1h for SO₂/NO₂, 24h for PM)
- EPA unit conversions and truncation
- Breakpoint interpolation with the May 2024 PM₂.₅ update
- Category assignment (Good through Hazardous)

Two target definitions:
- **Daily:** `aqi_us_epa` — complete calendar-day aggregation
- **Hourly:** `aqi_hourly_rolling` — rolling-hour estimate from EPA sub-indices

---

## 7. Model Selection: Choosing What to Train

See `docs/modeling-evaluation.md` for the full scoring methodology.

Final shortlist (stratified by family):
| Family | Model | Score |
|---|---|---|
| Boosting | XGBoost | 8.40 |
| Bagged trees | Random Forest | 8.10 |
| Statistical | SARIMA | 7.10 |
| Deep learning | LSTM | 6.70 |

Baselines kept: Persistence (naive), Ridge/MLR.

---

## 8. Training and Evaluation: The Numbers

### Hourly 30-output forecast (the model serving predictions)

Four models trained: Ridge, Random Forest, XGBoost, LSTM.

Rolling-origin evaluation (3 expanding folds, 168h test windows, 72h embargo):

| Model | Hourly RMSE | Six-hour RMSE | Twelve-hour RMSE |
|---|---|---|---|
| **Ridge** | **10.96** | **18.14** | **20.05** |
| Persistence | 14.49 | 22.25 | 25.17 |
| Seasonal persistence | 18.14 | 23.03 | 25.11 |

### Daily 1/2/3-day forecast

| Model | 1d RMSE | 2d RMSE | 3d RMSE |
|---|---|---|---|
| **XGBoost** | **15.85** | **19.37** | **20.53** |
| Random Forest | 16.33 | 19.50 | 20.65 |
| Ridge | 16.66 | 19.92 | 20.55 |
| Persistence | 18.73 | 22.91 | 25.61 |
| LSTM | 28.56 | 26.73 | 24.38 |
| SARIMA | 29.60 | 27.03 | 38.85 |

---

## 9. Validation: The Hardening Phase

After the initial audit flagged several issues:
- **Temporal leakage:** Fixed with horizon-aware purge/embargo (72h gap)
- **AQI target definition:** Documented as Open-Meteo modeled, not ground truth
- **Baseline comparison:** Added persistence and seasonal persistence
- **Metrics expansion:** Added category accuracy, macro F1, high-AQI recall

---

## 10. Infrastructure: Feature Store and Model Registry

### Feature Store (DuckDB)

Chosen over Hopsworks/Vertex AI because:
- Serverless, no API key, no paid tier
- Works on Windows (local dev)
- Same function: stores processed features with versioning

The feature store is used by the **training and forecast pipelines**, not the dashboard.

### Model Registry (MLflow)

File-backed MLflow (no tracking server):
- Registers the champion model with version tracking
- The forecast pipeline loads the champion from MLflow for inference
- Champion comparison ensures only better models are promoted

---

## 11. Dashboard: Building the Interface

Streamlit dashboard with:
- Hero section showing current AQI
- Two tabs: "My model" (predictions) and "Open-Meteo" (live reference)
- 30-output forecast chart with Open-Meteo comparison
- SHAP explanations (LinearExplainer for Ridge)
- Model evaluation metrics
- Weather insights (26-year trends, AQI seasonality)

---

## 12. CI/CD: Automating Everything

Three GitHub Actions pipelines:
- Feature pipeline (hourly :01) — incremental data fetch, DuckDB update
- Training pipeline (daily 00:00 UTC) — train 4 models, champion comparison, MLflow register
- Forecast pipeline (hourly :04) — load model from MLflow, read features from DuckDB, export JSON

---

## 13. The Data Repo Split

Data was moved from the main karAQI repo to a separate karAQI-data repo to keep the main commit history clean. The training pipeline pushes model files and eval JSON to karAQI-data. The forecast pipeline downloads models from karAQI-data for inference.

---

## 14. Champion Comparison and Model Promotion

The training pipeline implements a proper champion comparison:

1. Train today's models (Ridge, RF, XGBoost, LSTM)
2. Evaluate on rolling-origin holdout
3. Compare best model's RMSE against current champion in MLflow
4. If better: promote (new version, old version kept)
5. If worse: keep old champion, log the comparison

This prevents model regression — a bad training day doesn't overwrite a good model.

---

## 15. Current Architecture

```
karAQI (code repo)                     karAQI-data (data repo)
───────────────────                    ────────────────────────

Open-Meteo (keyless)
      │
      ▼
training_pipeline.yml (daily 00:00 UTC)
  → src/ingest.py (incremental fetch)
  → src/build_features.py (feature engineering)
  → src/feature_store.py (DuckDB rebuild)
  → src/train.py + src/train_hourly.py (train 4 models each)
  → Champion comparison (only promote if better)
  → src/model_registry.py (MLflow register)
  → src/export_eval.py (evaluation JSON)
      │
      ├──► models/*.joblib, *.keras  ──────► karAQI-data/models/
      └──► data/model_eval.json      ──────► karAQI-data/data/

forecast_pipeline.yml (hourly :04)
  → Downloads model from karAQI-data/models/
  → Reads features from DuckDB feature store
  → Runs inference (MLflow → local fallback)
  → Fetches Open-Meteo AQ forecast (always fresh)
  → Exports static_forecast.json
      │
      └──► data/static_forecast.json ──────► karAQI-data/data/

feature_pipeline.yml (hourly :01)
  → Incremental fetch (~1 row)
  → Build features → DuckDB feature store

                              karAQI-data/data/
                              ├── static_forecast.json (hourly predictions)
                              ├── model_eval.json (evaluation metrics)
                              └── models/ (trained .joblib + .keras files)
                                      │
                                      ▼
                              Dashboard (Streamlit Cloud)
                              Reads via raw GitHub URLs
                              Near-instant page load, zero runtime inference
```

---

## 16. Limitations and Honest Assessment

### What This Project Is

An end-to-end ML pipeline for AQI forecasting using modeled/reanalysis data. It demonstrates:
- Automated data collection and feature engineering
- Multiple model families with proper evaluation
- CI/CD with GitHub Actions
- Feature store and model registry
- Live dashboard with SHAP explanations

### What This Project Is Not

- It does not predict ground-truth AQI (Karak has no ground station)
- It does not use paid cloud services (100% free/serverless)
- It does not implement real-time streaming (pre-computed JSON)

### Honest Metrics

The hourly Ridge model achieves RMSE ~11 on the primary output group, which means predictions are typically ±11 AQI points off. Category accuracy is ~90% for hourly points. The model beats persistence baselines but the absolute accuracy depends on the quality of the Open-Meteo modeled data.

---

## 17. What Was Learned

1. **Open-Meteo > AQICN/OpenWeather** for this use case — free, keyless, no rate limits, includes AQI forecast
2. **DuckDB > Hopsworks** for a student project — same function, zero cloud dependency
3. **Static JSON serving** is the right architecture for a dashboard — zero runtime inference, near-instant load
4. **Champion comparison** prevents model regression — don't overwrite good models with bad training days
5. **No silent fallbacks** — fail loudly so you know when something is wrong
6. **Incremental fetching** saves CI time — don't re-download 4 years of data every hour
7. **Ridge often beats complex models** on small datasets — simplicity wins when n is small
8. **GitHub Actions cron is best-effort** — expect 5-30 minute delays, design accordingly
