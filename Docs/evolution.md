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
14. [Switching from IQAir to Open-Meteo](#14-switching-from-iqair-to-open-meteo)
15. [Current Architecture](#15-current-architecture)
16. [Limitations and Honest Assessment](#16-limitations-and-honest-assessment)
17. [What Was Learned](#17-what-was-learned)

---

## 1. Starting Point: The Assignment

The 10Pearls Shine Program asked for an end-to-end machine learning system that:

- Fetches weather and pollutant data from external APIs
- Computes features (time-based + derived)
- Stores features in a feature store (Hopsworks or Vertex AI)
- Trains multiple ML models (Random Forest, Ridge, TensorFlow/PyTorch)
- Evaluates with RMSE, MAE, R²
- Stores models in a model registry
- Builds a dashboard with real-time predictions for the next 3 days
- Includes SHAP/LIME explanations for model interpretability
- Supports alerts for hazardous AQI levels
- Runs automated pipelines (Airflow or GitHub Actions)
- Is 100% serverless and free

The assignment explicitly states that R² around 0.7 at 72 hours is difficult and recommends aiming for the highest realistic result. No absolute MSE/RMSE threshold is defined.

Location: Karak, Pakistan (33.1384° N, 71.1909° E). Karak has **no local ground-air-quality monitor**, which shapes every design decision in this project.

---

## 2. Research Phase: Reading the Papers

Five research papers were reviewed to select the four model families for training. The selection was not done by majority voting — it used a weighted scoring algorithm with seven criteria (published accuracy, stability, relevance to modeled data, small-sample fitness, interpretability, assignment stack fit, deployment feasibility). See [modeling-evaluation.md](modeling-evaluation.md) for the full scoring table.

### Papers Reviewed

| Paper | Key Finding |
|---|---|
| Comparative Analysis of Forecasting Models for AQI Prediction | LSTM, ARIMA/SARIMA, Prophet are top performers. Data cleaning = "ultimate success." |
| Forecasting AQI with Multiple Linear Regression | MLR best at 99.96% in-sample; PM2.5, PM10, SO₂ dominate AQI. |
| ML Models for Daily AQI Prediction: In-depth Analysis | Extra Trees most effective; overfitting common in trees → cap depth. |
| Mapping Socioeconomic AQI Disparities (Satellite) | Modeled/gridded satellite data can represent AQI where no ground monitor exists. |
| Real-Time AQI Estimation Using Gas Sensors | IQR outlier detection; sensor-based real-time estimation with DL/ML. |

### Final Model Selection (stratified by family)

| Family | Model | Score | Role |
|---|---|---|---|
| Boosting | XGBoost | 8.40 | Primary ML model |
| Bagged trees | Random Forest | 8.10 | Robust ensemble |
| Statistical | SARIMA | 7.10 | Seasonal baseline |
| Deep learning | LSTM | 6.70 | TensorFlow requirement |

Plus persistence (naive baseline) and Ridge (mandated by assignment).

---

## 3. Data Collection: Fetching 26 Years of Weather Data

### Weather Trends (2000–present)

Fetched from Open-Meteo's Historical Weather Archive API (`archive-api.open-meteo.com`). Daily variables: temperature (mean/max/min), precipitation, rain, wind speed, wind gusts, humidity. This data powers the weather insights section of the dashboard — 26 years of Karak climate patterns showing hot summers (40°C+), monsoon precipitation (July–September), and dust storm season (March–June).

### AQI Training Data (2022-08-05–present)

Two separate Open-Meteo endpoints provide hourly data:

1. **Air Quality API** (`air-quality-api.open-meteo.com`) — PM₂.₅, PM₁₀, CO, NO₂, SO₂, O₃, aerosol optical depth, dust, UV index
2. **Weather Archive API** — temperature, humidity, dew point, precipitation, pressure, cloud cover, wind speed/direction/gusts

The AQI data starts from 2022-08-05 because the first pull revealed a contiguous upstream null block from 2022-08-01 through 2022-08-04 (77 timestamps, all nine AQ variables null). Rather than imputing, the project starts after the gap.

All timestamps normalized to `Asia/Karachi` (UTC+5) at ingestion time.

### Incremental Fetching

The initial implementation pulled the full 4-year history on every run. This was wasteful — 365 unnecessary API calls per year downloading data that hadn't changed. The fix: incremental fetching. The feature pipeline runs hourly and only fetches data from the day after the last timestamp in the existing raw files. Since Open-Meteo reanalysis data is immutable (historical values never change), this is safe. At training time, only ~1 hour of new data is fetched instead of 4 years.

### Why Open-Meteo?

Initially, the project also used OpenWeather and WAQI/AQICN for cross-validation. This changed after the sanity check (see next section). Open-Meteo was chosen because:

- **Free, no API key** — satisfies the serverless requirement
- **Long historical record** — back to 2021 for AQ, decades for weather
- **Clean JSON output** — no binary file parsing
- **Built-in AQI conversions** — US EPA and European standards
- **Consistent timezone handling** — explicit `timezone=Asia/Karachi`

See [data-sources.md](data-sources.md) for the full API documentation.

---

## 4. Data Quality: The Sanity Check That Changed Everything

Before committing to a single data source, the project compared Open-Meteo with OpenWeather Air Pollution API and WAQI/AQICN (Peshawar station). This was a critical exercise that shaped the entire architecture.

### What Was Found

1. **Timezone mismatch was the dominant issue.** Open-Meteo returns Asia/Karachi time; OpenWeather returns UTC. Without normalization, PM₂.₅ correlation was ~0.4. After fixing: **0.68 hourly, 0.77 daily**.

2. **Cross-source correlation is realistic for modeled data.** Ozone (r=0.697) and PM₂.₅ (r=0.680) agree best. CO (r=0.481) and SO₂ (r=0.437) agree least. These are expected model-to-model disagreements for a rural area with complex topography.

3. **WAQI data was stale.** The Peshawar station (nearest to Karak, ~120 km away) reported data from March 2025 — over 4 months old. Not usable as a live reference.

4. **OpenWeather had gaps** (~2.6% of hours missing in the backfill).

### The Decision

Use **Open-Meteo only**. The sanity check proved that:

- Timezone normalization was the key fix, not source switching
- Remaining cross-source differences are model-to-model noise, not bugs
- No ground station exists in Karak, so cross-source comparison ≠ accuracy validation

OpenWeather and WAQI data were archived under `data/archive/secondary_validation/` and removed from the active pipeline.

---

## 5. Feature Engineering: From Raw Data to Model Inputs

### Daily Features (39 features)

- **Pollutant inputs:** PM₂.₅, PM₁₀, CO, NO₂, SO₂, O₃, aerosol optical depth, dust, UV index
- **Weather inputs:** temperature, humidity, dew point, precipitation, rain, pressure, cloud cover, wind speed/direction/gusts
- **Calendar features:** month sin/cos, day-of-year sin/cos, weekday, is_weekend
- **Lag features:** AQI lag 1/2/3/7 days, PM₂.₅ lag 1/2/3/7 days
- **Rolling features:** AQI and PM₂.₅ rolling mean (3/7 days), rolling std (3/7 days)

### Hourly Features (33 features)

Same pollutant and weather variables at hourly resolution, plus:
- **AQI lag features:** 1h, 24h
- **AQI rolling features:** 6h mean, 24h mean, 6h std, 24h std
- **Calendar features:** hour sin/cos, day-of-week sin/cos, month sin/cos, is_weekend

### Feature Store

DuckDB (serverless, no API key). The store consumes the validated feature builders verbatim rather than re-implementing them, preserving the training/inference contract. Schema version tracked in a `meta` table for drift detection.

---

## 6. AQI Calculation: Implementing the US EPA Standard

`src/aqi.py` implements the full US EPA AQI breakpoint method:

- **PM₂.₅:** 24-hour average, May 2024 breakpoint update
- **Ozone:** 8-hour rolling window maximum
- **CO:** 8-hour rolling window maximum
- **SO₂:** 1-hour maximum
- **NO₂:** 1-hour maximum
- Unit conversion, truncation, breakpoint interpolation, category assignment

The AQI for each hour is the **maximum valid pollutant sub-index**. Categories: Good (0–50), Moderate (51–100), Unhealthy for Sensitive Groups (101–150), Unhealthy (151–200), Very Unhealthy (201–300), Hazardous (301+).

This was a major correction from the initial "aqi_proxy" approach, which applied daily means to all pollutants without proper averaging windows. The project audit identified this as the most critical issue and it was fixed before any model training.

---

## 7. Model Selection: Choosing What to Train

See [modeling-evaluation.md](modeling-evaluation.md) for the full weighted scoring methodology.

### Daily Models (per-horizon)

Each horizon (+1, +2, +3 days) gets its own best model:

| Horizon | Best Model | RMSE | R² |
|---|---|---:|---:|
| +1 day | XGBoost | 15.78 | 0.513 |
| +2 days | XGBoost | 19.40 | 0.263 |
| +3 days | Ridge | 20.66 | 0.173 |

### Hourly Model (all output groups)

Ridge regression with alpha=10.0 (selected via purged `TimeSeriesSplit(gap=72)`). Ridge was chosen over LSTM, XGBoost, and Random Forest for all three output groups because it had the lowest RMSE and MAE.

---

## 8. Training and Evaluation: The Numbers

### Rolling-Origin Protocol

Instead of a single train/test split (which can be lucky or unlucky), the project uses **expanding rolling-origin evaluation**:

1. Train on data up to day N
2. Test on days N+1 to N+3 (72 hours)
3. Roll forward: train on data up to day N+7
4. Test on days N+8 to N+10
5. Repeat 3 times

The **72-hour embargo** means there's a 3-day gap between training and test data — preventing data leakage (today's lag features won't accidentally contain tomorrow's values).

### Hourly Ridge Results (from live dashboard)

| Group | RMSE | MAE | R² | Cat. Accuracy | High-AQI Recall |
|---|---:|---:|---:|---:|---:|
| Hourly points (1–24h) | 11.16 | 8.59 | 0.326 | 90.0% | 69.3% |
| Six-hour means (25–48h) | 18.52 | 13.95 | -0.026 | 77.7% | 55.3% |
| Twelve-hour means (49–72h) | 19.11 | 15.33 | -0.110 | 66.8% | 48.4% |

### Why R² is Negative for Long-Range

Negative R² means the model is worse than simply predicting the mean. This is expected at 48–72h because:

- AQI is highly variable at hourly resolution
- The model only sees historical data (no future weather)
- 72 hours is the maximum horizon the assignment requires

The release gate is **relative**: Ridge beats persistence on RMSE and MAE for every output group. The twelve-hour block group has negative R² but substantially lower RMSE than persistence (19.11 vs 27.50).

### What the Metrics Mean in Practice

An RMSE of 11.16 for hourly points means the typical prediction error is about ±11 AQI points. On a 0–500 scale, this is useful for:

- Identifying whether tomorrow will be "Good" vs "Unhealthy for Sensitive Groups"
- Flagging when AQI is trending upward toward hazardous levels
- Providing a 72-hour trend line for planning outdoor activities

It is **not** precise enough to say "AQI will be exactly 145 at 3 PM" — the prediction might be 134 or 156. But it's consistently better than persistence (which just says "tomorrow = today").

---

## 9. Validation: The Hardening Phase

The project audit identified several critical issues that were fixed before any infrastructure work:

1. **AQI target was a proxy, not EPA standard** → Implemented proper `src/aqi.py` with pollutant-specific averaging, truncation, and breakpoint interpolation
2. **Temporal evaluation had leakage** → Added 72-hour purge gap and rolling-origin protocol
3. **No persistence baseline** → Added persistence and seasonal persistence benchmarks
4. **Missing metrics** → Added MSE, category accuracy, macro F1, high-AQI recall
5. **No inference contract** → Created `src/inference_hourly.py` with schema validation
6. **Dependencies incomplete** → Pinned Python 3.11, added all required packages
7. **No automated tests** → Built test suite (41 tests)

The [modeling-readiness.md](modeling-readiness.md) checklist was completed and accepted before moving to infrastructure.

---

## 10. Infrastructure: Feature Store and Model Registry

### Why DuckDB Instead of Hopsworks?

The assignment names Hopsworks or Vertex AI. The project chose DuckDB because:

- Hopsworks free tier bills/freezes when hourly pipelines run
- Dependency conflicts (`imp` module) on Windows
- RPC disconnects during hourly writes
- Mentor explicitly allowed alternatives: *"Use the env flag so Hopsworks runs in Codespaces and is off locally. Just keep a working local fallback."*

DuckDB satisfies the serverless requirement: a single file, no server, no API key, no always-on process.

### Why MLflow File-Backed Instead of Hosted?

MLflow runs in file-backed mode (`models/mlruns/`): no tracking server, no paid tier, no API keys. The training pipeline runs on demand, registers model versions, and the dashboard loads the registered champion. A hosted MLflow server would be overkill.

---

## 11. Dashboard: Building the Interface

The Streamlit dashboard at [kaqindex.streamlit.app](https://kaqindex.streamlit.app/) was built in phases:

1. **Initial version:** Runtime inference — loads model, fetches live data, predicts on every page load. Slow (10–30 seconds per visit).

2. **Static JSON serving:** CI pre-computes predictions and stores them in `karAQI-data`. Dashboard fetches via raw GitHub URLs with 5-minute cache. Near-instant page load. Zero runtime inference.

3. **IQAir comparison:** Initially scraped IQAir's website for reference AQI. Failed due to rate limiting from cloud/CI IPs.

4. **Open-Meteo reference:** Replaced IQAir with Open-Meteo's free AQ forecast API. Same US EPA AQI scale, no rate limits, 96h forecast trimmed to 72h.

5. **UI polish:** Poppins/Inter fonts, mobile-responsive layout, compact hero section, AQI category color coding, bar charts, SHAP explanations.

### Dashboard Data Flow

```
karAQI-data/static_forecast.json
  ├── outputs[]       → 30 prediction values (the model)
  ├── ref_forecast[]  → 72h hourly US AQI from Open-Meteo
  ├── current_aqi     → Current hour AQI from observed data
  └── ref_now         → Open-Meteo live AQI

Dashboard reads this JSON → no runtime computation → instant load
```

---

## 12. CI/CD: Automating Everything

See [cicd-pipelines.md](cicd-pipelines.md) for the full troubleshooting history.

### Three Pipelines

1. **Feature pipeline** (hourly `:01`) — Fetches latest data, builds features, runs tests. Fast (<1 min).

2. **Forecast pipeline** (hourly `:04`) — Runs Ridge inference, fetches Open-Meteo AQ forecast reference, exports `static_forecast.json` to karAQI-data.

3. **Training pipeline** (daily `00:00 UTC`) — Trains all models (daily + hourly), registers in MLflow, exports `model_eval.json` to karAQI-data.

### Issues Encountered and Fixed

| Issue | Root Cause | Fix |
|---|---|---|
| `No module named pytest` in CI | CRLF line endings in requirements.txt | `.gitattributes` with `eol=lf` |
| Streamlit Cloud crash | Same CRLF issue | Same fix |
| Tests fail before data exists | Tests ran before data generation | Reordered pipeline: data first, tests after |
| pyarrow build failure on Streamlit | mlflow → pyarrow → cmake needed | Separate `app/requirements.txt` with lean deps |
| Push failures to karAQI-data | Missing DATA_REPO_TOKEN secret | Created fine-grained PAT, added as secret |
| IQAir rate limiting | Scraping from CI IPs blocked | Replaced with Open-Meteo AQ forecast API |
| Pipeline silently succeeds on push failure | `exit 0` on clone failure | Changed to `exit 1` with clear error messages |
| Stale forecast on dashboard | Push failure + silent exit | Now fails loudly; dashboard shows latest successful JSON |
| Training pipeline push race condition | Unstaged changes before pull | Added `git stash` before pull-rebase |

---

## 13. The Data Repo Split

Hourly CI commits (forecast JSON, model eval) were cluttering the main karAQI repo commit history. The solution:

- **karAQI** (this repo) — Source code, models, notebooks, tests
- **karAQI-data** — CI-generated JSON files (static_forecast.json, model_eval.json)

The dashboard reads from karAQI-data via `raw.githubusercontent.com` URLs. CI workflows push to karAQI-data using a fine-grained PAT (`DATA_REPO_TOKEN`).

---

## 14. Switching from IQAir to Open-Meteo

The initial comparison source was IQAir (scraping their website). This was replaced because:

1. **Rate limiting:** IQAir blocks anonymous scraping aggressively. GitHub Actions runners hit 429 errors after a few requests.
2. **Fragile scraping:** HTML scraping breaks when the site changes its layout.
3. **Open-Meteo is better suited:** Free, keyless, 96h hourly US AQI forecast, same EPA scale, no rate limits.

The IQAir scraping code (`fetch_iqair.py`) was deleted. The IQAir daily pipeline (`iqair_pipeline.yml`) was removed. All IQAir references in the codebase were cleaned up.

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
  → src/train.py + src/train_hourly.py (train models)
  → src/model_registry.py (MLflow register)
  → src/export_eval.py (evaluation JSON)
      │
      ├──► models/*.joblib, *.keras  ──────► karAQI-data/models/
      └──► data/model_eval.json      ──────► karAQI-data/data/

forecast_pipeline.yml (hourly :04)
  → Downloads model from karAQI-data/models/
  → src/export_forecast.py (inference + JSON export)
      │
      └──► data/static_forecast.json ──────► karAQI-data/data/

feature_pipeline.yml (hourly :01)
  → Incremental fetch (~1 row)
  → Build features → DuckDB feature store

                              karAQI-data/data/
                              ├── static_forecast.json (hourly predictions)
                              ├── model_eval.json (daily evaluation metrics)
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

- A working end-to-end ML pipeline that fetches data, trains models, stores features, registers models, runs CI/CD, and serves predictions on a live dashboard
- A 72-hour AQI forecast using US EPA standards on modeled data
- A 100% serverless, free system with no API keys and no paid services

### What This Project Is Not

- **Not ground-truth accuracy.** All data comes from Open-Meteo CAMS modeled/reanalysis products. Karak has no local monitor. The dashboard shows this disclaimer.
- **Not a real-time monitoring system.** Predictions are updated hourly by CI, not in real-time. There's a 5–35 minute delay depending on GitHub Actions queue time.
- **Not a replacement for professional AQI monitoring.** This is an academic project demonstrating ML pipeline capabilities.

### Known Limitations

- **Hourly R² at 48–72h is negative** — the model is worse than the mean for long-range hourly predictions. The twelve-hour block means are useful as smoothed guidance but not highly explanatory.
- **LSTM underperforms Ridge** — deep learning needs more data than the 28,190 hourly training rows provide. The LSTM is retained for comparison only.
- **SARIMA has negative R² at all horizons** — seasonal ARIMA doesn't capture AQI dynamics well with this data volume.
- **No uncertainty intervals** — the model produces point estimates, not confidence bands.
- **GitHub Actions delays** — the dashboard may show a previous hour's AQI during peak platform load.

---

## 17. What Was Learned

1. **Timezone normalization is critical.** A 5-hour offset between data sources dropped correlation from 0.68 to 0.4. Always normalize timestamps at ingestion time.

2. **Model-to-model correlation ≠ ground-truth accuracy.** Two APIs agreeing on PM₂.₅ doesn't mean either is right. Cross-source validation is useful for sanity-checking, not for accuracy claims.

3. **Persistence is a strong baseline.** "Tomorrow = today's AQI" is hard to beat at 72 hours. Any complex model must prove it's better than this simple heuristic.

4. **CI pipeline failures should be loud.** Silent `exit 0` on push failures meant the dashboard showed stale data for hours with no visible error. Failing loudly fixed this.

5. **Separate dashboard and CI dependencies.** Streamlit Cloud installs everything in requirements.txt. MLflow's pyarrow dependency needs cmake, which Streamlit Cloud doesn't have. A lean `app/requirements.txt` solves this.

6. **Data repos keep commit history clean.** Hourly JSON commits to the main repo would bury real code changes. A separate data repo with raw GitHub URL access is the standard pattern.

7. **Static serving beats runtime inference for dashboards.** Pre-computing predictions via CI and serving them as JSON gives every visitor instant page load, regardless of model complexity.

8. **Cron triggers are best-effort.** GitHub Actions scheduled workflows are frequently delayed 5–30 minutes. Design the system to tolerate this (show latest successful data, not fail).
