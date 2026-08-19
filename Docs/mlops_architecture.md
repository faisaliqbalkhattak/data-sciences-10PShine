# MLOps architecture — Pearls AQI Predictor (Karak)

**Status:** implemented 2026-08-15 · **Due:** Sep 02, 2026
**Companion docs:** `../MODELING_READINESS.md` (modeling gate), `../API_documentation.md` (data/API contract)

This document records the infrastructure decisions, the component map, how each
assignment bullet is satisfied, and how to reproduce everything. It exists so
the project remains **100% serverless / free** (criterion C7 in
`../../Docs/model_selection_methodology.md`) while still demonstrating a real
feature store, model registry, and automated CI/CD.

---

## 1. Decision records

### D1 — Feature store: DuckDB (local) instead of Hopsworks / Vertex AI

The assignment names **Hopsworks or Vertex AI**. The project chose a
**DuckDB-backed store** (`data/feature_store/karak_feature_store.duckdb`) as a
documented alternative, which the mentor session explicitly permits:

> "That's a good approach… Use the env flag so Hopsworks runs in Codespaces and
> is off locally. Just keep a working local fallback for when it's off, and
> note this in your report." — Hafsa Imtiaz (mentor Q&A, on Hopsworks Windows/CI
> problems; alternatives allowed when documented)

Reasons (recorded in the project's own notes and this repo's history):

| Problem | Evidence in repo |
|---|---|
| Hopsworks free tier bills/freezes when an hourly feature pipeline runs | `../../Docs/learning from others/issues_with_the_workflow.md`, `../learning.txt` |
| `imp` module / dependency conflicts on Windows | same notes; mentor Q&A question 5 |
| RPC disconnects during hourly writes | same notes |
| Other students in the cohort hit the same walls; mentors allow alternatives | `../../Docs/learning from others/answers to students questions from hafsa.md` |

Why DuckDB specifically:

- **Serverless by construction**: a single file, no server, no always-on
  process, no account, no API key — satisfies C7 exactly.
- **Windows-safe** (project runs on Python 3.11.9 / Windows; Feast-style
  tooling has poor Windows support).
- **Queryable**: training, inference, and the dashboard query the same store,
  so the schema contract cannot drift.
- **Idempotent backfill**: `backfill --replace` regenerates the store from the
  latest keyless Open-Meteo data, which is exactly what an hourly feature
  pipeline needs.

The store **consumes the validated feature builders**
(`train_hourly.build_hourly_training_frame`, `train.build_training_frame`)
verbatim rather than re-implementing them, preserving the contract recorded in
the model manifests.

### D2 — Model registry: MLflow (local file-backed)

MLflow runs in **file-backed mode** (`models/mlruns/`): no tracking server, no
paid tier, no API keys. This aligns with the assignment's "serverless"
requirement because, per the mentor session, *"serverless mainly applies to the
data/ML pipelines, storage, and automation"* — the training pipeline runs on
demand (locally or on GitHub Actions), registers model versions, and the
dashboard loads the registered `champion`. A hosted MLflow server would be
overkill and is not required.

Registered models: `aqi-hourly-ridge`, `aqi-daily-h1`, `aqi-daily-h2`,
`aqi-daily-h3` — each with a `champion` alias pointing at the latest accepted
version.

### D3 — Automation: GitHub Actions (no Airflow)

GitHub Actions was already the project's CI and is named in the assignment
("Apache Airflow, GitHub Actions, or similar"). It needs no paid resources and
no secrets (Open-Meteo is keyless).

---

## 2. Component map

```
Open-Meteo (keyless)                     GitHub Actions
      │                                        │
      ▼                                        ▼
src/ingest.py  ── raw CSV ──► src/build_features.py ── processed CSV
      │                                        │
      ▼                                        ▼
src/feature_store.py (DuckDB)  ◄───── feature_pipeline.yml (hourly cron)
      │                                        ▲
      ├── hourly_raw / hourly_observations / hourly_features / daily_features
      │
      ▼
src/train_hourly.py --store  ◄──── training_pipeline.yml (daily cron)
src/train.py --store
      │
      ▼
models/*.joblib + aqi_forecast_*_models.json (manifests)
      │
      ▼
src/model_registry.py (MLflow, file-backed) ── models/mlruns/
      │
      ├──────────────────────────────┐
      ▼                              ▼
app/api.py (FastAPI)        app/dashboard.py (Streamlit)
  /forecast /explain         72h chart · alerts · SHAP · comparison · EDA
```

Components:

| Component | Location | Role |
|---|---|---|
| Feature store | `src/feature_store.py` | Versioned DuckDB store; `meta` table records schema version, feature-column contract, row counts, source SHA-256 |
| Feature builder | `src/build_features.py` | Scripted notebook-02 equivalent (merge, EPA daily AQI, calendar columns); used by CI and the live dashboard path |
| Registry | `src/model_registry.py` | MLflow registration of selected models + metrics; `champion` alias; `list`/`load` helpers |
| Hourly pipeline | `.github/workflows/feature_pipeline.yml` | Runs every hour (`0 * * * *`), on push (tests), and manually; backfills store; uploads artifact |
| Training pipeline | `.github/workflows/training_pipeline.yml` | Runs daily (`15 1 * * *`) and manually; backfills → trains `--store` → registers → uploads registry artifact |
| API | `app/api.py` | FastAPI: `/health`, `/forecast`, `/explain`, `/registry`, `/latest-origin` |
| Dashboard | `app/dashboard.py` | Streamlit: 72h forecast, alerts, SHAP, comparison, EDA |

---

## 3. Assignment mapping

| Assignment bullet | Implementation |
|---|---|
| Fetch raw weather and pollutant data from external APIs | `src/ingest.py` — Open-Meteo AQ + weather archive (keyless); hourly feature pipeline refetches and rebuilds |
| Compute features: time-based (hour/day/month) + derived (AQI change/lag/rolling) | `src/build_features.py` + `train_hourly.build_hourly_training_frame` / `train.build_training_frame` (calendar sin/cos, lags, rolling mean/std) |
| Store processed features in a Feature Store | `src/feature_store.py` — DuckDB store with schema-versioned `meta` table (documented alternative to Hopsworks, D1) |
| Historical data backfill | `python -m src.feature_store backfill-hourly --replace` (34,9xx hourly rows) / `backfill-daily --replace` (1,4xx rows) from the 2022-08-05 start |
| Training pipeline fetches features/targets from the Feature Store | `python -m src.train_hourly --store`, `python -m src.train --store` |
| Experiment with ML models (RF, Ridge, TF/PyTorch) | Ridge, Random Forest, XGBoost, SARIMA, LSTM (+ persistence/seasonal baselines) |
| Evaluate with RMSE, MAE, R² | `data/processed/hourly_model_comparison.csv`, `model_comparison.csv`, rolling-origin CSVs |
| Store trained models in a Model Registry | `src/model_registry.py` (MLflow) — `aqi-hourly-ridge`, `aqi-daily-h1..h3`, `champion` alias |
| Feature pipeline runs every hour automatically | `feature_pipeline.yml` `schedule: cron "0 * * * *"` |
| Training pipeline runs daily | `training_pipeline.yml` `schedule: cron "15 1 * * *"` |
| Dashboard loads models and features from the store | `app/dashboard.py` reads the DuckDB store (or live Open-Meteo) and loads the registered `champion` (fallback: local artifact) |
| Real-time predictions for next 3 days | 30 outputs: 24 hourly points (t+1..24h) + four six-hour means (t+25..48h) + two twelve-hour means (t+49..72h) |
| Interactive dashboard with Streamlit/Gradio and Flask/FastAPI | Streamlit `app/dashboard.py` + FastAPI `app/api.py` |
| EDA to identify trends | Notebooks 01–04 + dashboard "History / EDA" tab |
| SHAP/LIME feature importance explanations | `app/explain.py` — exact `shap.LinearExplainer` for the Ridge pipeline; dashboard "Explanations (SHAP)" tab; `/explain` endpoint |
| Alerts for hazardous AQI levels | Dashboard banner + API `alerts` payload (Very Unhealthy ≥ 201 / Hazardous ≥ 301) |
| Support multiple forecasting models (statistical → deep learning) | SARIMA → Ridge → RF/XGB → LSTM, all on the same holdout/rolling-origin protocols |

---

## 4. How to reproduce

```bash
cd development
# 1. Environment (Python 3.11.9 only)
py -3.11 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
# Optional LSTM: .venv/Scripts/python -m pip install -r requirements-lstm.txt

# 2. Data + features
python -m src.ingest                       # raw Open-Meteo pull (keyless)
python -m src.build_features               # scripted notebook-02 (or run notebook 02)
python -m src.feature_store backfill-hourly --replace
python -m src.feature_store backfill-daily --replace
python -m src.feature_store stats          # inspect stored tables + schema

# 3. Training (from the feature store)
python -m src.train --store
python -m src.train_hourly --store

# 4. Registry
python -m src.model_registry register-hourly
python -m src.model_registry register-daily
python -m src.model_registry list

# 5. API + dashboard
uvicorn app.api:app --reload --port 8000          # terminal 1
streamlit run app/dashboard.py                    # terminal 2

# 6. Tests
python -m pytest -q                               # 34 checks
```

### Verification checklist

- [ ] `pytest -q` → 34 passed
- [ ] `python -m src.feature_store stats` shows `hourly_raw`, `hourly_observations`, `hourly_features`, `daily_features` with schema version 1
- [ ] `python -m src.model_registry list` shows `aqi-hourly-ridge` (champion) + `aqi-daily-h1..h3`
- [ ] `GET /forecast` returns 30 outputs; `GET /explain?output=0` returns SHAP values; dashboard renders the 72h chart, alerts, SHAP, comparison, EDA tabs
- [ ] Workflows present: `feature_pipeline.yml` (hourly) and `training_pipeline.yml` (daily) — triggerable via `workflow_dispatch` in GitHub Actions
- [ ] No API keys or paid services anywhere in the stack (C7)

---

## 5. Honest limitations

- **Modeled target, not ground truth.** All features and the AQI target come
  from Open-Meteo CAMS modeled/reanalysis data. Karak has no local monitor in
  this project; the dashboard shows this disclaimer.
- **CI artifact persistence.** GitHub Actions runners are ephemeral; the
  feature store and MLflow registry produced in CI are uploaded as workflow
  artifacts (downloadable) rather than persisted to a paid object store. The
  local copies are the working store/registry for the demo. This is consistent
  with the free-stack constraint.
- **Rolling-origin R² at 72 hours.** The twelve-hour block group is honest but
  not highly explanatory (see `../MODELING_READINESS.md`); the release gate is the
  relative RMSE/MAE beat of the deterministic baselines.
- **Registry scope.** The registry registers the *selected* sklearn models
  (Ridge hourly, best-per-horizon daily). The comparison LSTM/SARIMA artifacts
  remain in `models/` + manifests for the comparison story; they are not
  promoted to `champion`.
