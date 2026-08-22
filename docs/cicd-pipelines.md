# CI/CD Pipelines

GitHub Actions pipeline architecture, scheduling, data flow, and troubleshooting history.

---

## Pipeline Architecture

```
karAQI (code repo)                     karAQI-data (data repo)
───────────────────                    ────────────────────────

Open-Meteo (keyless)
      │
      ▼
feature_pipeline.yml (hourly :01)
  → Incremental fetch (~1 row) → Build features → DuckDB feature store
      │
      ▼
training_pipeline.yml (daily 00:00 UTC)
  → Reads from DuckDB feature store
  → Trains 2 models: Ridge, XGBoost
  → Champion comparison (only promotes if better)
  → Registers in MLflow
  → Pushes model files + eval JSON
      │
      ├──► models/*.joblib             ──────► karAQI-data/models/
      └──► data/model_eval.json      ──────► karAQI-data/data/

forecast_pipeline.yml (hourly :04)
  → Downloads model from karAQI-data (or MLflow)
  → Reads features from DuckDB feature store
  → Fetches Open-Meteo AQ forecast (always fresh, for comparison)
  → Exports static_forecast.json
      │
      └──► data/static_forecast.json ──────► karAQI-data/data/

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

## Pipeline Schedules

| Pipeline | Cron (UTC) | Pakistan Time | Purpose | Duration |
|---|---|---|---|---|
| Feature | `1 * * * *` | XX:01 +5h | Incremental fetch, feature build, tests | ~1 min |
| Forecast | `4 * * * *` | XX:04 +5h | Inference + JSON export | ~45 sec |
| Training | `0 0 * * *` | 05:00 +5h | Incremental fetch, train 4 models, register, export | ~8 min |

### Why These Specific Minutes?

- **Feature at :01** — First to run each hour. Fetches latest data before forecast needs it.
- **Forecast at :04** — Waits for feature pipeline to complete (typically takes ~1 min).
- **Training at 00:00 UTC (05:00 PKT)** — Runs once daily, takes ~5 min. Scheduled before hourly pipelines to avoid resource contention.

---

## Training Pipeline Details

The training pipeline runs two training scripts:

```yaml
- name: Train models from the feature store
  run: |
    python -m src.train --store        # daily models (Ridge, RF, XGBoost, SARIMA, LSTM)
    python -m src.train_hourly --store  # hourly 30-output models (Ridge, XGBoost)
```

### Champion Comparison

The registry compares new model metrics against the current champion:

```
New model trained → Compare hourly_points RMSE against current champion
  → If better: promote (new version in MLflow, old version kept)
  → If worse: keep old champion, log the comparison
  → If no champion exists: promote automatically
```

The `model_eval.json` records which model was promoted and why.

### Model Files Pushed to karAQI-data

```yaml
cp models/aqi_forecast_hourly_ridge.joblib /tmp/karAQI-data/models/
cp models/aqi_forecast_hourly_xgb.joblib /tmp/karAQI-data/models/
cp models/aqi_forecast_hourly_models.json /tmp/karAQI-data/models/
cp models/aqi_forecast_models.json /tmp/karAQI-data/models/
```

---

## Forecast Pipeline Details

The forecast pipeline loads the model from MLflow and features from DuckDB:

```yaml
- name: Fetch trained model from karAQI-data
  run: |
    # Downloads model files from karAQI-data repo
    cp /tmp/karAQI-data/models/aqi_forecast_hourly_ridge.joblib models/

- name: Export pre-computed forecast (static JSON)
  run: python -m src.export_forecast --source store
```

The `--source store` flag means features come from DuckDB feature store. The Open-Meteo AQ forecast reference is always fetched fresh (independent of the `--source` flag) for the live comparison on the dashboard.

### No Silent Fallbacks

The pipeline fails loudly if:
- MLflow registry has no champion → `FileNotFoundError` with instructions
- DuckDB feature store is empty → `FileNotFoundError` from `load_latest_hourly()`
- Model file not found in karAQI-data → `ERROR` exit code

This ensures the pipeline either works correctly or fails visibly.

---

## Feature Pipeline Details

The feature pipeline runs hourly and fetches only new data:

```yaml
- name: Fetch raw Open-Meteo data incrementally (keyless)
  run: python -m src.build_features --fetch --incremental
```

Incremental fetching: checks the last timestamp in existing CSVs, only pulls new data. First run pulls everything (~4 years), subsequent runs pull ~1 row. Safe because Open-Meteo reanalysis data is immutable.

---

## GitHub Actions Timing Caveat

Cron triggers are best-effort, not precise. Workflows are frequently delayed 5–30 minutes (sometimes longer) due to:

- Repository activity level (less active repos get lower priority)
- Platform load (shared runner infrastructure)
- Workflow concurrency (queues behind previous runs)

See [github/community#156282](https://github.com/orgs/community/discussions/156282).

**Impact on dashboard:** If the hero AQI shows a previous hour's value, the CI pipeline was delayed. The data is correct — it reflects the most recent successful run.

---

## Troubleshooting History

| Issue | Root cause | Fix |
|---|---|---|
| `lstm skipped (TensorFlow not installed)` | TensorFlow not in requirements-training.txt | Added TensorFlow to requirements-training.txt |
| Feature pipeline tests fail (`ModuleNotFoundError: mlflow`) | MLflow in feature pipeline requirements | Used `--ignore=tests/test_model_registry.py` in feature pipeline |
| Push to karAQI-data fails (`Permission denied`) | github-actions[bot] token lacks write access | Created fine-grained PAT (karAQI-data-push) with Contents: R/W |
| Push to karAQI-data fails (`Authentication failed`) | Token not set or expired | Regenerated fine-grained token, updated secret |
| Push rejected (`Updates were rejected`) | Concurrent workflows push to same branch | Added `git pull --rebase` before push |
| `Unstaged changes` error on pull-rebase | Untracked files in working directory | Stash untracked files before rebase |
| Dashboard shows stale predictions | Push to karAQI-data silently failing | Changed `exit 0` to `exit 1` on clone failure |
| `pivot.columns` length mismatch | model_eval.json schema changed | Fixed pivot table column assignment |
| `ImageMixin.image() got unexpected keyword argument` | Streamlit version doesn't support `use_container_width` | Used `use_column_width` instead |
| `ValueError: '>=' not supported between numpy.ndarray and Timestamp` | Type mismatch in date comparison | Cast to proper types before comparison |
| `DATA_REPO_TOKEN` auth failure | Classic token with wrong scope | Switched to fine-grained token scoped to karAQI-data only |
| Model files stale in karAQI-data | Training pipeline only saved models as ephemeral artifacts | Added model file push to training pipeline |
| Dashboard loads stale model | Forecast pipeline loaded from local .joblib (Aug 19) | Forecast pipeline now downloads from karAQI-data before inference |
