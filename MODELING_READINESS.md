# Modeling readiness contract

## Decision summary — 2026-08-13

The modeling foundation is complete enough to begin the next infrastructure phase, but it is **not evidence of station-measured AQI accuracy**. The active target is derived from Open-Meteo modeled/reanalysis concentrations and Karak has no local monitor in this project.

The selected hourly model is a multi-output Ridge model. It predicts the requested 30-value contract:

- 24 point rolling-AQI estimates at `t+1h` through `t+24h`;
- four six-hour means covering `t+25..t+30`, `t+31..t+36`, `t+37..t+42`, and `t+43..t+48`;
- two twelve-hour means covering `t+49..t+60` and `t+61..t+72`.

Feature-store integration, model registry, dashboard, deployment, scheduled retraining, and monitoring remain the **next phase**. They should consume this contract rather than change it silently.

## Accuracy answer

There is no universal numeric MSE/RMSE pass threshold in the supplied assignment. The assignment requires RMSE, MAE, and R², while the mentor guidance says that `R² = 0.7` at 72 hours is difficult and to aim for the highest realistic result. Therefore, a claim such as “R² above 0.7 is required” is not supported by the project requirements.

The project uses a defensible release gate instead:

> On expanding rolling-origin evaluation, the selected model must beat the strongest deterministic baseline (persistence or seasonal persistence) on RMSE and MAE for every output group. MSE is reported as RMSE², but no independent absolute MSE threshold is asserted without station ground truth or an assignment-defined limit.

### Rolling-origin evidence

Protocol: three expanding folds, 168-hour test windows, and a 72-hour embargo. Ridge regularization is selected using purged `TimeSeriesSplit(gap=72)` on each training fold only.

| Model | Output group | MSE | RMSE | MAE | R² | Category accuracy | High-AQI recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Persistence | 1–24h points | 186.04 | 12.57 | 9.73 | 0.142 | 0.872 | 0.628 |
| Ridge | 1–24h points | **138.76** | **11.16** | **8.59** | 0.326 | 0.900 | 0.693 |
| Persistence | 25–48h six-hour means | 558.79 | 22.95 | 17.91 | -0.556 | 0.697 | 0.560 |
| Ridge | 25–48h six-hour means | **378.43** | **18.52** | **13.95** | -0.026 | 0.777 | 0.553 |
| Persistence | 49–72h twelve-hour means | 790.64 | 27.50 | 22.95 | -1.269 | 0.563 | 0.516 |
| Ridge | 49–72h twelve-hour means | **426.35** | **19.11** | **15.33** | -0.110 | 0.668 | 0.484 |

### Conclusion

- **Relative accuracy gate: passed.** Ridge beats persistence on RMSE and MAE for all three output groups in rolling-origin evaluation, and it also beats the seasonal persistence benchmark.
- **Absolute 72-hour quality: conditional.** The twelve-hour block group has negative rolling-origin R², even though its RMSE/MAE are substantially better than the baselines. This means long-range forecasts are useful as smoothed guidance but are not highly explanatory.
- **Public-health deployment gate: not fully passed.** Without a local station, no claim about real-world Karak AQI accuracy can be made. High-AQI recall must remain visible in the future dashboard; it is not an acceptable hidden metric.
- **LSTM decision:** LSTM is functional and retained for comparison, but Ridge is selected for all output groups because it is more accurate and more stable on this dataset.

The daily model was also regenerated. Its chronological holdout selected XGBoost for +1/+2 days and Ridge for +3 days, with R² values of approximately 0.506, 0.283, and 0.159. These are modeled-target forecasting results, not percentage accuracy.

## Target and data contract

- Daily target: `aqi_us_epa`, calculated with pollutant-specific EPA windows, truncation, conversion, interpolation, and categories.
- Hourly target: `aqi_hourly_rolling`, a rolling-hour estimate from EPA pollutant sub-indices; it is not the official once-per-day AQI report.
- Timezone: `Asia/Karachi`; timestamps must be sorted, unique, and complete at hourly cadence.
- Inputs: historical-only observations available at the forecast origin. No future weather or pollutant values are features.
- Missing observations: fail closed; training and inference do not silently impute or move the forecast origin backward.
- Maximum hourly target horizon: 72 hours; every split and validation fold uses at least a 72-hour gap.

## Inference contract

The tested implementation is `src/inference_hourly.py`:

```python
from pathlib import Path
import pandas as pd
from src.inference_hourly import predict_latest

hourly = pd.read_csv(
    Path("data/processed/karak_aqi_open_meteo_hourly_features.csv"),
    parse_dates=["time"],
)
forecast = predict_latest(hourly)
```

It loads the manifest and selected Ridge artifact, verifies the feature-column schema, rejects missing/non-finite latest inputs, and returns 30 rows with `forecast_origin`, `output`, `kind`, `start_time`, `end_time`, and `value`.

The inference module is a local library contract, not yet a deployed API. A dashboard or API must call this module and must not duplicate feature engineering.

## Reproducibility and validation outputs

- Python: 3.11.9; all project declarations require Python 3.11.
- Hourly manifest: `models/aqi_forecast_hourly_models.json`.
- Hourly holdout metrics: `data/processed/hourly_model_comparison.csv`.
- Hourly rolling-origin metrics: `data/processed/hourly_rolling_origin_comparison.csv`.
- Training frame: `data/processed/training_frame_hourly.csv`.
- Source SHA-256, feature schema, package versions, split protocol, and selected model are recorded in the hourly manifest.
- Generated data and model binaries are ignored by Git until a registry or Git LFS policy is approved; the JSON manifest remains the handoff record.

## Pre-infrastructure checklist

| Check | Status | Evidence |
|---|---|---|
| EPA target and category calculation | Pass | `src/aqi.py`, `tests/test_aqi.py` |
| Exact 30-output target grid | Pass | `src/train_hourly.py`, `tests/test_hourly_training.py` |
| Historical-only feature contract | Pass | feature builder and inference schema check |
| Duplicate/cadence/missing-input protection | Pass | training validation, inference validation, tests |
| 72-hour purge-aware holdout | Pass | `chronological_split` |
| Purged rolling-origin evaluation | Pass | `hourly_rolling_origin_comparison.csv` |
| Persistence and seasonal baselines | Pass | hourly comparison outputs |
| RMSE, MSE, MAE, R², category, high-AQI diagnostics | Pass | comparison CSVs and manifest |
| Selected persisted model | Pass | Ridge for all hourly output groups |
| Local inference smoke test | Pass | `src/inference_hourly.py` |
| Automated tests and CI definition | Pass locally | 21 tests; GitHub Actions workflows |
| Station-ground-truth accuracy | Not available | Open-Meteo modeled source limitation |

Only after this checklist is accepted should the project add a feature store, model registry, dashboard, or deployment layer.

## Infrastructure phase — 2026-08-15

The checklist above is accepted and the infrastructure phase is implemented on
top of it **without changing any modeling contract**:

- **Feature store** (`src/feature_store.py`): DuckDB store backfilled from the
  same validated feature builders; `meta` records schema version + feature
  columns so drift is detectable (`python -m src.feature_store stats`).
- **Model registry** (`src/model_registry.py`): MLflow file-backed registry;
  registers `aqi-hourly-ridge` + `aqi-daily-h1..h3` with a `champion` alias.
- **Automation**: `feature_pipeline.yml` (hourly) and `training_pipeline.yml`
  (daily) in `.github/workflows/`; both keyless.
- **Dashboard + API**: `app/dashboard.py` (Streamlit, 72h forecast, alerts,
  SHAP, comparison, EDA) and `app/api.py` (FastAPI `/forecast`, `/explain`, ...).

Full decision records and the assignment mapping are in
`Docs/mlops_architecture.md`. The station-ground-truth limitation is unchanged.
