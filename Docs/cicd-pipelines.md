# CI/CD Pipelines

GitHub Actions pipeline architecture, scheduling, data flow, and the troubleshooting history.

---

## Pipeline Architecture

```
Open-Meteo (keyless)
      │
      ▼
training_pipeline.yml (daily 00:00 UTC)
  → Incremental fetch → Train Ridge + LSTM → Register in MLflow
  → Export eval JSON → Push model files (.joblib, .keras)
      │
      ▼
karAQI-data/models/*.joblib, *.keras  (trained model files)
karAQI-data/data/model_eval.json       (evaluation metrics)
      │
      ▼
forecast_pipeline.yml (hourly :04)
  → Downloads model from karAQI-data → Runs inference
  → Fetches Open-Meteo AQ ref → Exports JSON
      │
      ▼
karAQI-data/data/static_forecast.json
      │
      ▼
Dashboard reads via raw GitHub URLs

feature_pipeline.yml (hourly :01)
  → Incremental fetch (~1 row) → Build features → Run tests
  → Updates DuckDB feature store (used by training pipeline)
```

---

## Pipeline Schedules

| Pipeline | Cron (UTC) | Pakistan Time | Purpose | Duration |
|---|---|---|---|---|
| Feature | `1 * * * *` | XX:01 +5h | Incremental fetch, feature build, tests | ~1 min |
| Forecast | `4 * * * *` | XX:04 +5h | Inference + JSON export | ~45 sec |
| Training | `0 0 * * *` | 05:00 +5h | Incremental fetch, train models, register, export, push to karAQI-data | ~8 min |

### Why These Specific Minutes?

- **Feature at :01** — First to run each hour. Fetches latest data before forecast needs it.
- **Forecast at :04** — Waits for feature pipeline to complete (typically takes ~1 min).
- **Training at 00:00 UTC (05:00 PKT)** — Runs once daily, takes ~8 min (TensorFlow LSTM training). Scheduled before hourly pipelines to avoid resource contention.

### GitHub Actions Timing Caveat

Cron triggers are best-effort, not precise. Workflows are frequently delayed 5–30 minutes (sometimes longer) due to:

- Repository activity level (less active repos get lower priority)
- Platform load (shared runner infrastructure)
- Workflow concurrency (queues behind previous runs)

See [github/community#156282](https://github.com/orgs/community/discussions/156282).

**Impact on dashboard:** If the hero AQI shows a previous hour's value, the CI pipeline was delayed. The data is correct — it reflects the most recent successful run.

---

## Data Push to karAQI-data

Each pipeline pushes its output to the separate data repo (`karAQI-data`) using a fine-grained PAT:

```bash
# Clone data repo
git clone https://oauth2:${DATA_REPO_TOKEN}@github.com/faisaliqbalkhattak/karAQI-data.git /tmp/karAQI-data

# Copy output
cp data/static_forecast.json /tmp/karAQI-data/data/

# Commit and push
cd /tmp/karAQI-data
git add data/static_forecast.json
git diff --cached --quiet || git commit -m "ci: refresh forecast JSON [skip ci]"
git push
```

**Token setup:** Fine-grained PAT with Contents: Read/Write on karAQI-data only. Stored as `DATA_REPO_TOKEN` secret in karAQI repo settings.

---

## Dashboard Data Flow

The dashboard never runs inference. It reads pre-computed JSON from karAQI-data:

```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def _load_forecast():
    url = "https://raw.githubusercontent.com/faisaliqbalkhattak/karAQI-data/main/data/static_forecast.json"
    return requests.get(url, timeout=10).json()
```

This gives every visitor near-instant page load regardless of model complexity.

---

## Troubleshooting History

### Issue 1: CRLF Line Endings Broke CI

**Symptom:** `No module named pytest` in GitHub Actions.

**Root cause:** `requirements.txt` had CRLF line endings (`\r\n`) from Windows editing. On Linux runners, pip silently skipped some packages.

**Fix:** `.gitattributes` with `* text=auto eol=lf` + converted all text files to LF.

### Issue 2: Streamlit Cloud "Oh No" Crash

**Symptom:** App showed "Oh no. Error running app."

**Root cause:** Same CRLF issue. Streamlit Cloud runs on Linux.

**Fix:** Same `.gitattributes` fix.

### Issue 3: pyarrow Build Failure on Streamlit Cloud

**Symptom:** `error: command 'cmake' failed` during dependency installation.

**Root cause:** `requirements.txt` included mlflow, which depends on pyarrow. Streamlit Cloud tried to compile pyarrow from source (no pre-built wheel) and failed (no cmake).

**Fix:** Created lean `app/requirements.txt` with only dashboard dependencies. Excluded mlflow, fastapi, pytest, etc. Streamlit Cloud picks it up by proximity to `app/dashboard.py`.

### Issue 4: Python 3.14 on Streamlit Cloud

**Symptom:** numpy, pandas, pyarrow fail to build.

**Root cause:** Streamlit Cloud defaults to Python 3.14, ignores `.python-version` file.

**Fix:** Set Python version to 3.11 in Streamlit Cloud Advanced Settings UI.

### Issue 5: Push Failures to karAQI-data

**Symptom:** `fatal: Authentication failed` when pushing to karAQI-data.

**Root cause:** `DATA_REPO_TOKEN` secret not set or token lacked permissions.

**Fix:** Created fine-grained PAT with Contents: Read/Write on karAQI-data only. Added as secret in karAQI settings.

### Issue 6: Silent Pipeline Success on Push Failure

**Symptom:** Dashboard showed stale data (4+ hours old) but all workflows showed green.

**Root cause:** `exit 0` on clone failure — pipeline silently skipped the push.

**Fix:** Changed to `exit 1` with clear error message. Workflows now fail visibly when push fails.

### Issue 7: Training Pipeline Push Race Condition

**Symptom:** `error: cannot pull with rebase: You have unstaged changes`

**Root cause:** The pipeline generated `model_eval.json` (unstaged) before trying to pull-rebase the data repo.

**Fix:** Added `git stash` before pull-rebase to handle any local changes.

### Issue 8: IQAir Rate Limiting

**Symptom:** `IQAir 429 rate-limited, attempt 1... All attempts failed`

**Root cause:** Scraping IQAir's HTML from GitHub Actions runners (cloud IPs). IQAir aggressively rate-limits anonymous scraping.

**Fix:** Replaced IQAir entirely with Open-Meteo's free AQ forecast API. Deleted `fetch_iqair.py` and `iqair_pipeline.yml`.

---

## Lessons Learned

1. **Always add `.gitattributes` with `eol=lf`** to Python projects that run CI on Linux
2. **Never pin exact Python patch versions** in CI — use major.minor
3. **Separate fast CI checks from slow training** — feature pipeline <1 min, training ~8 min
4. **Use skip-gracefully patterns in tests** — check `if not path.exists(): return`
5. **Separate dashboard and CI dependencies** — lean `app/requirements.txt` for Streamlit Cloud
6. **Pipeline failures should be loud** — `exit 1`, not `exit 0`
7. **Cron triggers are best-effort** — design for delayed execution
