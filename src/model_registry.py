"""MLflow model registry for the Karak AQI pipeline.

Decision record
---------------
The registry uses **MLflow with a local file-backed store** (``models/mlruns/``):
no tracking server, no API key, no paid tier. This matches the assignment's
"100% serverless / free student stack" requirement: the training pipeline runs
on demand (locally or on GitHub Actions), logs experiments and registers model
versions, and the dashboard loads the registered "champion" version. The
existing JSON manifests remain the source of truth for the evaluation contract;
this module registers their artifacts into MLflow so the registry is a
first-class, queryable component rather than a renaming of the manifest.

Registered models
-----------------
``aqi-hourly-ridge``   the selected multi-output Ridge (30-output hourly).
``aqi-daily-h1``       best daily model for the +1 day horizon.
``aqi-daily-h2``       best daily model for the +2 day horizon.
``aqi-daily-h3``       best daily model for the +3 day horizon.

Every registration marks the new version with the ``champion`` alias, so
``load_hourly_model()`` and the dashboard always load the latest accepted
model.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import pandas as pd

from . import config

HOURLY_MODEL_NAME = "aqi-hourly"
DAILY_MODEL_NAMES = {1: "aqi-daily-h1", 2: "aqi-daily-h2", 3: "aqi-daily-h3"}
CHAMPION_ALIAS = "champion"
HOURLY_MANIFEST = config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_models.json"
DAILY_MANIFEST = config.PROJECT_ROOT / "models" / "aqi_forecast_models.json"
RIDGE_ARTIFACT = config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_ridge.joblib"

HOURLY_MANIFEST = config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_models.json"
DAILY_MANIFEST = config.PROJECT_ROOT / "models" / "aqi_forecast_models.json"
RIDGE_ARTIFACT = config.PROJECT_ROOT / "models" / "aqi_forecast_hourly_ridge.joblib"


def _tracking_uri(store_dir: Optional[Path] = None) -> str:
    directory = Path(store_dir or config.MLRUNS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve().as_uri()


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _set_champion(client, model_name: str, version: str) -> None:
    """Mark ``version`` as the champion (production) alias."""
    versions = client.search_model_versions(f"name='{model_name}'")
    for existing in versions:
        if CHAMPION_ALIAS in (existing.aliases or []):
            client.delete_registered_model_alias(model_name, CHAMPION_ALIAS)
            break
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)


def _get_current_champion_metrics(
    model_name: str,
    store_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Get the current champion's primary metric (RMSE for hourly, grouped)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    try:
        mlflow.set_tracking_uri(_tracking_uri(store_dir))
        client = MlflowClient(_tracking_uri(store_dir))
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if CHAMPION_ALIAS in (v.aliases or []):
                run = client.get_run(v.run_id)
                return run.data.metrics
        return None
    except Exception:
        return None


def _should_promote(
    new_metrics: dict,
    current_champion_metrics: Optional[dict],
    metric_key: str = "hourly_points_rmse",
) -> tuple[bool, str]:
    """Check if new model should be promoted over current champion.

    Returns (should_promote, reason).
    """
    if current_champion_metrics is None:
        return True, "No current champion; promoting new model."

    current_rmse = current_champion_metrics.get(metric_key)
    new_rmse = new_metrics.get(metric_key)

    if current_rmse is None or new_rmse is None:
        return True, "Missing metric in champion or new model; promoting."

    if new_rmse < current_rmse:
        improvement = (current_rmse - new_rmse) / current_rmse * 100
        return True, f"New model RMSE {new_rmse:.2f} < champion {current_rmse:.2f} ({improvement:.1f}% better)"
    else:
        degradation = (new_rmse - current_rmse) / current_rmse * 100
        return False, f"New model RMSE {new_rmse:.2f} >= champion {current_rmse:.2f} ({degradation:.1f}% worse); keeping champion."


def register_hourly(
    manifest_path: Optional[Path] = None,
    store_dir: Optional[Path] = None,
) -> dict:
    """Register the best hourly model and its metrics as one MLflow run.

    Compares against the current champion and only promotes if the new model
    is better on the primary metric (hourly_points RMSE).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    manifest = _load_manifest(Path(manifest_path) if manifest_path else HOURLY_MANIFEST)
    meta = manifest.get("_meta", {})
    selected_by_group = meta.get("selected_model_by_group", {})

    # Find the best model from the selected groups
    # Use the model that was selected for hourly_points (primary group)
    best_model_name = selected_by_group.get("hourly_points", "ridge")
    best_model = manifest.get(best_model_name)
    if best_model is None or not best_model.get("path"):
        raise ValueError(f"Hourly manifest has no persisted artifact for {best_model_name}.")

    artifact = _resolve_artifact(best_model["path"], manifest_path)
    if not artifact.exists():
        raise FileNotFoundError(f"Hourly {best_model_name} artifact not found: {artifact}")

    # Get new model metrics
    new_metrics = {}
    for group, metrics in best_model.get("metrics_by_group", {}).items():
        for name, value in metrics.items():
            new_metrics[f"{group}_{name}"] = float(value)

    # Check against current champion
    current_champion_metrics = _get_current_champion_metrics(HOURLY_MODEL_NAME, store_dir)
    promote, reason = _should_promote(new_metrics, current_champion_metrics)
    logger.info("Champion comparison: %s", reason)

    if not promote:
        logger.info("Keeping current champion. Skipping registration.")
        return {
            "model_name": HOURLY_MODEL_NAME,
            "version": "unchanged",
            "alias": CHAMPION_ALIAS,
            "reason": reason,
        }

    mlflow.set_tracking_uri(_tracking_uri(store_dir))
    mlflow.set_experiment("karak-aqi-hourly")
    with mlflow.start_run(run_name=f"hourly-{best_model_name}-{date.today().isoformat()}") as run:
        mlflow.log_params(
            {
                "selected_model": best_model_name,
                "selected_by_group": json.dumps(selected_by_group),
                "output_count": meta.get("output_count"),
                "n_train_rows": meta.get("n_train_rows"),
                "n_test_rows": meta.get("n_test_rows"),
                "source_sha256": meta.get("source_sha256"),
            }
        )
        for name, value in new_metrics.items():
            mlflow.log_metric(name, value)
        feature_columns = meta.get("feature_columns", []) or []
        input_example = (
            pd.DataFrame(
                [[0.0] * len(feature_columns)], columns=feature_columns
            )
            if feature_columns
            else None
        )
        mlflow.sklearn.log_model(
            _load_pipeline(artifact), artifact_path="model", input_example=input_example
        )
        mlflow.log_artifact(Path(manifest_path) if manifest_path else HOURLY_MANIFEST)
        registered = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", HOURLY_MODEL_NAME
        )

    client = MlflowClient(_tracking_uri(store_dir))
    _set_champion(client, HOURLY_MODEL_NAME, registered.version)
    return {
        "model_name": HOURLY_MODEL_NAME,
        "version": registered.version,
        "alias": CHAMPION_ALIAS,
        "promoted_model": best_model_name,
        "reason": reason,
        "run_id": run.info.run_id,
    }


def register_daily(
    manifest_path: Optional[Path] = None,
    store_dir: Optional[Path] = None,
) -> dict:
    """Register the best daily model per horizon and log all comparison metrics."""
    import mlflow
    from mlflow.tracking import MlflowClient

    manifest = _load_manifest(Path(manifest_path) if manifest_path else DAILY_MANIFEST)
    meta = manifest.get("_meta", {})
    best_by_horizon = meta.get("best_model_by_horizon", {})

    mlflow.set_tracking_uri(_tracking_uri(store_dir))
    mlflow.set_experiment("karak-aqi-daily")
    registered = {}
    with mlflow.start_run(run_name=f"daily-{date.today().isoformat()}") as run:
        mlflow.log_params(
            {
                "horizons_days": json.dumps(meta.get("horizons_days", [])),
                "best_by_horizon": json.dumps(best_by_horizon),
            }
        )
        for horizon, model_name in DAILY_MODEL_NAMES.items():
            key = f"{best_by_horizon.get(str(horizon), 'ridge')}_h{horizon}"
            entry = manifest.get(key)
            if not entry or not entry.get("path"):
                logger.warning("No persisted artifact for %s; skipping registration.", key)
                continue
            artifact = _resolve_artifact(entry["path"], manifest_path)
            if not artifact.exists():
                logger.warning("Artifact %s missing; skipping registration.", artifact)
                continue
            for metric_name, metric_value in entry.get("metrics", {}).items():
                mlflow.log_metric(f"h{horizon}_{metric_name}", float(metric_value))
            feature_columns = meta.get("features", []) or []
            input_example = (
                pd.DataFrame([[0.0] * len(feature_columns)], columns=feature_columns)
                if feature_columns
                else None
            )
            mlflow.sklearn.log_model(
                _load_pipeline(artifact),
                artifact_path=f"model_h{horizon}",
                input_example=input_example,
            )
            client = MlflowClient(_tracking_uri(store_dir))
            model_version = mlflow.register_model(
                f"runs:/{run.info.run_id}/model_h{horizon}", model_name
            )
            _set_champion(client, model_name, model_version.version)
            registered[model_name] = model_version.version
        mlflow.log_artifact(Path(manifest_path) if manifest_path else DAILY_MANIFEST)
    return {"registered": registered, "run_id": run.info.run_id}


def _resolve_artifact(artifact_path: str, manifest_path: Optional[Path]) -> Path:
    """Resolve a manifest artifact path relative to project root or the manifest."""
    artifact = Path(artifact_path)
    if artifact.is_absolute():
        return artifact
    candidate = config.PROJECT_ROOT / artifact
    if not candidate.exists() and manifest_path is not None:
        candidate = Path(manifest_path).parent / artifact
    return candidate


def _load_pipeline(path: Path):
    """Load a persisted sklearn estimator (joblib or keras directory)."""
    import joblib

    if path.is_dir() or path.suffix == ".keras":
        # Keras artifacts are not registered by this module; the selected
        # models are sklearn pipelines. Raise a clear error if asked to load one.
        raise ValueError(
            f"{path} is a Keras artifact; the registry registers sklearn estimators only."
        )
    return joblib.load(path)


def load_hourly_model(store_dir: Optional[Path] = None):
    """Load the champion hourly model, falling back to the local manifest artifact.

    Prefers the MLflow-registered version so the forecast pipeline uses whatever
    the training pipeline accepted last. Falls back to the local joblib so the
    pipeline and tests work even before the first registry run.
    """
    import mlflow

    try:
        mlflow.set_tracking_uri(_tracking_uri(store_dir))
        return mlflow.pyfunc.load_model(f"models:/{HOURLY_MODEL_NAME}@{CHAMPION_ALIAS}")
    except Exception as exc:  # noqa: BLE001 - registry may be empty on first run
        logger.info("MLflow champion not found (%s); falling back to local artifact.", exc)
        # Try to find any local model artifact
        manifest = _load_manifest(HOURLY_MANIFEST)
        meta = manifest.get("_meta", {})
        selected = meta.get("selected_model_by_group", {})
        best_model = selected.get("hourly_points", "ridge")
        entry = manifest.get(best_model, {})
        artifact_path = entry.get("path")
        if artifact_path:
            artifact = _resolve_artifact(artifact_path, None)
            if artifact.exists():
                return _load_pipeline(artifact)
        # Final fallback: try Ridge
        if RIDGE_ARTIFACT.exists():
            return _load_pipeline(RIDGE_ARTIFACT)
        raise FileNotFoundError(
            "No registered hourly model and no local artifact; run "
            "`python -m src.train_hourly` first."
        ) from exc


def list_registered(store_dir: Optional[Path] = None) -> list[dict]:
    """List every registered model, its latest version, and its aliases."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(_tracking_uri(store_dir))
    rows = []
    for model in client.search_registered_models():
        latest = model.latest_versions
        rows.append(
            {
                "name": model.name,
                "latest_versions": [
                    {"version": v.version, "alias": v.aliases, "stage": v.current_stage}
                    for v in latest
                ],
            }
        )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.model_registry",
        description="Karak AQI MLflow registry (local file-backed).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register-hourly", help="Register the hourly Ridge model + metrics.").set_defaults(
        func=lambda args: print(json.dumps(register_hourly(), indent=2))
    )
    sub.add_parser("register-daily", help="Register the best daily model per horizon.").set_defaults(
        func=lambda args: print(json.dumps(register_daily(), indent=2))
    )
    sub.add_parser("list", help="List registered models.").set_defaults(
        func=lambda args: print(json.dumps(list_registered(), indent=2))
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
