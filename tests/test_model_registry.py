import json

import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

mlflow = pytest.importorskip("mlflow", reason="MLflow not installed (training pipeline only)")

from src.model_registry import (
    HOURLY_MODEL_NAME,
    list_registered,
    load_hourly_model,
    register_hourly,
)


def _fake_manifest(artifact_path):
    return {
        "ridge": {
            "path": artifact_path,
            "params": {"alpha": 10.0, "candidate_alphas": [1.0, 10.0]},
            "metrics_by_group": {
                "hourly_points": {"rmse": 11.0, "mae": 8.5, "r2": 0.3},
                "six_hour_means": {"rmse": 18.0, "mae": 14.0, "r2": -0.02},
                "twelve_hour_means": {"rmse": 19.0, "mae": 15.0, "r2": -0.1},
            },
        },
        "_meta": {
            "target": "aqi_hourly_rolling",
            "output_count": 30,
            "selected_model_by_group": {
                "hourly_points": "ridge",
                "six_hour_means": "ridge",
                "twelve_hour_means": "ridge",
            },
            "n_train_rows": 1000,
            "n_test_rows": 250,
            "source_sha256": "abc123",
        },
    }


def _fake_ridge(tmp_path):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=["f1", "f2", "f3", "f4"])
    y = pd.DataFrame(rng.normal(size=(120, 30)))
    pipeline = Ridge(alpha=10.0).fit(X, y)
    artifact = tmp_path / "ridge.joblib"
    joblib.dump(pipeline, artifact)
    return artifact, X.columns.tolist()


def test_register_and_load_hourly_champion(tmp_path):
    artifact, feature_names = _fake_ridge(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_fake_manifest(artifact.name)))

    result = register_hourly(manifest_path=manifest_path, store_dir=tmp_path / "mlruns")

    assert result["model_name"] == HOURLY_MODEL_NAME
    assert int(result["version"]) >= 1
    assert result["alias"] == "champion"

    registered = list_registered(store_dir=tmp_path / "mlruns")
    assert any(model["name"] == HOURLY_MODEL_NAME for model in registered)

    loaded = load_hourly_model(store_dir=tmp_path / "mlruns")
    row = pd.DataFrame([[0.1, 0.2, -0.1, 0.4]], columns=feature_names)
    prediction = loaded.predict(row)
    assert len(np.asarray(prediction).reshape(-1)) == 30


def test_load_falls_back_to_local_artifact_when_registry_empty(tmp_path):
    # No MLflow registration: load_hourly_model must still work if the local
    # joblib exists. Point it at the real local artifact only when present.
    from src import config

    if not config.PROJECT_ROOT.joinpath("models/aqi_forecast_hourly_ridge.joblib").exists():
        pytest.skip("No local hourly Ridge artifact present.")
    loaded = load_hourly_model(store_dir=tmp_path / "empty_mlruns")
    assert hasattr(loaded, "predict")
