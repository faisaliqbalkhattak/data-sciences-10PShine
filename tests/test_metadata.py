import json
from pathlib import Path


def test_model_manifest_describes_current_trained_target():
    project_root = Path(__file__).parents[1]
    manifest = json.loads(
        (project_root / "models" / "aqi_forecast_models.json").read_text()
    )
    meta = manifest["_meta"]
    assert meta["target"] == "aqi_us_epa"
    assert meta["horizons_days"] == [1, 2, 3]
    assert set(meta["best_model_by_horizon"]) == {"1", "2", "3"}
    for horizon in meta["horizons_days"]:
        artifact_path = project_root / f"models/lstm_h{horizon}.keras"
        assert manifest[f"lstm_h{horizon}"]["path"] == f"models/lstm_h{horizon}.keras"
        # Binary artifacts are generated locally and intentionally ignored until
        # a registry/LFS policy is approved; validate them when present.
        if artifact_path.exists():
            assert artifact_path.stat().st_size > 0


def test_hourly_manifest_describes_the_release_contract():
    project_root = Path(__file__).parents[1]
    manifest_path = project_root / "models" / "aqi_forecast_hourly_models.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    meta = manifest["_meta"]
    assert meta["target"] == "aqi_hourly_rolling"
    assert meta["output_count"] == 30
    assert meta["selected_model_by_group"] == {
        "hourly_points": "ridge",
        "six_hour_means": "ridge",
        "twelve_hour_means": "ridge",
    }
    assert Path(project_root / meta["rolling_origin_metrics_path"]).exists()
