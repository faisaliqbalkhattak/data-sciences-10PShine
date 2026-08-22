"""Tests for the forecast export pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestExportForecast:
    """Tests for src/export_forecast.py."""

    def test_export_forecast_file_is_valid_json(self, tmp_path: Path) -> None:
        """The exported file must be valid JSON with the required keys."""
        from src import config as _cfg

        # Skip if no trained model exists — models live in karAQI-data, not here
        manifest_path = _cfg.PROJECT_ROOT / "models" / "aqi_forecast_hourly_models.json"
        if not manifest_path.exists():
            pytest.skip("No trained model manifest (run training pipeline first)")

        mock_hourly = pd.DataFrame(
            {col: [10.0] * 168 for col in _cfg.AIR_QUALITY_HOURLY_VARS + _cfg.WEATHER_HOURLY_VARS},
            index=pd.date_range("2026-08-13", periods=168, freq="h"),
        )
        mock_hourly.index.name = "time"

        try:
            with patch("app.live_data.load_latest_hourly", return_value=mock_hourly):
                import src.export_forecast as ef

                original_path = ef.FORECAST_PATH
                ef.FORECAST_PATH = tmp_path / "test_forecast.json"
                try:
                    path = ef.export_forecast("live")
                    assert path.exists()
                    data = json.loads(path.read_text())
                    assert "generated_at" in data
                    assert "origin" in data
                    assert "outputs" in data
                    assert len(data["outputs"]) == 30
                    assert "current_aqi" in data
                finally:
                    ef.FORECAST_PATH = original_path
        except FileNotFoundError:
            pytest.skip("MLflow registry not available (run training pipeline first)")

    def test_forecast_json_has_required_output_fields(self, tmp_path: Path) -> None:
        """Each output in the JSON must have output, kind, start_time, end_time, value, category."""
        from src import config

        forecast_path = config.PROJECT_ROOT / "data" / "static_forecast.json"
        if not forecast_path.exists():
            pytest.skip("No static forecast file (run export first)")
        data = json.loads(forecast_path.read_text())
        for item in data["outputs"]:
            assert "output" in item, f"Missing 'output' in {item}"
            assert "kind" in item, f"Missing 'kind' in {item}"
            assert "start_time" in item, f"Missing 'start_time' in {item}"
            assert "end_time" in item, f"Missing 'end_time' in {item}"
            assert "value" in item, f"Missing 'value' in {item}"
            assert "category" in item, f"Missing 'category' in {item}"
            assert isinstance(item["value"], (int, float)), f"Value not numeric: {item['value']}"

    def test_forecast_json_has_ref_data(self) -> None:
        """The static forecast JSON should contain Open-Meteo reference data."""
        from src import config

        forecast_path = config.PROJECT_ROOT / "data" / "static_forecast.json"
        if not forecast_path.exists():
            pytest.skip("No static forecast file")
        data = json.loads(forecast_path.read_text())
        ref = data.get("ref_forecast", [])
        # Reference data may be empty if API failed, but the key must exist
        assert "ref_forecast" in data, "Missing 'ref_forecast' key in forecast JSON"
        assert isinstance(ref, list), "ref_forecast must be a list"
