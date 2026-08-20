"""Tests for the forecast export pipeline and IQAir scraping."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestExportForecast:
    """Tests for src/export_forecast.py."""

    def test_export_forecast_file_is_valid_json(self, tmp_path: Path) -> None:
        """The exported file must be valid JSON with the required keys."""
        from src.export_forecast import FORECAST_PATH, export_forecast

        # Mock the live data loading to avoid network calls
        from src import config as _cfg

        mock_hourly = pd.DataFrame(
            {col: [10.0] * 168 for col in _cfg.AIR_QUALITY_HOURLY_VARS + _cfg.WEATHER_HOURLY_VARS},
            index=pd.date_range("2026-08-13", periods=168, freq="h"),
        )
        mock_hourly.index.name = "time"

        with (
            patch("app.live_data.load_latest_hourly", return_value=mock_hourly),
            patch("app.live_data.iqair_forecast_aqi", return_value=pd.Series(dtype=float)),
        ):
            # Import and monkeypatch the internal function
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


class TestIQAirScraping:
    """Tests for the IQAir HTML scraper in app/live_data.py."""

    def test_scraper_regex_matches_current_html(self) -> None:
        """The regex must match the current IQAir HTML structure."""
        import re

        # Simplified version of the actual IQAir HTML (as of Aug 2026)
        sample_html = """
        <td><div class="flex"><div class="flex flex-col items-center gap-2 border-r border-dashed border-gray-200 px-2.5 text-sm text-gray-900">
        <p class="max-w-12 truncate">Now</p>
        <div class="text-black-50 aqi-bg-orange lgsm:h-[26px] lgsm:w-[50px] h-[22px] w-11 rounded-sm border border-solid border-transparent">
        <p class="flex h-full w-full flex-col items-center justify-center text-sm font-medium">143</p>
        </div>
        </div></div></td>
        <td><div class="flex"><div class="flex flex-col items-center gap-2 border-r border-dashed border-gray-200 px-2.5 text-sm text-gray-900">
        <p class="max-w-12 truncate">17:00</p>
        <div class="text-black-50 aqi-bg-orange lgsm:h-[26px] lgsm:w-[50px] h-[22px] w-11 rounded-sm border border-solid border-transparent">
        <p class="flex h-full w-full flex-col items-center justify-center text-sm font-medium">142</p>
        </div>
        </div></div></td>
        """

        values = []
        for m in re.finditer(r"aqi-bg-[a-z-]+.*?<p[^>]*>\s*(\d+)\s*</p>", sample_html, re.S):
            values.append(float(m.group(1)))

        assert values == [143.0, 142.0], f"Expected [143, 142], got {values}"

    def test_iqair_forecast_returns_series(self) -> None:
        """iqair_forecast_aqi should return a pd.Series (or raise on network failure)."""
        from app.live_data import iqair_forecast_aqi

        try:
            series = iqair_forecast_aqi()
            assert isinstance(series, pd.Series)
            if len(series) > 0:
                assert series.index.dtype == "datetime64[ns]"
                assert series.name == "aqi"
        except RuntimeError:
            # Network failure is acceptable in CI (IQAir rate-limits)
            pytest.skip("IQAir unavailable (rate-limited or no network)")

    def test_forecast_json_has_iqair_refs(self) -> None:
        """The static forecast JSON should contain IQAir reference data."""
        from src import config

        forecast_path = config.PROJECT_ROOT / "data" / "static_forecast.json"
        if not forecast_path.exists():
            pytest.skip("No static forecast file")
        data = json.loads(forecast_path.read_text())
        iqair = data.get("iqair_forecast", [])
        # IQAir may be empty if rate-limited during export, but the key must exist
        assert "iqair_forecast" in data, "Missing 'iqair_forecast' key in forecast JSON"
        assert isinstance(iqair, list), "iqair_forecast must be a list"
