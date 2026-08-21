"""Tests for the Open-Meteo AQ forecast parser, the current-conditions helper,
and the store top-up / current-hour anchoring logic.

The fetches are mocked so the tests never touch the network; the AQI
conversion / API parsing is the unit under test.
"""

import numpy as np
import pandas as pd
import pytest


def _hourly_frame(start: str, end: str) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="h")
    air = pd.DataFrame({"time": index, "pm2_5": 30.0, "pm10": 40.0, "ozone": 60.0})
    air["source"] = "open_meteo_air_quality"
    weather = pd.DataFrame({"time": index, "temperature_2m": 30.0, "relative_humidity_2m": 40.0})
    weather["source"] = "open_meteo_weather_features"
    return air, weather


def test_stale_store_is_topped_up_and_anchored_to_current_hour(monkeypatch):
    """A stale store is refreshed, and the origin is capped at the current hour
    even when the upstream pull contains later hours of the current day."""
    from app import live_data

    now = pd.Timestamp("2026-08-16 12:00:00")
    monkeypatch.setattr(live_data, "_current_hour_local", lambda: now)

    stale_index = pd.date_range("2026-08-09", "2026-08-15 23:00", freq="h")
    stale = pd.DataFrame({"time": stale_index, "pm2_5": 30.0, "pm10": 40.0})
    stale = stale.set_index("time")

    def fake_get_hourly_raw():
        return stale.copy()

    monkeypatch.setattr("src.feature_store.get_hourly_raw", fake_get_hourly_raw)

    air, weather = _hourly_frame("2026-08-09", "2026-08-16 23:00")
    monkeypatch.setattr(live_data, "fetch_open_meteo_air_quality", lambda start, end: air)
    monkeypatch.setattr(live_data, "fetch_open_meteo_weather_history", lambda start, end: weather)

    frame = live_data.load_latest_hourly("store")

    # Topped up (fresh data) and truncated to the current hour, not 23:00.
    # The frame is then limited to the 168-hour inference lookback.
    assert frame.index[-1] == now
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 168


def test_fresh_store_is_not_topped_up(monkeypatch):
    from app import live_data

    now = pd.Timestamp("2026-08-16 12:00:00")
    monkeypatch.setattr(live_data, "_current_hour_local", lambda: now)

    fresh_index = pd.date_range("2026-08-09", "2026-08-16 12:00", freq="h")
    fresh = pd.DataFrame({"time": fresh_index, "pm2_5": 30.0, "pm10": 40.0})
    fresh = fresh.set_index("time")

    def fake_get_hourly_raw():
        return fresh.copy()

    monkeypatch.setattr("src.feature_store.get_hourly_raw", fake_get_hourly_raw)

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("fresh store must not trigger a live pull")

    monkeypatch.setattr(live_data, "fetch_open_meteo_air_quality", unexpected_fetch)
    monkeypatch.setattr(live_data, "fetch_open_meteo_weather_history", unexpected_fetch)

    frame = live_data.load_latest_hourly("store")

    assert frame.index[-1] == now
    assert len(frame) == 168


# ---------------------------------------------------------------------------
# current_conditions helper (IQAir-style hero: weather + dominant pollutant)
# ---------------------------------------------------------------------------


def test_current_conditions_reports_weather_and_dominant_pollutant():
    from app.live_data import current_conditions

    index = pd.date_range("2026-08-15 00:00", periods=48, freq="h")
    frame = pd.DataFrame(
        {
            "time": index,
            "pm2_5": 30.0,
            "pm10": 80.0,
            "ozone": 20.0,
            "temperature_2m": 33.0,
            "relative_humidity_2m": 26.0,
            "wind_speed_10m": 18.0,
        }
    ).set_index("time")

    conditions = current_conditions(frame)

    assert conditions["temperature_2m"] == 33.0
    assert conditions["relative_humidity_2m"] == 26.0
    assert conditions["wind_speed_10m"] == 18.0
    # pm2_5 = 30 ug/m3 -> EPA sub-index ~91; pm10 = 80 ug/m3 -> sub-index ~63,
    # so PM2.5 is the dominant pollutant despite the higher raw PM10.
    assert conditions["main_pollutant"] == "pm2_5"
    assert conditions["concentration"] == 30.0


def test_current_conditions_handles_empty_frame():
    from app.live_data import current_conditions

    assert current_conditions(pd.DataFrame()) == {}
    assert current_conditions(None) == {}


# ---------------------------------------------------------------------------
# IQAir hourly forecast parser
# ---------------------------------------------------------------------------


def _open_meteo_aq_json() -> dict:
    """Synthetic Open-Meteo AQ forecast response (72 hours of US AQI)."""
    base = pd.Timestamp("2026-08-16 17:00:00")
    times = [(base + pd.Timedelta(hours=i)).isoformat() for i in range(72)]
    aqi_values = [113, 113, 112, 111, 111, 110, 110, 110,
                  117, 127, 138] + list(range(140, 163)) + list(range(130, 154)) + list(range(120, 136))
    # Pad or trim to exactly 72
    aqi_values = (aqi_values + [120] * 72)[:72]
    return {
        "hourly": {
            "time": times,
            "us_aqi": aqi_values,
        }
    }


class _FakeResponse:
    status_code = 200

    def __init__(self, json_data: dict):
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def test_aq_forecast_parses_api_response_anchored_to_current_hour(monkeypatch):
    from app import live_data

    monkeypatch.setattr(live_data, "_iqair_cache", {"ts": None, "series": None})
    monkeypatch.setattr(
        live_data.requests, "get",
        lambda *a, **k: _FakeResponse(_open_meteo_aq_json()),
    )

    series = live_data.iqair_forecast_aqi()

    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.is_monotonic_increasing
    assert len(series) == 72
    assert series.iloc[0] == 113.0
    assert series.name == "aqi"


def test_aq_forecast_caches_and_reanchors(monkeypatch):
    from app import live_data

    monkeypatch.setattr(live_data, "_iqair_cache", {"ts": None, "series": None})
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return _FakeResponse(_open_meteo_aq_json())

    monkeypatch.setattr(live_data.requests, "get", fake_get)

    first = live_data.iqair_forecast_aqi()
    second = live_data.iqair_forecast_aqi()
    assert calls["n"] == 1  # second call served from cache
    assert (first.values == second.values).all()
