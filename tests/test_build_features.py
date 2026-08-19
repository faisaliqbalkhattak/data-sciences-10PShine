import numpy as np
import pandas as pd
import pytest

from src.build_features import (
    add_calendar_columns,
    build_feature_frames,
    merge_raw_frames,
)
from src.train_hourly import BASE_FEATURES, build_hourly_training_frame


def _raw_frames(rows=200, start="2026-01-01"):
    index = pd.date_range(start, periods=rows, freq="h")
    aq = pd.DataFrame(
        {"time": index, **{c: 15.0 + np.arange(rows) % 30 for c in BASE_FEATURES}},
    )
    aq["source"] = "open_meteo_air_quality"
    weather = pd.DataFrame(
        {"time": index, **{c: 20.0 + np.arange(rows) % 25 for c in BASE_FEATURES}},
    )
    weather["source"] = "open_meteo_weather_features"
    return aq, weather


def test_merge_raw_frames_aligns_and_drops_metadata():
    aq, weather = _raw_frames()
    master = merge_raw_frames(aq, weather)

    assert isinstance(master.index, pd.DatetimeIndex)
    assert "source" not in master.columns
    assert "source_weather" not in master.columns
    assert master.index.is_monotonic_increasing
    assert not master.index.has_duplicates
    assert len(master) == 200


def test_calendar_columns_are_valid_sin_cos_pairs():
    aq, weather = _raw_frames(rows=24 * 30)
    master = add_calendar_columns(merge_raw_frames(aq, weather))

    for pair in (("hour_sin", "hour_cos"), ("month_sin", "month_cos")):
        sin, cos = master[pair[0]], master[pair[1]]
        assert np.allclose(sin**2 + cos**2, 1.0, atol=1e-9)


def test_build_feature_frames_matches_notebook_contract():
    aq, weather = _raw_frames(rows=200)
    master, daily = build_feature_frames(aq, weather)

    # The hourly master keeps the full aligned grid plus calendar columns.
    assert len(master) == 200
    assert {"hour_sin", "hour_cos", "month_sin", "month_cos"}.issubset(master.columns)
    # The daily frame has the EPA daily target and category.
    assert "aqi_us_epa" in daily.columns
    assert "aqi_category" in daily.columns
    assert "hour_count" in daily.columns

    # The master feeds the validated hourly training builder unchanged.
    frame = build_hourly_training_frame(master, include_targets=False)
    assert frame.index.is_monotonic_increasing
    assert not frame.empty


def test_missing_raw_file_raises_clear_error(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="No raw file matched"):
        from src.build_features import build_from_raw

        build_from_raw()
