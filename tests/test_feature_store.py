import numpy as np
import pandas as pd
import pytest

from src.feature_store import (
    backfill_daily,
    backfill_hourly,
    get_daily_features,
    get_hourly_features,
    get_hourly_observations,
    get_hourly_raw,
    latest_hourly_origin,
    store_stats,
    validate_feature_schema,
)
from src.train import build_training_frame
from src.train_hourly import (
    BASE_FEATURES,
    TARGET_COLUMNS,
    build_hourly_training_frame,
)

HOURLY_COLUMNS = ["time", *BASE_FEATURES]


def _synthetic_hourly(rows=200, start="2026-01-01"):
    index = pd.date_range(start, periods=rows, freq="h")
    frame = pd.DataFrame(index=index)
    for column in BASE_FEATURES:
        frame[column] = 10.0 + np.arange(rows, dtype=float) % 40
    frame["time"] = index
    return frame[HOURLY_COLUMNS]


def _synthetic_daily(rows=500, start="2024-01-01"):
    index = pd.date_range(start, periods=rows, freq="D")
    frame = pd.DataFrame(index=index)
    for column in BASE_FEATURES:
        frame[column] = 10.0 + np.arange(rows, dtype=float) % 40
    frame["aqi_us_epa"] = 30.0 + np.arange(rows, dtype=float) % 70
    frame["time"] = index
    return frame


def test_hourly_backfill_stores_features_and_targets(tmp_path):
    source = tmp_path / "hourly.csv"
    _synthetic_hourly().to_csv(source, index=False)
    store = tmp_path / "store.duckdb"

    summary = backfill_hourly(source, store_path=store, replace=True)

    assert summary["rows"] > 0
    # Future 72-hour targets cannot exist for the last rows of the series, so
    # the training frame (features + targets) is a strict suffix subset.
    assert 0 < summary["target_rows"] <= summary["rows"]

    features = get_hourly_observations(store_path=store)
    full = get_hourly_features(store_path=store)
    raw = get_hourly_raw(store_path=store)

    assert isinstance(features.index, pd.DatetimeIndex)
    assert set(TARGET_COLUMNS).issubset(full.columns)
    assert not set(TARGET_COLUMNS).intersection(features.columns)
    assert set(BASE_FEATURES).issubset(raw.columns)
    assert not features.empty


def test_backfill_matches_the_validated_training_contract(tmp_path):
    source = tmp_path / "hourly.csv"
    _synthetic_hourly().to_csv(source, index=False)
    store = tmp_path / "store.duckdb"
    backfill_hourly(source, store_path=store, replace=True)

    hourly = pd.read_csv(source, parse_dates=["time"])
    expected = build_hourly_training_frame(hourly, include_targets=False)
    stored = get_hourly_observations(store_path=store)

    pd.testing.assert_frame_equal(stored, expected)
    validate_feature_schema(list(expected.columns), store_path=store)


def test_schema_validation_detects_drift(tmp_path):
    source = tmp_path / "hourly.csv"
    _synthetic_hourly().to_csv(source, index=False)
    store = tmp_path / "store.duckdb"
    backfill_hourly(source, store_path=store, replace=True)

    with pytest.raises(ValueError, match="schema drift"):
        validate_feature_schema(["not_a_real_feature"], store_path=store)


def test_daily_backfill_and_retrieval(tmp_path):
    source = tmp_path / "daily.csv"
    _synthetic_daily().to_csv(source, index=False)
    store = tmp_path / "store.duckdb"

    summary = backfill_daily(source, store_path=store, replace=True)
    assert summary["rows"] > 0

    daily = get_daily_features(store_path=store)
    assert isinstance(daily.index, pd.DatetimeIndex)
    assert {"target_1d", "target_2d", "target_3d"}.issubset(daily.columns)

    expected = build_training_frame(pd.read_csv(source))
    pd.testing.assert_frame_equal(daily, expected)


def test_stats_and_latest_origin(tmp_path):
    source = tmp_path / "hourly.csv"
    _synthetic_hourly().to_csv(source, index=False)
    store = tmp_path / "store.duckdb"
    backfill_hourly(source, store_path=store, replace=True)
    daily_source = tmp_path / "daily.csv"
    _synthetic_daily().to_csv(daily_source, index=False)
    backfill_daily(daily_source, store_path=store, replace=True)

    stats = store_stats(store_path=store)
    tables = {row["table"] for row in stats}
    assert {"hourly_raw", "hourly_observations", "hourly_features", "daily_features"}.issubset(tables)
    assert all(row["schema_version"] == 1 for row in stats)
    assert latest_hourly_origin(store_path=store) is not None


def test_backfill_is_idempotent_with_replace(tmp_path):
    source = tmp_path / "hourly.csv"
    _synthetic_hourly().to_csv(source, index=False)
    store = tmp_path / "store.duckdb"

    first = backfill_hourly(source, store_path=store, replace=True)
    second = backfill_hourly(source, store_path=store, replace=True)

    assert first["rows"] == second["rows"]
    assert first["target_rows"] == second["target_rows"]
    assert len(get_hourly_features(store_path=store)) == first["target_rows"]
