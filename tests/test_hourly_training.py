import numpy as np
import pandas as pd
import pytest

from src.train_hourly import (
    BLOCK_TARGET_COLUMNS,
    HOURLY_TARGET_COLUMNS,
    MAX_FORECAST_HOURS,
    TARGET_COLUMNS,
    build_hourly_targets,
    build_hourly_training_frame,
    chronological_split,
    evaluate_outputs,
    rolling_origin_splits,
    seasonal_persistence_predictions,
)


def make_aqi(rows=120):
    index = pd.date_range("2026-01-01", periods=rows, freq="h")
    return pd.Series(np.arange(rows, dtype=float), index=index)


def test_hourly_target_grid_has_24_points_and_six_blocks():
    targets = build_hourly_targets(make_aqi())

    assert len(HOURLY_TARGET_COLUMNS) == 24
    assert len(BLOCK_TARGET_COLUMNS) == 6
    assert len(TARGET_COLUMNS) == 30

    origin = targets.index[0]
    assert targets.loc[origin, "aqi_plus_01h"] == 1
    assert targets.loc[origin, "aqi_plus_24h"] == 24
    assert targets.loc[origin, "aqi_mean_25_30h"] == pytest.approx(np.mean(range(25, 31)))
    assert targets.loc[origin, "aqi_mean_49_60h"] == pytest.approx(np.mean(range(49, 61)))


def test_hourly_training_rejects_missing_observations():
    hourly = pd.DataFrame(
        {"pm2_5": np.ones(30)},
        index=pd.date_range("2026-01-01", periods=30, freq="h"),
    )
    hourly.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing observed"):
        build_hourly_training_frame(hourly)


def test_hourly_target_requires_complete_cadence():
    aqi = make_aqi()
    with pytest.raises(ValueError, match="complete one-hour cadence"):
        build_hourly_targets(aqi.drop(aqi.index[4]))


def test_hourly_split_purges_full_target_horizon():
    frame = pd.DataFrame({"value": range(200)})
    train, test = chronological_split(frame)
    split_at = int(len(frame) * 0.8)

    assert len(train) == split_at - MAX_FORECAST_HOURS
    assert test.index.min() == split_at
    assert train.index.max() == split_at - MAX_FORECAST_HOURS - 1


def test_hourly_split_rejects_shorter_gap():
    with pytest.raises(ValueError, match="72-hour"):
        chronological_split(pd.DataFrame({"value": range(200)}), gap=24)


def test_rolling_origin_splits_expand_and_embargo():
    frame = pd.DataFrame({"value": range(1000)})
    folds = rolling_origin_splits(frame, n_splits=3, test_size=100)

    assert len(folds) == 3
    for fold, train, test in folds:
        assert test.index.min() - train.index.max() - 1 == MAX_FORECAST_HOURS
        assert len(test) == 100
        assert fold in {1, 2, 3}
    assert len(folds[1][1]) > len(folds[0][1])


def test_seasonal_persistence_uses_only_prior_days():
    frame = pd.DataFrame(
        {"aqi_hourly_rolling": np.arange(200, dtype=float)},
        index=pd.date_range("2026-01-01", periods=200, freq="h"),
    )
    predictions = seasonal_persistence_predictions(frame)[72:]
    # The first evaluated origin is 72 hours into the source history; its
    # one-hour forecast is the value at origin-23 hours.
    assert predictions[0, 0] == 49
    assert predictions[0, 23] == 72
    assert predictions[0, 24] == pytest.approx(np.mean(range(49, 55)))


def test_hourly_metrics_include_mse_and_public_health_diagnostics():
    y_true = pd.DataFrame(np.array([[0.0] * 30, [200.0] * 30]))
    y_pred = np.array([[0.0] * 30, [150.0] * 30])
    rows, grouped = evaluate_outputs(y_true, y_pred, "test")

    assert rows[0]["mse"] == pytest.approx(1250.0)
    assert "category_accuracy" in grouped["hourly_points"]
    assert "category_macro_f1" in grouped["hourly_points"]
    assert "high_aqi_recall" in grouped["hourly_points"]
