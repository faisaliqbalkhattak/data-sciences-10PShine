import numpy as np
import pandas as pd
import pytest

from src.train import (
    MAX_HORIZON,
    TARGET,
    build_training_frame,
    chronological_split,
    evaluate,
    validate_daily_input,
)


def make_daily(rows=24):
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "time": index,
            TARGET: np.arange(rows, dtype=float),
            "pm2_5": np.arange(rows, dtype=float) + 10,
        }
    )


def test_daily_input_rejects_duplicates_and_missing_days():
    daily = make_daily()
    duplicate = pd.concat([daily, daily.iloc[[2]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_daily_input(duplicate)

    missing_day = daily.drop(index=5).reset_index(drop=True)
    with pytest.raises(ValueError, match="complete one-day cadence"):
        validate_daily_input(missing_day)


def test_direct_targets_shift_forward_by_each_horizon():
    frame = build_training_frame(make_daily())
    first = frame.index[0]
    assert frame.loc[first, "target_1d"] == frame.loc[first + pd.Timedelta(days=1), TARGET]
    assert frame.loc[first, "target_2d"] == frame.loc[first + pd.Timedelta(days=2), TARGET]
    assert frame.loc[first, "target_3d"] == frame.loc[first + pd.Timedelta(days=3), TARGET]


def test_chronological_split_purges_the_maximum_horizon():
    frame = pd.DataFrame({"value": range(20)})
    train, test = chronological_split(frame, test_fraction=0.2)

    assert len(train) == 16 - MAX_HORIZON
    assert train.index.max() == 12
    assert test.index.min() == 16
    assert set(range(13, 16)).isdisjoint(train.index)


def test_evaluate_returns_regression_metrics():
    result = evaluate(np.array([1.0, 3.0]), np.array([1.0, 2.0]))
    assert result["mae"] == 0.5
    assert result["rmse"] == pytest.approx(np.sqrt(0.5))
    assert set(result) == {"rmse", "mae", "r2"}
