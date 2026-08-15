import numpy as np
import pandas as pd
import pytest

from src.inference_hourly import forecast_rows
from src.train_hourly import TARGET_COLUMNS


def test_forecast_rows_labels_point_and_block_outputs():
    origin = pd.Timestamp("2026-08-13 00:00")
    result = forecast_rows(origin, np.arange(len(TARGET_COLUMNS), dtype=float))

    assert len(result) == 30
    assert result.iloc[0]["kind"] == "point"
    assert result.iloc[0]["start_time"] == origin + pd.Timedelta(hours=1)
    assert result.iloc[23]["end_time"] == origin + pd.Timedelta(hours=24)
    assert result.iloc[24]["kind"] == "six_hour_mean"
    assert result.iloc[24]["start_time"] == origin + pd.Timedelta(hours=25)
    assert result.iloc[24]["end_time"] == origin + pd.Timedelta(hours=30)
    assert result.iloc[28]["kind"] == "twelve_hour_mean"
    assert result.iloc[29]["end_time"] == origin + pd.Timedelta(hours=72)


def test_forecast_rows_rejects_wrong_output_count():
    with pytest.raises(ValueError, match="Expected 30"):
        forecast_rows(pd.Timestamp("2026-08-13"), np.zeros(29))
