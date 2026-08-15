import numpy as np
import pandas as pd
import pytest

from src.aqi import (
    aqi_category,
    calculate_daily_us_aqi,
    calculate_hourly_us_aqi,
    calculate_subindex,
)


def test_pm25_uses_current_epa_breakpoints_without_float_gaps():
    assert calculate_subindex(9.0, "pm2_5") == 50
    # Values between published decimal intervals are truncated before lookup.
    assert calculate_subindex(9.09, "pm2_5") == 50
    assert calculate_subindex(9.1, "pm2_5") == 51
    assert calculate_subindex(35.4, "pm2_5") == 100
    assert calculate_subindex(35.5, "pm2_5") == 101
    assert calculate_subindex(125.4, "pm2_5") == 200
    assert calculate_subindex(125.5, "pm2_5") == 201


def test_categories_follow_epa_ranges():
    assert aqi_category(50) == "Good"
    assert aqi_category(100) == "Moderate"
    assert aqi_category(150) == "Unhealthy for Sensitive Groups"
    assert aqi_category(200) == "Unhealthy"
    assert aqi_category(300) == "Very Unhealthy"
    assert aqi_category(301) == "Hazardous"
    assert aqi_category(np.nan) is None


def test_daily_aqi_uses_a_24_hour_pm25_window_and_max_subindex():
    index = pd.date_range("2026-01-01", periods=24, freq="h")
    hourly = pd.DataFrame(
        {
            "pm2_5": 35.4,
            "pm10": 254.0,
        },
        index=index,
    )
    result = calculate_daily_us_aqi(hourly)

    assert result.loc[pd.Timestamp("2026-01-01"), "aqi_pm2_5"] == 100
    assert result.loc[pd.Timestamp("2026-01-01"), "aqi_pm10"] == 150
    assert result.loc[pd.Timestamp("2026-01-01"), "aqi_us_epa"] == 150
    assert result.loc[pd.Timestamp("2026-01-01"), "dominant_pollutant"] == "pm10"
    assert result.loc[pd.Timestamp("2026-01-01"), "aqi_category"] == "Unhealthy for Sensitive Groups"


def test_eight_hour_ozone_window_is_not_a_daily_mean():
    index = pd.date_range("2026-01-01", periods=8, freq="h")
    # Approximately 0.070 ppm, the top of the EPA Moderate ozone band.
    hourly = pd.DataFrame({"ozone": 137.43}, index=index)
    result = calculate_hourly_us_aqi(hourly)
    assert result["aqi_ozone"].iloc[:7].isna().all()
    assert result["aqi_ozone"].iloc[7] == 100


def test_incomplete_or_duplicate_hourly_input_fails_closed():
    index = pd.date_range("2026-01-01", periods=24, freq="h").delete(5)
    with pytest.raises(ValueError, match="complete hourly cadence"):
        calculate_hourly_us_aqi(pd.DataFrame({"pm2_5": 1.0}, index=index))

    duplicate_index = pd.date_range("2026-01-01", periods=24, freq="h").insert(5, pd.Timestamp("2026-01-01 04:00"))
    with pytest.raises(ValueError, match="duplicate"):
        calculate_hourly_us_aqi(pd.DataFrame({"pm2_5": 1.0}, index=duplicate_index))


def test_negative_concentration_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        calculate_subindex(-1, "pm2_5")
