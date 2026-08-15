"""US EPA AQI calculation for hourly Open-Meteo pollutant concentrations.

The active project target is ``aqi_us_epa``.  This module follows the US EPA
AQI method rather than treating a daily mean as an AQI input:

* PM2.5 and PM10 use trailing 24-hour concentrations.
* O3 and CO use trailing 8-hour concentrations.
* SO2 and NO2 use the 1-hour concentration.
* Concentrations are converted to the units used by the EPA tables, truncated
  as required, interpolated between complete breakpoint intervals, and rounded
  to the nearest AQI integer.

The result is an AQI calculated from Open-Meteo modeled concentrations. It is
not a station measurement and cannot be presented as local ground truth.

The EPA tables are published by the US EPA Air Quality System and 40 CFR Part
58 Appendix G; the project pins the table version in this source file so a
future standards change requires an explicit code and test update.

Reference: US EPA, Technical Assistance Document for Reporting the Daily AQI,
and 40 CFR Part 58 Appendix G. The PM2.5 breakpoints below include the EPA
breakpoint update effective May 2024.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

# Breakpoints are (low concentration, high concentration, low AQI, high AQI).
# Concentration units are pollutant-specific and documented in the keys.
EPA_BREAKPOINTS: Final[dict[str, tuple[tuple[float, float, int, int], ...]]] = {
    "pm2_5": (
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 500.4, 401, 500),
    ),
    "pm10": (
        (0.0, 54.0, 0, 50),
        (55.0, 154.0, 51, 100),
        (155.0, 254.0, 101, 150),
        (255.0, 354.0, 151, 200),
        (355.0, 424.0, 201, 300),
        (425.0, 504.0, 301, 400),
        (505.0, 604.0, 401, 500),
    ),
    # Ozone values are ppm after conversion from Open-Meteo ug/m3.
    "ozone_8h": (
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ),
    # The 1-hour ozone table is used for the high-ozone AQI range.
    "ozone_1h": (
        (0.125, 0.164, 101, 150),
        (0.165, 0.204, 151, 200),
        (0.205, 0.404, 201, 300),
        (0.405, 0.504, 301, 400),
        (0.505, 0.604, 401, 500),
    ),
    # CO values are ppm.
    "carbon_monoxide": (
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ),
    # SO2 and NO2 values are ppb.
    "sulphur_dioxide": (
        (0.0, 35.0, 0, 50),
        (36.0, 75.0, 51, 100),
        (76.0, 185.0, 101, 150),
        (186.0, 304.0, 151, 200),
        (305.0, 604.0, 201, 300),
        (605.0, 1004.0, 301, 400),
        (1005.0, 1604.0, 401, 500),
    ),
    "nitrogen_dioxide": (
        (0.0, 53.0, 0, 50),
        (54.0, 100.0, 51, 100),
        (101.0, 360.0, 101, 150),
        (361.0, 649.0, 151, 200),
        (650.0, 1249.0, 201, 300),
        (1250.0, 1649.0, 301, 400),
        (1650.0, 2049.0, 401, 500),
    ),
}

AQI_CATEGORY_BOUNDS: Final[tuple[tuple[int, int, str], ...]] = (
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
)

# Open-Meteo returns mass concentration in ug/m3. The ideal gas conversion is
# ppm = ug/m3 * 24.45 / (molecular weight * 1000) at 25 C and 1 atm.
_OZONE_PPM_PER_UG_M3 = 24.45 / (48.0 * 1000.0)
_CO_PPM_PER_UG_M3 = 24.45 / (28.01 * 1000.0)
_NO2_PPB_PER_UG_M3 = 24.45 / 46.005
_SO2_PPB_PER_UG_M3 = 24.45 / 64.066


def _truncate(value: float, decimals: int) -> float:
    """Truncate toward zero, as required before EPA interpolation."""
    factor = 10**decimals
    return float(np.trunc(value * factor) / factor)


def aqi_category(aqi: float | int | None) -> str | None:
    """Return the standard EPA category for an AQI value."""
    if aqi is None or pd.isna(aqi):
        return None
    value = int(round(float(aqi)))
    for low, high, category in AQI_CATEGORY_BOUNDS:
        if low <= value <= high:
            return category
    return "Hazardous" if value > 500 else None


def calculate_subindex(concentration: float, pollutant: str) -> float:
    """Calculate one EPA sub-index from an already converted concentration.

    ``pollutant`` must be a key in :data:`EPA_BREAKPOINTS`. The input is
    truncated using the pollutant's EPA precision before interpolation.
    Values above the table's maximum are capped at AQI 500, matching the
    reporting range; negative values are rejected as invalid measurements.
    """
    if pollutant not in EPA_BREAKPOINTS:
        raise KeyError(f"Unsupported EPA AQI pollutant: {pollutant}")
    if pd.isna(concentration):
        return np.nan
    value = float(concentration)
    if value < 0:
        raise ValueError(f"AQI concentration cannot be negative: {value}")

    decimals = {
        "pm2_5": 1,
        "pm10": 0,
        "ozone_8h": 3,
        "ozone_1h": 3,
        "carbon_monoxide": 1,
        "sulphur_dioxide": 0,
        "nitrogen_dioxide": 0,
    }[pollutant]
    value = _truncate(value, decimals)
    breakpoints = EPA_BREAKPOINTS[pollutant]
    if value > breakpoints[-1][1]:
        return 500.0
    if value < breakpoints[0][0]:
        # For example, the EPA 1-hour ozone table starts at AQI 101;
        # lower ozone is represented by the valid 8-hour table instead.
        return np.nan
    for low, high, ilow, ihigh in breakpoints:
        if low <= value <= high:
            interpolated = ilow + (ihigh - ilow) * (value - low) / (high - low)
            return float(np.floor(interpolated + 0.5))
    # Gaps are intentionally impossible after EPA-required truncation. Raise
    # instead of silently returning a missing target if a table is malformed.
    raise ValueError(
        f"No EPA AQI breakpoint contains {value} for pollutant {pollutant}."
    )


def _as_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an hourly local-time frame."""
    result = frame.copy()
    if "time" in result.columns:
        result["time"] = pd.to_datetime(result["time"])
        result = result.set_index("time")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError("AQI input must have a DatetimeIndex or a 'time' column.")
    if result.index.has_duplicates:
        raise ValueError("AQI input contains duplicate timestamps.")
    if not result.index.is_monotonic_increasing:
        raise ValueError("AQI input timestamps must be sorted ascending.")
    if len(result.index) > 1:
        intervals = result.index.to_series().diff().dropna()
        if not (intervals == pd.Timedelta(hours=1)).all():
            raise ValueError("AQI input must contain a complete hourly cadence.")
    return result


def _subindex_series(values: pd.Series, pollutant: str) -> pd.Series:
    return values.map(lambda value: calculate_subindex(value, pollutant))


def calculate_hourly_us_aqi(hourly: pd.DataFrame) -> pd.DataFrame:
    """Return hourly rolling pollutant sub-indices using the US EPA method."""
    frame = _as_hourly_frame(hourly)
    output = pd.DataFrame(index=frame.index)

    if "pm2_5" in frame:
        output["aqi_pm2_5"] = _subindex_series(
            frame["pm2_5"].rolling(24, min_periods=24).mean(), "pm2_5"
        )
    if "pm10" in frame:
        output["aqi_pm10"] = _subindex_series(
            frame["pm10"].rolling(24, min_periods=24).mean(), "pm10"
        )
    if "carbon_monoxide" in frame:
        co_ppm = (
            frame["carbon_monoxide"].rolling(8, min_periods=8).mean()
            * _CO_PPM_PER_UG_M3
        )
        output["aqi_carbon_monoxide"] = _subindex_series(co_ppm, "carbon_monoxide")
    if "ozone" in frame:
        ozone_ppm_8h = (
            frame["ozone"].rolling(8, min_periods=8).mean() * _OZONE_PPM_PER_UG_M3
        )
        ozone_ppm_1h = frame["ozone"] * _OZONE_PPM_PER_UG_M3
        ozone_8h = _subindex_series(ozone_ppm_8h, "ozone_8h")
        ozone_1h = _subindex_series(ozone_ppm_1h, "ozone_1h")
        output["aqi_ozone"] = pd.concat([ozone_8h, ozone_1h], axis=1).max(axis=1)
    if "sulphur_dioxide" in frame:
        so2_ppb = frame["sulphur_dioxide"] * _SO2_PPB_PER_UG_M3
        output["aqi_sulphur_dioxide"] = _subindex_series(
            so2_ppb, "sulphur_dioxide"
        )
    if "nitrogen_dioxide" in frame:
        no2_ppb = frame["nitrogen_dioxide"] * _NO2_PPB_PER_UG_M3
        output["aqi_nitrogen_dioxide"] = _subindex_series(
            no2_ppb, "nitrogen_dioxide"
        )

    if not len(output.columns):
        raise ValueError("AQI input contains no supported pollutant columns.")
    return output


def calculate_daily_us_aqi(hourly: pd.DataFrame) -> pd.DataFrame:
    """Return daily AQI, category, dominant pollutant, and sub-indices.

    Each pollutant's daily value is the maximum valid rolling sub-index
    observed during that local calendar day. A day with no complete valid
    averaging window has a missing AQI and must not be used as a target.
    """
    frame = _as_hourly_frame(hourly)
    hourly_indices = calculate_hourly_us_aqi(frame)
    aqi_columns = list(hourly_indices.columns)
    daily = hourly_indices.resample("D").max()

    # EPA daily reporting uses the complete calendar-day 24-hour average for
    # particulate matter. Other pollutants use the maximum valid rolling
    # window observed during the day.
    for pollutant, column in (("pm2_5", "aqi_pm2_5"), ("pm10", "aqi_pm10")):
        if pollutant in frame and column in daily:
            daily_average = frame[pollutant].resample("D").agg(
                lambda values: values.mean()
                if len(values) == 24 and values.notna().all()
                else np.nan
            )
            daily[column] = _subindex_series(daily_average, pollutant)

    daily["aqi_us_epa"] = daily[aqi_columns].max(axis=1)
    daily["aqi_category"] = daily["aqi_us_epa"].map(aqi_category)
    dominant = daily[aqi_columns].idxmax(axis=1, skipna=True)
    daily["dominant_pollutant"] = dominant.str.removeprefix("aqi_")
    return daily
