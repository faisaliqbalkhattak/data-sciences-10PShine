"""Build the four executable notebooks for the Karak Open-Meteo-only workflow.

Run from ``development`` with Python 3.11.9. The builders
use only the standard library so notebook generation is independent of data
and network access.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text],
    }


def notebook(cells: list[dict]) -> dict:
    for cell in cells:
        cell["id"] = str(uuid.uuid4())[:8]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3.11.9", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_IMPORTS = """from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'data').exists():
    PROJECT_ROOT = (PROJECT_ROOT / '..').resolve()
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))
from src.aqi import calculate_daily_us_aqi
print('Project root:', PROJECT_ROOT)
print('Raw data directory:', RAW_DIR)
"""


NB01 = [
    md("""# 01 — Raw data quality check (Open-Meteo only)

**Purpose:** establish whether the active Karak datasets are complete and internally consistent before feature engineering.

**Order:** run this after `python -m src.ingest`, before notebook 02. The notebook selects only files with the explicit `karak_aqi_training_open_meteo_hourly_...` and `karak_weather_features_open_meteo_hourly_...` names. No secondary-provider file can satisfy these patterns.

**Interpretation rule:** every conclusion below is printed from the loaded files. The prose is a decision record, not a generic assumption.
"""),
    code(COMMON_IMPORTS),
    code("""def newest(pattern):
    matches = sorted(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No file matched {pattern}. Run `python -m src.ingest` first.')
    return matches[-1]

aq_path = newest('karak_aqi_training_open_meteo_hourly_*.csv')
wx_path = newest('karak_weather_features_open_meteo_hourly_*.csv')
aq = pd.read_csv(aq_path, parse_dates=['time'])
wx = pd.read_csv(wx_path, parse_dates=['time'])
print('AQ file:', aq_path.name)
print('Weather file:', wx_path.name)
print('AQ rows/columns:', len(aq), len(aq.columns))
print('Weather rows/columns:', len(wx), len(wx.columns))
print('AQ range:', aq['time'].min(), '->', aq['time'].max())
print('Weather range:', wx['time'].min(), '->', wx['time'].max())
print('AQ source labels:', aq['source'].dropna().unique().tolist())
print('Weather source labels:', wx['source'].dropna().unique().tolist())
expected_aq_start = pd.Timestamp('2022-08-05')
# Derive the requested end date from the filename and require both files to
# contain that same final hour. This remains valid on a future data refresh.
def filename_end_date(path):
    return path.name.split('_to_', 1)[1].split('_', 1)[0]
file_end_aq = filename_end_date(aq_path)
file_end_wx = filename_end_date(wx_path)
actual_end_date = aq['time'].max().date().isoformat()
contract_pass = (
    aq_path.name.startswith('karak_aqi_training_open_meteo_hourly_2022-08-05_to_')
    and wx_path.name.startswith('karak_weather_features_open_meteo_hourly_2022-08-05_to_')
    and aq['time'].min() == expected_aq_start
    and wx['time'].min() == expected_aq_start
    and aq['time'].max() == wx['time'].max()
    and file_end_aq == file_end_wx == actual_end_date
    and all(str(label).startswith('open_meteo_') for label in aq['source'].dropna().unique())
    and all(str(label).startswith('open_meteo_') for label in wx['source'].dropna().unique())
)
print('Filename/date/source contract:', 'PASS' if contract_pass else 'REVIEW')
"""),
    md("""### Finding from file identity and coverage

The output immediately above is the audit trail: it records the exact filenames, row counts, date ranges, and source labels used for this run. The active files should report only `open_meteo_*` labels. The following cell converts those values into a pass/fail statement so a later refresh cannot silently reuse the old narrative.

**Diagnostic note from the first pull:** the original 2022-08-01 start produced a contiguous 77-hour all-variable AQ null block through 2022-08-04 04:00. It was not imputed. The active AQ and weather-feature files begin on 2022-08-05; the original pulls are retained outside `data/raw/` for audit only.
"""),
    code("""def hourly_gap_report(frame, label):
    times = pd.to_datetime(frame['time'])
    duplicates = int(times.duplicated().sum())
    expected = pd.date_range(times.min(), times.max(), freq='h')
    missing = expected.difference(pd.DatetimeIndex(times))
    print(f'{label}: duplicates={duplicates}, missing hourly timestamps={len(missing)}, expected_hours={len(expected)}')
    return duplicates, missing

aq_dupes, aq_missing = hourly_gap_report(aq, 'AQ')
wx_dupes, wx_missing = hourly_gap_report(wx, 'weather')
na_report = pd.DataFrame({'aq_missing_values': aq.isna().sum(), 'weather_missing_values': wx.isna().sum()}).fillna(0).astype(int)
print('\\nMissing-value counts by column:')
print(na_report.to_string())
quality_pass = (contract_pass and aq_dupes == 0 and wx_dupes == 0 and len(aq_missing) == 0 and len(wx_missing) == 0 and int(aq.isna().sum().sum()) == 0 and int(wx.isna().sum().sum()) == 0)
print('\\nQC verdict:', 'PASS — no duplicates, hourly gaps, or missing cells in the active files.' if quality_pass else 'REVIEW — at least one completeness rule failed; do not train yet.')
"""),
    md("""### Findings from the executed completeness check

The printed `QC verdict` is the authoritative result for this pull. A `PASS` means the two active Open-Meteo files can proceed to notebook 02 without imputation for missing cells or timestamps. A `REVIEW` means the files must remain active but the issue must be fixed and this notebook rerun before modeling. The initial 77-hour source gap was resolved by changing the requested start date, not by filling values.
"""),
    code("""aq_numeric = aq.select_dtypes(include='number')
negative_counts = (aq_numeric < 0).sum().sort_values(ascending=False)
print('Negative-value counts in AQ numeric columns:')
print(negative_counts.to_string())
for column in ['pm2_5', 'pm10', 'ozone']:
    if column in aq:
        print(f'{column}: min={aq[column].min():.3f}, median={aq[column].median():.3f}, max={aq[column].max():.3f}')
print('\\nSource labels:', aq['source'].dropna().unique().tolist(), wx['source'].dropna().unique().tolist())
"""),
    md("""### Finding from physical-range checks

The numeric output above is deliberately retained with the notebook. Negative pollutant concentrations indicate a source/parse problem; unusually large values are flagged for review but are not automatically deleted because dust events can be real. Source labels must identify only `open_meteo_*` values in the active workflow.
"""),
    code("""summary = pd.DataFrame({
    'rows': [len(aq), len(wx)],
    'columns': [len(aq.columns), len(wx.columns)],
    'start': [aq.time.min(), wx.time.min()],
    'end': [aq.time.max(), wx.time.max()],
    'missing_cells': [int(aq.isna().sum().sum()), int(wx.isna().sum().sum())],
    'contract_pass': [contract_pass, contract_pass],
}, index=['aqi_training', 'weather_features'])
summary.to_csv(PROCESSED_DIR / 'qc_raw_open_meteo_summary.csv')
print(summary.to_string())
print('\\nSaved evidence table:', PROCESSED_DIR / 'qc_raw_open_meteo_summary.csv')
"""),
    md("""## 01 decision

This notebook does not compare providers. It verifies that the **single active provider's two products** are structurally usable. The historical OpenWeather/AQICN sanity check is documented in `data_sources_and_file_naming.md` and is not required to pass this notebook.
"""),
]


NB02 = [
    md("""# 02 — Open-Meteo feature EDA and AQI analysis

**Purpose:** combine the active Open-Meteo air-quality and weather-feature files, calculate the US EPA AQI with pollutant-specific averaging windows, examine seasonality and weather relationships, and save model-ready outputs.

**Important separation:** the long-run weather-trend file is not loaded here. It belongs to notebook 04 and has a different analytical purpose.
"""),
    code(COMMON_IMPORTS),
    code("""def newest(pattern):
    matches = sorted(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No file matched {pattern}. Run `python -m src.ingest` first.')
    return matches[-1]

aq_path = newest('karak_aqi_training_open_meteo_hourly_*.csv')
wx_path = newest('karak_weather_features_open_meteo_hourly_*.csv')
aq = pd.read_csv(aq_path, parse_dates=['time']).set_index('time')
wx = pd.read_csv(wx_path, parse_dates=['time']).set_index('time')
aq = aq.drop(columns=['source'], errors='ignore')
wx = wx.drop(columns=['source'], errors='ignore')
master = aq.join(wx, how='inner', rsuffix='_weather').sort_index()
print('Loaded:', aq_path.name)
print('Loaded:', wx_path.name)
print('Master shape:', master.shape)
print('Master range:', master.index.min(), '->', master.index.max())
"""),
    md("""### Finding from the active master frame

The output records how many aligned hours remain after joining the two Open-Meteo products. This is the number available to feature engineering; it is not inferred from a previous run.
"""),
    code("""pollutant_cols = [c for c in ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'aerosol_optical_depth', 'dust', 'uv_index'] if c in master]
weather_cols = [c for c in ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation', 'rain', 'surface_pressure', 'cloud_cover', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'] if c in master]
agg = {c: 'mean' for c in pollutant_cols + weather_cols}
for c in ['precipitation', 'rain']:
    if c in agg:
        agg[c] = 'sum'
daily = master[pollutant_cols + weather_cols].resample('D').agg(agg)
daily = daily.dropna(subset=['pm2_5'])
daily['hour_count'] = master['pm2_5'].resample('D').count().reindex(daily.index)
print('Daily rows with PM2.5:', len(daily))
print('Median hourly observations per retained day:', daily['hour_count'].median())
print('Days with fewer than 20 observations:', int((daily['hour_count'] < 20).sum()))
"""),
    code("""# Official US EPA AQI from pollutant-specific rolling windows.
# calculate_daily_us_aqi applies EPA unit conversions, truncation, breakpoint
# interpolation, and category assignment. The output is an AQI calculated from
# Open-Meteo modeled concentrations, not a station measurement.
aqi_daily = calculate_daily_us_aqi(master[pollutant_cols])
daily = daily.join(aqi_daily, how='left')
daily = daily.dropna(subset=['aqi_us_epa'])
print('US EPA AQI days:', int(daily['aqi_us_epa'].notna().sum()), '/', len(daily))
print('US EPA AQI summary:')
print(daily['aqi_us_epa'].describe().round(1).to_string())
print('Highest modeled AQI day:', daily['aqi_us_epa'].idxmax(), int(daily['aqi_us_epa'].max()))
print('AQI categories:')
print(daily['aqi_category'].value_counts().to_string())
"""),
    md("""### Finding from the executed AQI calculation

The target is `aqi_us_epa`, calculated with the US EPA AQI method. Each pollutant uses its required averaging window and unit conversion, and the daily value is the maximum valid pollutant sub-index. Open-Meteo supplies modeled concentrations, so this remains a modeled estimate rather than Karak station ground truth.
"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))
monthly = daily['pm2_5'].groupby(daily.index.month).mean()
hourly = master['pm2_5'].groupby(master.index.hour).mean()
axes[0].plot(monthly.index, monthly.values, marker='o')
axes[0].set(title='Karak modeled PM2.5 by calendar month', xlabel='Month', ylabel='µg/m³')
axes[0].grid(alpha=.3)
axes[1].plot(hourly.index, hourly.values, marker='o', color='darkorange')
axes[1].set(title='Karak modeled PM2.5 by local hour', xlabel='Hour (Asia/Karachi)', ylabel='µg/m³')
axes[1].grid(alpha=.3)
fig.tight_layout()
fig.savefig(PROCESSED_DIR / 'karak_aqi_open_meteo_seasonality.png', dpi=120)
plt.show()
print('Highest monthly mean:', int(monthly.idxmax()), round(monthly.max(), 2))
print('Lowest monthly mean:', int(monthly.idxmin()), round(monthly.min(), 2))
print('Highest hourly mean:', int(hourly.idxmax()), round(hourly.max(), 2))
"""),
    md("""### Finding from seasonality output

The three printed extrema are the interpretation anchors for the figure: they show which month and local hour have the highest modeled PM2.5 in this actual dataset. They should not be explained as causal emissions effects without local observations; weather, dust, and model smoothing remain possible explanations.
"""),
    code("""available_corr = [c for c in ['pm2_5', 'pm10', 'ozone', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation', 'surface_pressure', 'cloud_cover'] if c in daily]
corr = daily[available_corr].corr()['pm2_5'].drop('pm2_5').sort_values(key=lambda s: s.abs(), ascending=False)
print('Correlations with daily PM2.5 (sorted by absolute value):')
print(corr.round(3).to_string())
"""),
    md("""### Finding from weather relationships

The correlation list above is descriptive, not proof of causation. It is nevertheless useful for feature selection because it is computed after daily aggregation from the same aligned Open-Meteo products used by the model.
"""),
    code("""master['hour_sin'] = np.sin(2 * np.pi * master.index.hour / 24)
master['hour_cos'] = np.cos(2 * np.pi * master.index.hour / 24)
master['month_sin'] = np.sin(2 * np.pi * master.index.month / 12)
master['month_cos'] = np.cos(2 * np.pi * master.index.month / 12)
daily.to_csv(PROCESSED_DIR / 'karak_aqi_open_meteo_daily_features.csv')
master.to_csv(PROCESSED_DIR / 'karak_aqi_open_meteo_hourly_features.csv')
print('Saved hourly features:', len(master))
print('Saved daily features:', len(daily))
"""),
    md("""## 02 decision

The active modeling frame contains only Open-Meteo air quality and Open-Meteo weather. The separate 2000–present weather trend analysis is intentionally kept out of this training frame and is documented in notebook 04.
"""),
]


NB03 = [
    md("""# 03 — Live Open-Meteo sanity check

**Purpose:** verify that the one active provider returns current Karak air-quality and weather observations with the expected timezone and fields. This is not a cross-provider comparison.
"""),
    code("""from pathlib import Path
from datetime import date, timedelta
import sys
import requests
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = (PROJECT_ROOT / '..').resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from config import LATITUDE, LONGITUDE, CITY_NAME, TIMEZONE, OPEN_METEO_AIR_QUALITY_URL, OPEN_METEO_WEATHER_FORECAST_URL

def get(url, params):
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()

today = date.today().isoformat()
air = get(OPEN_METEO_AIR_QUALITY_URL, {'latitude': LATITUDE, 'longitude': LONGITUDE, 'hourly': 'pm2_5,pm10,ozone,us_aqi', 'forecast_days': 2, 'timezone': TIMEZONE})
weather = get(OPEN_METEO_WEATHER_FORECAST_URL, {'latitude': LATITUDE, 'longitude': LONGITUDE, 'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation', 'forecast_days': 2, 'timezone': TIMEZONE})
print('Target:', CITY_NAME, LATITUDE, LONGITUDE)
print('API timezone:', air.get('timezone'), '| UTC offset seconds:', air.get('utc_offset_seconds'))
print('Air response hours:', len(air['hourly']['time']))
print('Weather response hours:', len(weather['hourly']['time']))
air_frame = pd.DataFrame(air['hourly'])
weather_frame = pd.DataFrame(weather['hourly'])
print('Air timestamps:', air_frame['time'].min(), '->', air_frame['time'].max())
print('Weather timestamps:', weather_frame['time'].min(), '->', weather_frame['time'].max())
print('Air missing cells:', int(air_frame.isna().sum().sum()))
print('Weather missing cells:', int(weather_frame.isna().sum().sum()))
print('Live Open-Meteo verdict:', 'PASS' if air.get('timezone') == TIMEZONE and weather.get('timezone') == TIMEZONE and not air_frame.isna().any().any() and not weather_frame.isna().any().any() else 'REVIEW')
"""),
    md("""### Finding from the live request

The cell above records the actual endpoint response, local timezone, row counts, date ranges, missing-cell counts, and a pass/review verdict. Because both requests are Open-Meteo, the check verifies operational consistency without introducing a secondary data source.
"""),
]


NB04 = [
    md("""# 04 — Karak historical weather trends (2000–present)

**Purpose:** describe how modeled weather has varied over the long term at the configured Karak/Sabir Abad coordinate. This is intentionally separate from AQI training.

**Dataset:** `karak_weather_trend_open_meteo_daily_2000-01-01_to_<end>_<pull>.csv`. Daily summaries are used instead of a second sparse API pull: this preserves enough information to calculate annual means, extremes, precipitation totals, and trend slopes while keeping the file small and purpose-specific.

**Caveat:** these are Open-Meteo reanalysis/model values, not local weather-station measurements. A trend is descriptive evidence for this grid point, not a claim about every part of Karak.
"""),
    code(COMMON_IMPORTS),
    code("""matches = sorted(RAW_DIR.glob('karak_weather_trend_open_meteo_daily_*.csv'))
if not matches:
    raise FileNotFoundError('No weather-trend file found. Run `python -m src.ingest` first.')
trend_path = matches[-1]
trend = pd.read_csv(trend_path, parse_dates=['time']).set_index('time').sort_index()
print('Trend file:', trend_path.name)
print('Rows:', len(trend), '| Columns:', len(trend.columns))
print('Range:', trend.index.min().date(), '->', trend.index.max().date())
print('Missing cells:', int(trend.isna().sum().sum()))
"""),
    md("""### Finding from the loaded trend file

The printed filename, row count, date range, and missing-cell count are the provenance check. The file must begin in 2000 and must not be confused with the 2022-present AQI training file.
"""),
    code("""annual = trend.resample('YS').agg({
    'temperature_2m_mean': 'mean',
    'temperature_2m_max': 'mean',
    'temperature_2m_min': 'mean',
    'precipitation_sum': 'sum',
    'rain_sum': 'sum',
    'wind_speed_10m_mean': 'mean',
    'wind_gusts_10m_max': 'max',
    'relative_humidity_2m_mean': 'mean',
})
annual.index = annual.index.year
print('Annual rows:', len(annual))
print('Latest annual values:')
print(annual.tail(3).round(2).to_string())
"""),
    code("""def slope_per_decade(series):
    # 2026 is incomplete at the time of this pull. Exclude the current year
    # from regression, especially for annual totals such as precipitation.
    complete = series[series.index < pd.Timestamp.today().year].dropna()
    x = complete.index.to_numpy(dtype=float)
    y = complete.to_numpy(dtype=float)
    return np.polyfit(x, y, 1)[0] * 10

trend_rows = []
for column in ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'rain_sum', 'wind_speed_10m_mean', 'relative_humidity_2m_mean']:
    if column in annual:
        complete_annual = annual.loc[annual.index < pd.Timestamp.today().year]
        trend_rows.append({'variable': column, 'slope_per_decade': slope_per_decade(annual[column]), 'first_5y_mean': complete_annual[column].head(5).mean(), 'last_5y_mean': complete_annual[column].tail(5).mean()})
trend_table = pd.DataFrame(trend_rows).set_index('variable')
print('Trend estimates (linear slope per decade; complete years only):')
print(trend_table.round(3).to_string())
print('Current year shown as YTD in annual table but excluded from slopes and first/last complete-year means:', pd.Timestamp.today().year)
trend_table.to_csv(PROCESSED_DIR / 'karak_weather_trend_estimates_per_decade.csv')
"""),
    md("""### Findings from the trend estimates

The `slope_per_decade` column is the quantitative answer to “what is happening.” Compare it with the first-versus-last five-year means, because a linear slope can hide breaks, missing years, or non-linear variability. The notebook deliberately reports temperature, precipitation, wind, humidity, and rain separately rather than collapsing weather into one unsupported conclusion.
"""),
    code("""fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
plots = [
    ('temperature_2m_mean', 'Annual mean temperature (°C)'),
    ('temperature_2m_max', 'Annual mean daily maximum (°C)'),
    ('precipitation_sum', 'Annual precipitation sum (mm)'),
    ('relative_humidity_2m_mean', 'Annual mean relative humidity (%)'),
]
for ax, (column, title) in zip(axes.flat, plots):
    if column in annual:
        ax.plot(annual.index, annual[column], marker='.', linewidth=1, label='annual')
        clean = annual.loc[annual.index < pd.Timestamp.today().year, column].dropna()
        if len(clean) > 1:
            fit = np.polyval(np.polyfit(clean.index, clean.to_numpy(), 1), clean.index)
            ax.plot(clean.index, fit, '--', label='complete-year linear fit')
        ax.set_title(title)
        ax.grid(alpha=.3)
        ax.legend()
fig.suptitle('Karak modeled weather trends — Open-Meteo daily archive', y=.995)
fig.tight_layout()
fig.savefig(PROCESSED_DIR / 'karak_weather_trends_2000_present.png', dpi=140)
plt.show()
print('Saved trend figure:', PROCESSED_DIR / 'karak_weather_trends_2000_present.png')
"""),
    md("""## 04 decision and interpretation

Use the generated trend table and figure to discuss the Karak weather history. Do not use this file as an AQI target, and do not merge it into the 2022-present hourly AQI dataset unless the sampling/aggregation decision is explicitly documented.
"""),
]


NOTEBOOKS = {
    "01_raw_data_check.ipynb": NB01,
    "02_feature_eda.ipynb": NB02,
    "03_live_open_meteo_check.ipynb": NB03,
    "04_karak_weather_trends.ipynb": NB04,
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(cells)} cells -> {path}")


if __name__ == "__main__":
    main()
