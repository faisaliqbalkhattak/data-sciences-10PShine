# API Documentation — AQI Forecasting Project (Karak, Pakistan)

**Project**: 10Pearls Shine Program — 3-day AQI forecast for Karak, Pakistan  
**Coordinates**: 33.1189° N, 71.0947° E  
**Local timezone**: Asia/Karachi (PKT, UTC+5)  
**Last updated**: 2026-07-29

This file contains the authoritative reference for every external data source used by the pipeline. Process findings and analysis conclusions are documented separately in `Docs/learning.txt`.

---

## 1. Open-Meteo Air Quality API

**Endpoint**  
`https://air-quality-api.open-meteo.com/v1/air-quality`

**What it returns**  
Hourly atmospheric composition and pollutant variables: `pm10`, `pm2_5`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `ammonia`, `dust`, `aerosol_optical_depth`, `uv_index`, `european_aqi`, `us_aqi`, etc.

**Units**  
- Mass concentrations (PM₂.₅, PM₁₀, NO₂, SO₂, CO, O₃): **µg/m³**  
- AQI: unitless index  
- UV index: unitless

**Authentication**  
None required for non-commercial use. Free, rate-limited by IP.

**Timezone handling**  
- Defaults to UTC.  
- Accepts `timezone=auto` or an explicit IANA timezone such as `Asia/Karachi`.  
- Returned timestamps are naive strings in the requested timezone.

**Data source**  
Copernicus Atmosphere Monitoring Service (CAMS) global/European ensemble forecasting and reanalysis systems. Open-Meteo is a wrapper that re-grids CAMS output to the requested lat/lon.

**Historical coverage**  
Several years of historical reanalysis are available via `start_date` and `end_date` parameters (`YYYY-MM-DD`).

**Strengths**  
- No API key needed.  
- Long historical record.  
- Returns a rich set of pollutants and AQI indices.  

**Limitations / caveats**  
- CAMS is a **model/reanalysis** product, not a ground-station measurement.  
- Spatial resolution is coarse (~10–40 km). Karak has complex topography; a single grid cell may miss local valley effects.  
- Some variables (e.g., `ozone`) can hit an upper saturation value (≈200 µg/m³).  
- Correlation with other model or station sources is typically **0.4–0.6** for PM₂.₅ at a rural grid point; this is expected, not necessarily wrong.

---

## 2. Open-Meteo Weather API

**Endpoint**  
`https://archive-api.open-meteo.com/v1/archive`

**What it returns**  
Hourly weather variables: `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `rain`, `surface_pressure`, `cloud_cover`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m`.

**Units**  
- Temperature: °C  
- Humidity: %  
- Precipitation/rain: mm  
- Pressure: hPa  
- Wind speed: km/h  
- Wind direction: degrees

**Authentication**  
None required for non-commercial use.

**Data source**  
ERA5 / IFS reanalysis and forecast blend.

**Use in this project**  
Provides weather features for the AQI forecasting model. Merged with Open-Meteo air-quality data on the hourly index.

---

## 3. OpenWeather Air Pollution API (archieve after validation)

**Endpoints**  
- Current: `http://api.openweathermap.org/data/2.5/air_pollution`  
- Forecast: `http://api.openweathermap.org/data/2.5/air_pollution/forecast`  
- Historical: `http://api.openweathermap.org/data/2.5/air_pollution/history`

**What it returns**  
Standardized AQI (`aqi`, 1–5 scale) and component concentrations: `co`, `no`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`, `nh3`.

**Units**  
- PM₂.₅, PM₁₀, NO₂, SO₂, O₃, NH₃: **µg/m³**  
- CO: **µg/m³** (note: many other sources report CO in mg/m³ or ppm)  
- AQI: 1 (Good) to 5 (Very Poor)

**Authentication**  
Requires `OPENWEATHER_API_KEY`.

**Timezone handling**  
- All timestamps are Unix timestamps in **UTC**.  
- The consumer must convert to the local timezone (`Asia/Karachi`, UTC+5).

**Data source**  
Proprietary blend of global atmospheric models, often derived from CAMS/ECMWF data, fused with weather model outputs.

**Historical coverage**  
Available from 27 Nov 2020 onward. The historical endpoint requires `start` and `end` as Unix timestamps.

**Strengths**  
- Good for cross-validation against Open-Meteo.  
- Provides a single AQI value per hour.

**Limitations / caveats**  
- Also a model product, not a ground measurement.  
- Data gaps can occur (~2.6% of hours in the backfill).  
- CO units differ from typical EPA tables (µg/m³ vs ppm), so CO-based AQI should be avoided unless converted.  
- Hour boundaries may not align perfectly with Open-Meteo; sub-hour phase noise can reduce correlation.

---

## 4. WAQI / AQICN Feed API (archieve after validatoin)

**Endpoint**  
`https://api.waqi.info/feed/{station}/?token={AQICN_TOKEN}`

**What it returns**  
Real-time air-quality readings from a specific monitoring station or city. Pollutants: PM₂.₅ (`pm25`), PM₁₀ (`pm10`), O₃ (`o3`), NO₂ (`no2`), SO₂ (`so2`), CO (`co`), plus an overall AQI.

**Units**  
- Particulates and most gases: **µg/m³**  
- CO: often **ppm** (depends on the local station's reporting)  
- AQI: unitless, usually based on US EPA breakpoints

**Authentication**  
Requires a free `AQICN_TOKEN`.

**Timezone handling**  
Timestamps and update times are reported in the **station's local timezone**, including the UTC offset.

**Rate limits**  
Free tier: ~1,000 requests per minute per token (fair-use; may be lower).

**Use in this project**  
Sanity-check reference only. The nearest station with reliable data is **Peshawar**, not Karak. Karak itself has no ground monitor, so WAQI cannot provide a direct Karak measurement.

**Strengths**  
- Ground-station data (where available).  
- Useful for validating model estimates against reality.

**Limitations / caveats**  
- Nearest station (Peshawar) is ~120 km from Karak and may not represent local Karak conditions.  
- Station coverage in Pakistan is sparse and data quality varies.  
- Different AQI calculation methods may be used depending on the station/country.

---

## 5. Cross-source comparison notes

| Concern | Open-Meteo | OpenWeather | WAQI/AQICN |
|--------|------------|-------------|------------|
| Type | Model/reanalysis | Model/reanalysis | Ground station (where available) |
| PM₂.₅ units | µg/m³ | µg/m³ | µg/m³ |
| CO units | µg/m³ | µg/m³ | often ppm |
| Timestamps | Requested timezone (we use `Asia/Karachi`) | UTC | Station local time |
| Spatial resolution | Coarse grid (~10–40 km) | Coarse grid | Point/station |
| API key | No | Yes | Yes |
| Best use | Primary training source | Cross-validation | Sanity-check reference |

---

## 6. Recommended usage in the pipeline

1. **Primary pollutant source**: Open-Meteo Air Quality (uses CAMS under the hood).  
2. **Weather features**: Open-Meteo Weather Archive (ERA5/IFS).  
3. **Cross-validation**: OpenWeather Air Pollution API.  
4. **Ground-truth sanity check**: WAQI nearest station (Peshawar) — use only as a reference, not as a training target.  
5. **Timezone**: Always normalize every timestamp to `Asia/Karachi` (UTC+5) before merging or correlating.  
6. **Correlation expectations**:  
   - Open-Meteo vs OpenWeather (same grid): expect **0.4–0.6** for hourly PM₂.₅.  
   - Model vs ground station (Peshawar): expect **0.6–0.8** if the model represents the region well.  
   - Lower values do not automatically mean a bug; they may reflect model noise, topography, or local emissions not captured by coarse models.
