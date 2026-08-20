"""Streamlit dashboard for the Karak AQI Predictor -- Google Material / IQAir style.

Architecture: predictions are pre-computed by the CI pipelines (feature pipeline
hourly, training pipeline daily) and stored as ``data/static_forecast.json``.
The dashboard reads this JSON file and renders a static page for every visitor,
giving near-instant response times without runtime inference.

The only real-time fetch is IQAir's live AQI reading for the hero widget,
which is cached for 5 minutes.

Run from ``development``::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import altair as alt  # noqa: E402

from src import config  # noqa: E402
from src.aqi import aqi_category  # noqa: E402

st.set_page_config(
    page_title="Karak AQI Predictor",
    page_icon="AQI",
    layout="wide",
    initial_sidebar_state="expanded",
)

FORECAST_PATH = PROJECT_ROOT / "data" / "static_forecast.json"
IQAIR_PATH = PROJECT_ROOT / "data" / "iqair_forecast.json"
API_URL = os.environ.get("AQI_API_URL", "http://127.0.0.1:8000")

# Semantic palette tailored from the portfolio design: green for environment,
# orange for warning, red for hazards, blue for information.
CATEGORY_COLORS = {
    "Good": "#2e7d32",
    "Moderate": "#9ccc65",
    "Unhealthy for Sensitive Groups": "#f47a32",
    "Unhealthy": "#e26225",
    "Very Unhealthy": "#d93025",
    "Hazardous": "#8f2f12",
}
INK = "#241812"
MUTED = "#5c4a3f"
SURFACE = "#fffaf5"
CANVAS = "#f6eee7"
LINE = "#eadbd0"
ORANGE_700 = "#c84f1b"
KICKER = "#a83c10"
INFO_BLUE = "#4a7dd6"
INFO_BLUE_TEXT = "#1a56c9"
DISPLAY_FONT = "'Space Grotesk',sans-serif"

BAND_BOUNDS = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif;
}
.stApp {
    background: #f6eee7;
    background-image: radial-gradient(circle at 12% 4%, rgba(244, 122, 50, 0.14), transparent 26rem);
}
.block-container { max-width: 1560px; padding-top: 4.2rem; padding-bottom: 4rem; }
.main .block-container { padding-left: 2.5rem; padding-right: 2.5rem; }
h1, h2, h3 {
    color: #241812; font-weight: 700; letter-spacing: -0.02em;
    font-family: 'Space Grotesk', 'DM Sans', sans-serif;
}
section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
}
[data-testid="stDecoration"] { display: none !important; }

/* Top bar */
.topbar {
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    background: #fffaf5; border: 1px solid #eadbd0; border-radius: 16px;
    padding: 12px 18px; margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(91, 44, 18, 0.08);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fffaf5;
    border: 1px solid #eadbd0 !important;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(91, 44, 18, 0.06);
    padding: 6px 18px;
    margin-bottom: 14px;
}
[data-testid="stVerticalBlockBorderWrapper"] > div > div > div > div > div {
    box-shadow: none !important;
    border: none !important;
}

/* Cards */
div[data-testid="stMetric"], div[data-testid="stExpander"] {
    background: #fffaf5; border-radius: 16px; border: 1px solid #eadbd0;
    box-shadow: 0 8px 24px rgba(91, 44, 18, 0.08);
}
div[data-testid="stMetric"] { padding: 16px 18px; }
div[data-testid="stMetricLabel"] p {
    color: #5c4a3f; font-size: 11px; letter-spacing: .4px; text-transform: uppercase;
}
div[data-testid="stMetricValue"] {
    color: #241812; font-size: 26px; font-weight: 700;
    font-family: 'Space Grotesk', 'DM Sans', sans-serif;
}
div[data-testid="stMetricDelta"] { font-size: 12px; font-weight: 600; }
div[data-testid="stExpander"] { margin-top: 12px; }
div[data-testid="stExpander"] summary {
    font-weight: 700; color: #241812; font-family: 'Space Grotesk', 'DM Sans', sans-serif;
}
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
div[data-testid="stCaptionContainer"] p { color: #5c4a3f !important; }

/* Segmented control */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important; flex-direction: row !important; align-items: center; gap: 4px;
    background: #eadbd0; border-radius: 999px; padding: 4px; width: 100%;
}
div[data-testid="stRadio"] label {
    flex: 1 1 0; text-align: center; border-radius: 999px; padding: 7px 6px; margin: 0;
    color: #5c4a3f !important; font-weight: 600; font-size: 13px; white-space: nowrap;
}
div[data-testid="stRadio"] label:hover { background: rgba(255, 255, 255, 0.55); }
div[data-testid="stRadio"] label:has(input:checked) {
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(91, 44, 18, 0.25), 0 0 0 1px #eadbd0;
}
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"],
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
    color: #5c4a3f !important;
}
div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"],
div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color: #c84f1b !important;
}
div[data-testid="stRadio"] label > div:first-child { display: none; }

/* Buttons */
div.stButton > button {
    background: #241812; color: #fffaf5 !important; border: none; border-radius: 999px;
    padding: 7px 24px; font-weight: 600; white-space: nowrap; height: 38px;
    box-shadow: 0 2px 6px rgba(36, 24, 18, 0.25);
}
div.stButton > button div[data-testid="stMarkdownContainer"] p { color: #fffaf5 !important; white-space: nowrap; }
div.stButton > button:hover { background: #3a2a1e; color: #ffffff !important; border: none; }
div.stButton > button:active, div.stButton > button:focus { border: none; outline: none; }

/* Toggle */
div[data-testid="stToggle"] label[data-baseweb="checkbox"] > div:first-child,
div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child {
    border: 2px solid #a83c10 !important;
    border-radius: 6px !important;
}
div[data-testid="stToggle"] label p,
div[data-testid="stToggle"] label div[data-testid="stMarkdownContainer"] p,
div[data-testid="stCheckbox"] label p,
div[data-testid="stCheckbox"] label div[data-testid="stMarkdownContainer"] p {
    color: #241812 !important; font-weight: 500;
}

:focus-visible { outline: 2px solid #c84f1b; outline-offset: 2px; }
hr { border-color: #eadbd0; }
::selection { background: #f6ae76; color: #241812; }

/* Mobile responsive */
@media (max-width: 768px) {
    .block-container { padding-top: 2.8rem !important; padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { padding: 4px 10px !important; }
    div[data-testid="stRadio"] label { font-size: 11px !important; padding: 5px 4px !important; }
    div.stButton > button { height: 32px !important; padding: 4px 14px !important; font-size: 12px !important; }
    div[data-testid="stMetric"] { padding: 10px 12px !important; }
    div[data-testid="stMetricValue"] { font-size: 20px !important; }
    /* Compact header on mobile */
    .main .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
"""

HERO_SKELETON = """
<div style="border-radius:16px; padding:26px 30px; margin-bottom:12px; background:#fffaf5;
     border:1px solid #eadbd0; animation:aqiPulse 1.6s ease-in-out infinite;">
  <div style="height:12px; width:200px; background:#eadbd0; border-radius:6px;"></div>
  <div style="height:54px; width:170px; background:#eadbd0; border-radius:10px; margin-top:18px;"></div>
  <div style="height:14px; width:280px; background:#eadbd0; border-radius:7px; margin-top:14px;"></div>
</div>
<style>@keyframes aqiPulse { 0%, 100% { opacity: 1; } 50% { opacity: .55; } }</style>
"""


def inject_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def category_color(category: str | None) -> str:
    return CATEGORY_COLORS.get(category or "", "#5f6368")


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)


def tint(hex_color: str, alpha: float) -> str:
    r, g, b = _rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def shade(hex_color: str, factor: float = 0.72) -> str:
    r, g, b = _rgb(hex_color)
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


def is_light(hex_color: str) -> bool:
    r, g, b = _rgb(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 140


# --------------------------------------------------------------------------
# Load pre-computed forecast
# --------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _load_forecast() -> dict:
    """Read the pre-computed forecast JSON generated by the CI pipeline."""
    if not FORECAST_PATH.exists():
        return {}
    return json.loads(FORECAST_PATH.read_text(encoding="utf-8"))


def _load_forecast_as_frames(forecast: dict) -> tuple[pd.Timestamp, pd.DataFrame, pd.Series, dict]:
    """Convert the raw forecast dict into the frames the dashboard needs.

    Returns (origin, rows, iqair_series, current_aqi_dict).
    """
    origin = pd.Timestamp(forecast["origin"])
    outputs = forecast["outputs"]
    rows = pd.DataFrame(outputs)
    rows["start_time"] = pd.to_datetime(rows["start_time"])
    rows["end_time"] = pd.to_datetime(rows["end_time"])
    rows["category"] = rows["value"].map(aqi_category)

    # IQAir reference series
    iqair_data = forecast.get("iqair_forecast", [])
    if iqair_data:
        iqair_series = pd.Series(
            [item["aqi"] for item in iqair_data],
            index=pd.to_datetime([item["time"] for item in iqair_data]),
            name="aqi",
        )
    else:
        iqair_series = pd.Series(dtype=float, name="aqi")

    current_aqi = forecast.get("current_aqi", {})
    return origin, rows, iqair_series, current_aqi


@st.cache_data(ttl=300, show_spinner=False)
def _load_iqair_forecast() -> list[dict]:
    """Read the pre-fetched IQAir forecast from the stored JSON file.

    The IQAir fetch workflow (iqair_pipeline.yml) stores data in
    data/iqair_forecast.json so we never scrape at runtime.
    """
    if not IQAIR_PATH.exists():
        return []
    try:
        return json.loads(IQAIR_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _iqair_now_from_json() -> float | None:
    """Get IQAir's current AQI from the pre-fetched JSON file."""
    data = _load_iqair_forecast()
    if data:
        return round(float(data[0]["aqi"]), 1)
    return None


def _iqair_series_from_json() -> pd.Series:
    """Get IQAir's hourly forecast as a Series from the pre-fetched JSON."""
    data = _load_iqair_forecast()
    if not data:
        return pd.Series(dtype=float, name="aqi")
    return pd.Series(
        [item["aqi"] for item in data],
        index=pd.to_datetime([item["time"] for item in data]),
        name="aqi",
    )


def _model_label() -> str:
    forecast = _load_forecast()
    model = forecast.get("model", "aqi-hourly-ridge")
    generated = forecast.get("generated_at", "")
    if generated:
        try:
            dt = pd.Timestamp(generated)
            return f"{model} (updated {dt:%d %b %H:%M})"
        except Exception:
            pass
    return model


# --------------------------------------------------------------------------
# Rendering: Google Material components
# --------------------------------------------------------------------------
POLLUTANT_LABELS = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "ozone": "O\u2083",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO\u2082",
    "sulphur_dioxide": "SO\u2082",
}


def _pollutant_label(key: str | None) -> str:
    return POLLUTANT_LABELS.get(key or "", "PM2.5")


WORRIED_FACE_SVG = (
    '<svg width="46" height="46" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9"/>'
    '<path d="M8.5 14.5c1 1 2.3 1.4 3.5 1.4s2.5-.4 3.5-1.4"/>'
    '<path d="M9 9.6h.01M15 9.6h.01"/>'
    "</svg>"
)


def render_hero(
    source: str,
    rows: pd.DataFrame,
    current_aqi: dict,
    iqair_now: float | None,
    model_label: str,
) -> None:
    """Hero AQI panel per user spec:

    * ``live`` -- primary = IQAir live AQI, secondary = our current-hour AQI
      (calculated from observed data using US EPA formula).
    * ``store`` -- primary = our current-hour AQI, secondary = IQAir current
      value, tertiary = our model's next-hour prediction.
    """
    # Our current-hour AQI from observed data
    our_current = current_aqi.get("aqi")
    our_category = current_aqi.get("category") or "Good"

    # Model's next-hour prediction
    first = rows.iloc[0]
    model_next = float(first["value"])
    model_next_category = first["category"] or "Good"

    if source == "live" and iqair_now is not None:
        # Live tab: IQAir is primary
        badge_aqi = iqair_now
        badge_category = aqi_category(iqair_now) or our_category
        badge_label = "US AQI\u202f\u00b7\u202fIQAir live"
        secondary_line = f"Ours (this hour): {our_current:.0f}" if our_current is not None else ""
        tertiary_line = f"Ours (next hour): {model_next:.0f}"
    else:
        # Store tab: our current-hour is primary
        badge_aqi = our_current if our_current is not None else model_next
        badge_category = our_category if our_current is not None else model_next_category
        badge_label = "US AQI\u202f\u00b7\u202fthis hour"
        secondary_line = f"IQAir: {iqair_now:.0f}" if iqair_now is not None else ""
        tertiary_line = f"Ours (next hour): {model_next:.0f}"

    color = category_color(badge_category)
    text_color = INK if is_light(color) else "#ffffff"
    panel = shade(color, 0.86)

    pollutant = _pollutant_label(current_aqi.get("main_pollutant"))
    concentration = current_aqi.get("concentration")
    concentration_html = (
        f"{concentration:.1f} \u00b5g/m\u00b3" if concentration is not None else "\u2014"
    )

    lines_html = ""
    if secondary_line:
        lines_html += (
            f'<div style="font-size:12px; margin-top:8px; opacity:.92; font-weight:600;">'
            f"{secondary_line}</div>"
        )
    if tertiary_line:
        lines_html += (
            f'<div style="font-size:11px; margin-top:4px; opacity:.80; font-weight:500;">'
            f"{tertiary_line}</div>"
        )

    html = dedent(f"""
    <div style="border-radius:20px; overflow:hidden; margin-bottom:14px;
         box-shadow:0 18px 45px {tint(color, .22)}; border:1px solid {shade(color, .9)}; width:100%;">
      <div style="background:linear-gradient(135deg, {color}, {panel}); padding:24px 28px 20px;
           color:{text_color};">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
          <div style="flex:1;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
              <div style="background:rgba(25,16,10,.82); border-radius:14px; padding:14px 24px; min-width:160px;
                   text-align:center; color:#fff;">
                <div style="font-size:52px; font-weight:700; line-height:1; letter-spacing:-.03em;
                     font-family:'Space Grotesk', sans-serif;">{badge_aqi:.0f}</div>
                <div style="font-size:11px; letter-spacing:.06em; opacity:.85; margin-top:4px; font-weight:600;">{badge_label}</div>
              </div>
              <div style="color:{text_color}; padding-top:4px;">{WORRIED_FACE_SVG}</div>
            </div>
            <div style="font-size:22px; font-weight:700; margin-top:14px; letter-spacing:-.02em;">{badge_category}</div>
            <div style="height:1px; background:rgba(255,255,255,.35); margin:12px 0;"></div>
            <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;
                 font-size:13px; font-weight:600;">
              <span>Main pollutant: {pollutant}</span>
              <span>{concentration_html}</span>
            </div>
            {lines_html}
          </div>
        </div>
      </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


def render_metric_cards(origin: pd.Timestamp, rows: pd.DataFrame, iqair_now: float | None) -> None:
    peak24 = float(rows[rows["kind"] == "point"]["value"].max())
    max72 = float(rows["value"].max())
    tiles = [
        ("Forecast origin", origin.strftime("%m-%d %H:%M"), MUTED, ""),
        ("Peak hourly \u00b7 next 24h", f"{peak24:.0f}", MUTED, ""),
        ("Max \u00b7 full 72h", f"{max72:.0f}", MUTED, ""),
        ("IQAir now", f"{iqair_now:.0f}" if iqair_now is not None else "\u2014", INFO_BLUE_TEXT, "US AQI\u202f\u200a"),
    ]
    cards = []
    for label, value, accent, note in tiles:
        note_html = (
            f'<div style="font-size:12px; color:{accent}; font-weight:600; '
            f'margin-top:3px;">{note}</div>'
            if note
            else '<div style="height:15px;"></div>'
        )
        cards.append(
            '<div style="flex:1 1 0; min-width:160px; background:#fffaf5; '
            'border:1px solid #eadbd0; border-radius:16px; padding:16px 20px; '
            'box-shadow:0 4px 12px rgba(91,44,18,.06);">'
            f'<div style="font-size:11px; letter-spacing:.4px; text-transform:uppercase; '
            f'color:{MUTED}; font-weight:600;">{label}</div>'
            f'<div style="font-size:26px; font-weight:700; color:{INK}; '
            f'font-family:{DISPLAY_FONT}; margin-top:6px;">{value}</div>'
            f"{note_html}</div>"
        )
    st.markdown(
        '<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px;">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_alerts(rows: pd.DataFrame) -> None:
    hazardous = rows[rows["category"] == "Hazardous"]
    very_unhealthy = rows[rows["category"] == "Very Unhealthy"]
    if not hazardous.empty:
        windows = ", ".join(f"{r.start_time:%m-%d %H}h" for r in hazardous.itertuples())
        st.markdown(
            dedent(f"""
            <div style="border-radius:16px; padding:14px 18px; margin-bottom:12px;
                 background:#fdecea; border:1px solid #f5b5b1; color:#8f2f12;">
              <b>HAZARDOUS AQI (\u2265 301) predicted</b> in the next 72 hours at: {windows}.
              Limit outdoor exposure and follow local health advisories.
            </div>
            """),
            unsafe_allow_html=True,
        )
    elif not very_unhealthy.empty:
        windows = ", ".join(f"{r.start_time:%m-%d %H}h" for r in very_unhealthy.itertuples())
        st.markdown(
            dedent(f"""
            <div style="border-radius:16px; padding:14px 18px; margin-bottom:12px;
                 background:#fff0e5; border:1px solid #f6ae76; color:#a83c10;">
              <b>Very Unhealthy AQI (201\u2013300) predicted</b> at: {windows}.
              Sensitive groups should reduce prolonged outdoor activity.
            </div>
            """),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            dedent("""
            <div style="border-radius:16px; padding:12px 18px; margin-bottom:12px;
                 background:#e8f5e9; border:1px solid #c8e6c9; color:#1b5e20;">
              No Very Unhealthy or Hazardous AQI levels predicted in the next 72 hours.
            </div>
            """),
            unsafe_allow_html=True,
        )


def render_hourly_strip(rows: pd.DataFrame) -> None:
    points = rows[rows["kind"] == "point"]
    chips = []
    for hour, row in enumerate(points.itertuples(), start=1):
        category = row.category or "Good"
        color = category_color(category)
        chip_text = shade(color, 0.62)
        chips.append(
            dedent(f"""
            <div style="min-width:62px; border-radius:16px; background:{tint(color, .10)};
                 padding:10px 6px; text-align:center; flex:0 0 auto;">
              <div style="font-size:11px; color:{MUTED}; font-weight:600;">+{hour}h</div>
              <div style="font-size:12px; color:{MUTED};">{row.start_time:%H:%M}</div>
              <div style="font-size:21px; font-weight:600; color:{chip_text};">{row.value:.0f}</div>
              <div style="font-size:10px; color:{chip_text};">&#9679;</div>
            </div>
            """)
        )
    html = (
        '<div style="display:flex; gap:8px; overflow-x:auto; padding:6px 2px 10px;">'
        + "".join(chips)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_block_means(rows: pd.DataFrame) -> None:
    blocks = rows[rows["kind"] != "point"]
    chips = []
    for row in blocks.itertuples():
        category = row.category or "Good"
        color = category_color(category)
        label = row.kind.replace("_", " ")
        chip_text = shade(color, 0.62)
        chips.append(
            dedent(f"""
            <div style="flex:1 1 0; min-width:130px; border-radius:16px;
                 background:{tint(color, .10)}; padding:10px 12px; text-align:center;">
              <div style="font-size:11px; color:{MUTED}; font-weight:600;">{label}</div>
              <div style="font-size:11px; color:{MUTED};">{row.start_time:%d %b %H:%M} \u2192 {row.end_time:%d %b %H:%M}</div>
              <div style="font-size:22px; font-weight:600; color:{chip_text};">{row.value:.0f}</div>
            </div>
            """)
        )
    st.markdown(
        '<div style="display:flex; gap:8px; flex-wrap:wrap; padding:4px 2px 8px;">'
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


IQAIR_GREEN = "#2e7d32"


def render_main_chart(
    origin: pd.Timestamp,
    rows: pd.DataFrame,
    iqair_series: pd.Series,
    view: str,
) -> None:
    points = rows[rows["kind"] == "point"].copy()
    blocks = rows[rows["kind"] != "point"].copy()
    domain_min = points["start_time"].min()
    domain_max = max(blocks["end_time"].max(), points["start_time"].max())
    y_max = max(320.0, float(rows["value"].max()) + 25.0)
    y_scale = alt.Scale(domain=[0, y_max])

    bands = pd.DataFrame(
        [
            {"t0": domain_min, "t1": domain_max, "lo": lo, "hi": hi, "color": category_color(cat)}
            for lo, hi, cat in BAND_BOUNDS
        ]
    )
    layers = [
        alt.Chart(bands)
        .mark_rect(opacity=0.05)
        .encode(
            x=alt.X("t0:T", title=None, axis=alt.Axis(format="%d %b %H:%M", grid=False)),
            x2="t1:T",
            y=alt.Y("lo:Q", title="AQI", scale=y_scale),
            y2="hi:Q",
            color=alt.Color("color:N", scale=None),
        )
    ]

    tooltip = [
        alt.Tooltip("time:T", title="Time", format="%d %b %H:%M"),
        alt.Tooltip("aqi:Q", title="AQI", format=".1f"),
    ]
    if view in ("all", "ours"):
        model_df = points[["start_time", "value"]].rename(
            columns={"start_time": "time", "value": "aqi"}
        )
        model_layer = (
            alt.Chart(model_df)
            .mark_line(point=True, color=ORANGE_700, strokeWidth=2.5)
            .encode(x=alt.X("time:T", title=None), y=alt.Y("aqi:Q", title="AQI", scale=y_scale), tooltip=tooltip)
        )
        block_layer = (
            alt.Chart(blocks)
            .mark_line(strokeWidth=5, color="#8f2f12", opacity=0.8)
            .encode(x="start_time:T", x2="end_time:T", y="value:Q")
        )
        layers.extend([model_layer, block_layer])

    if view in ("all", "iqair") and len(iqair_series):
        try:
            iq_df = iqair_series.reset_index()
            iq_df.columns = ["time", "aqi"]
            iq_layer = (
                alt.Chart(iq_df)
                .mark_line(strokeDash=[4, 3], color=IQAIR_GREEN, strokeWidth=2.2)
                .encode(x=alt.X("time:T", title=None), y=alt.Y("aqi:Q", title="AQI", scale=y_scale), tooltip=tooltip)
            )
            layers.append(iq_layer)
        except Exception:
            pass

    chart = (
        alt.layer(*layers)
        .properties(height=390)
        .configure_axis(labelColor=MUTED, titleColor=INK, gridColor="#eadbd0")
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)
    swatches = (
        f'<div style="font-size:12px; color:{MUTED}; display:flex; gap:18px; padding:2px 2px 8px; flex-wrap:wrap;">'
        f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="display:inline-block;width:18px;height:3px;background:{ORANGE_700};border-radius:2px;"></span> Our model (Ridge)</span>'
        f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="display:inline-block;width:18px;height:3px;background:{IQAIR_GREEN};border-radius:2px;border-top:2px dashed {IQAIR_GREEN};"></span> IQAir hourly forecast (US AQI\u202f\u200a)</span>'
        f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="display:inline-block;width:18px;height:6px;background:#8f2f12;border-radius:2px;"></span> Six/twelve-hour means (our model)</span>'
        "</div>"
    )
    st.markdown(swatches, unsafe_allow_html=True)


def comparison_frame(rows: pd.DataFrame, iqair_series: pd.Series) -> pd.DataFrame:
    """Align our 30 outputs with IQAir on the same window."""
    records = []
    for row in rows.itertuples():
        window = (
            f"{row.start_time:%m-%d %H:%M}"
            if row.kind == "point"
            else f"{row.start_time:%m-%d %H:%M} \u2192 {row.end_time:%m-%d %H:%M}"
        )
        if row.kind == "point":
            iq_value = iqair_series.get(row.start_time, np.nan) if len(iqair_series) else np.nan
        else:
            try:
                idx = pd.DatetimeIndex(iqair_series.index)
                mask_iq = (idx >= row.start_time) & (idx <= row.end_time)
                iq_block = iqair_series[mask_iq]
                iq_value = float(iq_block.mean()) if len(iq_block) else np.nan
            except Exception:
                iq_value = np.nan
        records.append(
            {
                "window": window,
                "kind": row.kind.replace("_", " "),
                "ours": float(row.value),
                "iqair": iq_value,
            }
        )
    frame = pd.DataFrame(records)
    frame["diff_iqair"] = frame["ours"] - frame["iqair"]
    return frame


def render_comparison(rows: pd.DataFrame, iqair_series: pd.Series) -> None:
    section_header("Comparison", "Our model vs IQAir")
    st.markdown(
        f'<div style="font-size:13px; color:{INFO_BLUE_TEXT}; margin-bottom:8px;">'
        "IQAir publishes its own hourly forecast for Karak labelled \"US AQI\u202f\u200a\" -- "
        "the same US EPA AQI scale (categories, colors, breakpoints) this project's "
        "target uses, so the two are directly comparable. Mapped onto our exact 30 "
        "outputs with the same block-mean logic; diff = ours \u2212 IQAir.</div>",
        unsafe_allow_html=True,
    )
    frame = comparison_frame(rows, iqair_series)
    if frame.empty or frame["iqair"].isna().all():
        st.caption("IQAir reference unavailable right now (the site rate-limits anonymous reads).")
        return
    display = frame.copy()
    for col in ("ours", "iqair", "diff_iqair"):
        display[col] = display[col].round(1)
    display = display.rename(
        columns={
            "window": "Valid time",
            "kind": "Output",
            "ours": "Our model",
            "iqair": "IQAir (US AQI\u202f\u200a)",
            "diff_iqair": "\u0394 vs IQAir",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_output_table(rows: pd.DataFrame) -> None:
    section_header("Full detail", "30-output forecast table")
    table = rows.copy()
    table["window"] = table.apply(
        lambda r: f"{r.start_time:%m-%d %H:%M}"
        if r["kind"] == "point"
        else f"{r.start_time:%m-%d %H:%M} \u2192 {r.end_time:%m-%d %H:%M}",
        axis=1,
    )
    display = table[["window", "kind", "value", "category"]].rename(
        columns={
            "window": "Valid time",
            "kind": "Output kind",
            "value": "AQI",
            "category": "EPA category",
        }
    )
    display["AQI"] = display["AQI"].round(1)
    display["EPA category"] = display["EPA category"].fillna("Unknown")
    st.dataframe(display, use_container_width=True, hide_index=True)


def _load_model_eval() -> dict:
    """Read the pre-computed model evaluation JSON generated by the training pipeline."""
    eval_path = PROJECT_ROOT / "data" / "model_eval.json"
    if not eval_path.exists():
        return {}
    try:
        return json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_model_history() -> None:
    eval_data = _load_model_eval()
    if not eval_data:
        st.info("Model evaluation data not available. Wait for the training pipeline to run.")
        return

    # Registry
    registry = eval_data.get("registry", [])
    if registry:
        st.subheader("Model registry (MLflow)")
        st.dataframe(pd.DataFrame(registry), use_container_width=True, hide_index=True)
    else:
        st.caption("Registry unavailable.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly holdout (72h purge)")
        hourly = eval_data.get("hourly_holdout", [])
        if hourly:
            grouped = pd.DataFrame(hourly)
            fig, ax = plt.subplots(figsize=(8, 4))
            for model in grouped["model"].unique():
                subset = grouped[grouped["model"] == model]
                ax.plot(subset["group"], subset["rmse"], marker="o", label=model)
            ax.set_ylabel("RMSE (lower is better)")
            ax.set_xlabel("Output group")
            ax.set_title("Hourly holdout RMSE by model")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            st.dataframe(grouped, use_container_width=True, hide_index=True)
        else:
            st.info("No hourly holdout data available.")

    with col2:
        st.subheader("Daily holdout (+1/+2/+3 days)")
        daily = eval_data.get("daily_holdout", {})
        if daily and "rows" in daily:
            pivot = pd.DataFrame(daily["rows"]).set_index("model")
            st.dataframe(pivot, use_container_width=True)
        else:
            st.info("No daily holdout data available.")

    rolling = eval_data.get("rolling_origin", [])
    if rolling:
        st.subheader("Rolling-origin evaluation (3 expanding folds, 72h embargo)")
        st.dataframe(pd.DataFrame(rolling), use_container_width=True, hide_index=True)


def render_eda() -> None:
    eval_data = _load_model_eval()
    if not eval_data:
        st.info("EDA data not available. Wait for the training pipeline to run.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly rolling AQI \u2014 last 90 days")
        hourly_ts = eval_data.get("eda_hourly_ts", [])
        if hourly_ts:
            df = pd.DataFrame(hourly_ts)
            df["time"] = pd.to_datetime(df["time"])
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(df["time"], df["aqi"], lw=0.7, color="#1a73e8")
            ax.set_ylabel("AQI")
            ax.grid(alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No hourly EDA data available.")

    with col2:
        st.subheader("Daily EPA AQI \u2014 last 2 years")
        daily_ts = eval_data.get("eda_daily_ts", [])
        if daily_ts:
            df = pd.DataFrame(daily_ts)
            df["time"] = pd.to_datetime(df["time"])
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(df["time"], df["aqi"], lw=0.9, color="#d93025")
            ax.set_ylabel("AQI (US EPA)")
            ax.grid(alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No daily EDA data available.")

    st.subheader("Observed AQI category distribution (hourly)")
    hourly_dist = eval_data.get("eda_hourly_dist", [])
    if hourly_dist:
        df = pd.DataFrame(hourly_dist)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        colors = [category_color(cat) for cat in df["category"]]
        ax.bar(df["category"], df["hours"], color=colors)
        ax.set_ylabel("Hours")
        ax.tick_params(axis="x", rotation=15)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)


def render_shap() -> None:
    """Render SHAP explanations from pre-computed JSON."""
    eval_data = _load_model_eval()
    shap_data = eval_data.get("shap")
    if not shap_data:
        st.info("SHAP explanations not available. Wait for the training pipeline to run.")
        return

    st.markdown(
        f'<div style="font-size:13px; color:{MUTED}; margin-bottom:8px;">'
        f'Method: {shap_data.get("method", "unknown")} | '
        f'Output: {shap_data.get("output_column", "t+1h")} | '
        f'Expected value: {shap_data.get("expected_value", 0):.1f} | '
        f'Model prediction: {shap_data.get("prediction_base_plus_shap", 0):.1f}</div>',
        unsafe_allow_html=True,
    )

    features = shap_data.get("features", [])
    if not features:
        st.caption("No SHAP features available.")
        return

    # Build a horizontal bar chart of SHAP values
    df = pd.DataFrame(features)
    df["color"] = df["shap"].apply(lambda v: ORANGE_700 if v > 0 else INFO_BLUE)
    df = df.sort_values("shap", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    ax.barh(df["feature"], df["shap"], color=df["color"], height=0.6)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on AQI)")
    ax.set_title("Feature contributions to the next-hour prediction")
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    # Feature values table
    table = pd.DataFrame(features)[["feature", "value", "shap"]].copy()
    table["value"] = table["value"].round(3)
    table["shap"] = table["shap"].round(2)
    table = table.rename(columns={"feature": "Feature", "value": "Observed value", "shap": "SHAP contribution"})
    st.dataframe(table, use_container_width=True, hide_index=True)


def section_header(kicker: str, title: str) -> None:
    st.markdown(
        f'<div style="font-size:11px; letter-spacing:.18em; text-transform:uppercase; '
        f'color:{KICKER}; font-weight:700; margin-top:28px;">{kicker}</div>'
        f"<div style=\"font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700; "
        f"color:#241812; letter-spacing:-.03em; margin:3px 0 12px;\">{title}</div>",
        unsafe_allow_html=True,
    )


def render_topbar() -> dict:
    with st.container(border=True):
        col_brand, col_source, col_refresh, col_model = st.columns(
            [1.5, 1.6, 1.0, 2.5], vertical_alignment="center", gap="small"
        )
        with col_brand:
            st.markdown(
                f"<div style=\"font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700; "
                f"letter-spacing:-.04em; background:linear-gradient(120deg,#8f2f12,#f47a32); "
                f"-webkit-background-clip:text; background-clip:text; color:transparent;\">"
                f"Karak AQI</div>"
                f"<div style=\"font-size:11px; color:{MUTED};\">{config.CITY_NAME} \u00b7 {config.LOCATION_LABEL}</div>",
                unsafe_allow_html=True,
            )
        with col_source:
            source = st.radio(
                "Data source",
                options=["store", "live"],
                format_func=lambda v: "Store" if v == "store" else "Live",
                index=0,
                horizontal=True,
                label_visibility="collapsed",
            )
        with col_refresh:
            if st.button("Refresh"):
                st.cache_data.clear()
                st.rerun()
        with col_model:
            st.markdown(
                f'<span style="display:inline-block; background:#e8f0fe; color:#1a56c9; '
                f'border:1px solid #c5d7f2; border-radius:999px; padding:3px 12px; '
                f'font-size:12px; font-weight:600;">{_model_label()}</span>',
                unsafe_allow_html=True,
            )
    return {"source": source}


def main() -> None:
    inject_css()
    options = render_topbar()
    source = options["source"]
    model_label = _model_label()

    # Load pre-computed forecast
    forecast = _load_forecast()
    if not forecast:
        st.error(
            "No pre-computed forecast found. Run `python -m src.export_forecast` "
            "or wait for the CI pipeline to generate one."
        )
        return

    origin, rows, _, current_aqi = _load_forecast_as_frames(forecast)

    # IQAir data: read from pre-fetched JSON (zero runtime fetches)
    iqair_series = _iqair_series_from_json()
    iqair_now = _iqair_now_from_json()
    # Fallback to the value stored in the forecast JSON
    if iqair_now is None:
        iqair_now = forecast.get("iqair_now")

    # Re-anchor IQAir series to the forecast origin so both lines
    # start at the same time on the chart.
    if len(iqair_series) and origin is not None:
        iqair_series = pd.Series(
            iqair_series.values,
            index=pd.date_range(origin, periods=len(iqair_series), freq="h"),
            name="aqi",
        )

    # Location + meta on the left; AQI hero card on the right.
    left_col, right_col = st.columns([1.0, 0.9], gap="medium")
    with left_col:
        st.markdown(
            "<div style=\"font-family:'Space Grotesk',sans-serif; font-size:30px; font-weight:700; "
            "color:#241812; letter-spacing:-.03em; margin:6px 0 2px;\">Air quality in Karak</div>"
            f'<div style="font-size:13px; color:{MUTED};">Air quality index (AQI) and PM2.5 air pollution '
            f'in Karak \u00b7 As of {origin:%d %b %Y, %H:00} \u00b7 Asia/Karachi</div>'
            f'<div style="font-size:12px; color:{MUTED}; margin-top:10px;">'
            f"Forecast origin \u00b7 {model_label}</div>",
            unsafe_allow_html=True,
        )
    with right_col:
        hero_slot = st.empty()
        hero_slot.markdown(HERO_SKELETON, unsafe_allow_html=True)

    with hero_slot.container():
        render_hero(source, rows, current_aqi, iqair_now, model_label)

    render_metric_cards(origin, rows, iqair_now)
    render_alerts(rows)

    section_header("Hourly forecast", "Next 24 hours, hour by hour")
    render_hourly_strip(rows)

    view = st.radio(
        "Compare",
        options=["all", "ours", "iqair"],
        format_func=lambda v: {
            "all": "All sources",
            "ours": "Our model",
            "iqair": "IQAir (US AQI\u202f\u200a)",
        }[v],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    render_main_chart(origin, rows, iqair_series, view)

    section_header("Extended forecast", "Beyond 24 hours \u2014 six- and twelve-hour means")
    render_block_means(rows)

    render_comparison(rows, iqair_series)
    render_output_table(rows)

    with st.expander("Model comparison & evaluation"):
        render_model_history()

    with st.expander("SHAP explanations of the latest prediction"):
        render_shap()

    with st.expander("History / EDA"):
        render_eda()

    st.divider()
    generated = forecast.get("generated_at", "")
    generated_str = (
        pd.Timestamp(generated).strftime("%d %b %Y, %H:%M") if generated else "\u2014"
    )
    status_html = (
        '<div style="display:flex; gap:24px; flex-wrap:wrap; font-size:12px; color:'
        + MUTED
        + '; padding:8px 0;">'
        "<span>Forecast generated: <b>"
        + generated_str
        + "</b></span>"
        "<span>Model: <b>"
        + forecast.get("model", "aqi-hourly-ridge")
        + "</b></span>"
        "<span>Reference: IQAir (US AQI\u202f\u200a)</span>"
        "</div>"
        '<div style="font-size:11px; color:'
        + MUTED
        + '; margin-top:4px;">'
        "Pre-computed forecasts served statically. "
        "Auto-updates via GitHub Actions: feature pipeline (hourly) + training pipeline (daily 01:15 UTC)."
        "</div>"
    )
    st.markdown(status_html, unsafe_allow_html=True)

    # Debug panel: shows data state for troubleshooting
    with st.expander("Debug info", expanded=False):
        st.json({
            "forecast_file": str(FORECAST_PATH),
            "forecast_exists": FORECAST_PATH.exists(),
            "iqair_file": str(IQAIR_PATH),
            "iqair_file_exists": IQAIR_PATH.exists(),
            "generated_at": forecast.get("generated_at"),
            "source": forecast.get("source"),
            "model": forecast.get("model"),
            "outputs_count": len(forecast.get("outputs", [])),
            "iqair_from_json_count": len(_load_iqair_forecast()),
            "iqair_from_forecast_count": len(forecast.get("iqair_forecast", [])),
            "iqair_now": iqair_now,
            "current_aqi": current_aqi,
            "origin": str(origin),
            "rows_count": len(rows),
            "iqair_series_len": len(iqair_series),
        })


if __name__ == "__main__":
    main()
