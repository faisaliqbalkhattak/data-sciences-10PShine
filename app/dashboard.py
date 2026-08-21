"""Streamlit dashboard for the Karak AQI Predictor -- Google Material / IQAir style.

Architecture: predictions are pre-computed by the CI pipelines (feature pipeline
hourly, training pipeline daily) and stored in the karAQI-data repo as static
JSON files.  The dashboard fetches these via GitHub raw URLs, giving every
visitor a near-instant page load with zero runtime inference.

Run from ``development``::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import io
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
import requests as _requests  # noqa: E402

from src import config  # noqa: E402
from src.aqi import aqi_category  # noqa: E402

st.set_page_config(
    page_title="Karak AQI Predictor",
    page_icon="AQI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data source: karAQI-data repo (static JSON files)
# ---------------------------------------------------------------------------
DATA_REPO = os.environ.get(
    "AQI_DATA_REPO", "https://raw.githubusercontent.com/faisaliqbalkhattak/karAQI-data/main/data"
)
FORECAST_URL = f"{DATA_REPO}/static_forecast.json"
MODEL_EVAL_URL = f"{DATA_REPO}/model_eval.json"
IMAGES_REPO = "https://raw.githubusercontent.com/faisaliqbalkhattak/karAQI-data/main/images"

# Also support local fallback for development
FORECAST_PATH = PROJECT_ROOT / "data" / "static_forecast.json"

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
DISPLAY_FONT = "'Poppins', 'Inter', sans-serif"

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}
.stApp {
    background: #f6eee7;
    background-image: radial-gradient(circle at 12% 4%, rgba(244, 122, 50, 0.14), transparent 26rem);
}
.block-container { max-width: 1560px; padding-top: 1rem; padding-bottom: 2rem; }
.main .block-container { padding-left: 2.5rem; padding-right: 2.5rem; }
h1, h2, h3 {
    color: #241812; font-weight: 700; letter-spacing: -0.02em;
    font-family: 'Poppins', 'Inter', sans-serif;
}
section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"], #MainMenu { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

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
    font-family: 'Poppins', 'Inter', sans-serif;
}
div[data-testid="stMetricDelta"] { font-size: 12px; font-weight: 600; }
div[data-testid="stExpander"] { margin-top: 12px; }
div[data-testid="stExpander"] summary {
    font-weight: 700; color: #241812; font-family: 'Poppins', 'Inter', sans-serif;
}
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
div[data-testid="stCaptionContainer"] p { color: #5c4a3f !important; }

/* Segmented control */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important; flex-direction: row !important; align-items: center; gap: 4px;
    padding: 0; width: 100%;
}
div[data-testid="stRadio"] label {
    flex: 1 1 0; text-align: center; border-radius: 999px; padding: 7px 6px; margin: 0;
    color: #5c4a3f !important; font-weight: 600; font-size: 13px; white-space: nowrap;
}
div[data-testid="stRadio"] label:hover { background: rgba(234, 219, 208, 0.5); }
div[data-testid="stRadio"] label:has(input:checked) {
    background: #eadbd0 !important;
    box-shadow: 0 1px 3px rgba(91, 44, 18, 0.15);
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
# Data loading: fetch from karAQI-data repo (with local fallback)
# --------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_json_remote(url: str) -> dict | list | None:
    """Fetch JSON from a URL with a 10s timeout."""
    try:
        resp = _requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _load_forecast() -> dict:
    """Read the pre-computed forecast — try remote first, then local fallback."""
    data = _fetch_json_remote(FORECAST_URL)
    if data is not None:
        return data if isinstance(data, dict) else {}
    # Local fallback (development)
    if FORECAST_PATH.exists():
        try:
            return json.loads(FORECAST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# IQAir-specific functions removed — reference data now lives inside
# static_forecast.json under 'ref_forecast' / 'ref_now' keys.
# The reference source is Open-Meteo AQ forecast (free, keyless, same US AQI scale).


def _load_model_eval() -> dict:
    """Read the pre-computed model evaluation from remote or local."""
    data = _fetch_json_remote(MODEL_EVAL_URL)
    if data is not None:
        return data if isinstance(data, dict) else {}
    eval_path = PROJECT_ROOT / "data" / "model_eval.json"
    if eval_path.exists():
        try:
            return json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_forecast_as_frames(forecast: dict) -> tuple[pd.Timestamp, pd.DataFrame, pd.Series, dict]:
    """Convert the raw forecast dict into the frames the dashboard needs."""
    origin = pd.Timestamp(forecast["origin"])
    outputs = forecast["outputs"]
    rows = pd.DataFrame(outputs)
    rows["start_time"] = pd.to_datetime(rows["start_time"])
    rows["end_time"] = pd.to_datetime(rows["end_time"])
    rows["category"] = rows["value"].map(aqi_category)

    ref_data = forecast.get("ref_forecast", [])
    if ref_data:
        ref_series = pd.Series(
            [item["aqi"] for item in ref_data],
            index=pd.to_datetime([item["time"] for item in ref_data]),
            name="aqi",
        )
    else:
        ref_series = pd.Series(dtype=float, name="aqi")

    current_aqi = forecast.get("current_aqi", {})
    return origin, rows, ref_series, current_aqi


def _ref_now_from_forecast() -> float | None:
    """Get the reference AQI now value from the forecast JSON."""
    forecast = _load_forecast()
    return forecast.get("ref_now")


def _ref_series_from_forecast() -> pd.Series:
    """Get the reference AQI series from the forecast JSON."""
    forecast = _load_forecast()
    ref_data = forecast.get("ref_forecast", [])
    if not ref_data:
        return pd.Series(dtype=float, name="aqi")
    return pd.Series(
        [item["aqi"] for item in ref_data],
        index=pd.to_datetime([item["time"] for item in ref_data]),
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
    ref_now: float | None,
    model_label: str,
) -> None:
    """Hero AQI panel per user spec:

    * ``live`` -- primary = reference AQI (Open-Meteo), secondary = our current-hour AQI
    * ``store`` -- primary = our current-hour AQI, secondary = reference current,
      tertiary = our model's next-hour prediction.
    """
    our_current = current_aqi.get("aqi")
    our_category = current_aqi.get("category") or "Good"

    first = rows.iloc[0]
    model_next = float(first["value"])
    model_next_category = first["category"] or "Good"

    if source == "live" and ref_now is not None:
        badge_aqi = ref_now
        badge_category = aqi_category(ref_now) or our_category
        badge_label = "US AQI\u202f\u00b7\u202fLive from Open-Meteo"
        secondary_line = f"Ours (this hour): {our_current:.0f}" if our_current is not None else ""
        tertiary_line = f"Ours (next hour): {model_next:.0f}"
    else:
        badge_aqi = our_current if our_current is not None else model_next
        badge_category = our_category if our_current is not None else model_next_category
        badge_label = "US AQI\u202f\u00b7\u202fthis hour"
        secondary_line = f"Live (Open-Meteo): {ref_now:.0f}" if ref_now is not None else ""
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
            f'<div style="font-size:13px; margin-top:8px; opacity:.92; font-weight:600;">'
            f"{secondary_line}</div>"
        )
    if tertiary_line:
        lines_html += (
            f'<div style="font-size:12px; margin-top:4px; opacity:.80; font-weight:500;">'
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
                     font-family:'Poppins', sans-serif;">{badge_aqi:.0f}</div>
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


def render_metric_cards(origin: pd.Timestamp, rows: pd.DataFrame, ref_now: float | None) -> None:
    peak24 = float(rows[rows["kind"] == "point"]["value"].max())
    max72 = float(rows["value"].max())
    tiles = [
        ("Forecast origin", origin.strftime("%m-%d %H:%M"), MUTED, ""),
        ("Peak hourly \u00b7 next 24h", f"{peak24:.0f}", MUTED, ""),
        ("Max \u00b7 full 72h", f"{max72:.0f}", MUTED, ""),
        ("Live from Open-Meteo", f"{ref_now:.0f}" if ref_now is not None else "\u2014", INFO_BLUE_TEXT, "US AQI\u202f\u200a"),
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


def render_prediction_bar_chart(rows: pd.DataFrame) -> None:
    """Render all 30 forecast outputs as a colored bar chart (AQI category colors)."""
    points = rows[rows["kind"] == "point"].copy()
    blocks = rows[rows["kind"] != "point"].copy()

    # Build bar data from points first, then block means
    bars = []
    for _, r in points.iterrows():
        bars.append({
            "label": pd.Timestamp(r["start_time"]).strftime("%d %b %H:%M"),
            "aqi": float(r["value"]),
            "color": category_color(r.get("category") or aqi_category(r["value"])),
        })
    for _, r in blocks.iterrows():
        bars.append({
            "label": f"{pd.Timestamp(r['start_time']):%d %b %H:%M}→{pd.Timestamp(r['end_time']):%H:%M}",
            "aqi": float(r["value"]),
            "color": category_color(r.get("category") or aqi_category(r["value"])),
        })
    bar_df = pd.DataFrame(bars)

    chart = (
        alt.Chart(bar_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=14)
        .encode(
            x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=-45, labelFontSize=9, labelColor=MUTED)),
            y=alt.Y("aqi:Q", title="US AQI", axis=alt.Axis(gridColor="#eadbd0", labelColor=MUTED, titleColor=INK)),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("label:N", title="Time"),
                alt.Tooltip("aqi:Q", title="AQI", format=".0f"),
            ],
        )
        .properties(height=320)
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)

    # Category legend
    legend = (
        '<div style="font-size:11px; color:' + MUTED + '; display:flex; gap:12px; flex-wrap:wrap; padding:2px 0 8px;">'
    )
    for low, high, cat in BAND_BOUNDS:
        c = category_color(cat)
        legend += f'<span><span style="display:inline-block;width:10px;height:10px;background:{c};border-radius:2px;margin-right:3px;vertical-align:middle;"></span>{cat}</span>'
    legend += "</div>"
    st.markdown(legend, unsafe_allow_html=True)


def render_main_chart(
    origin: pd.Timestamp,
    rows: pd.DataFrame,
    ref_series: pd.Series,
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

    if view in ("all", "ref") and len(ref_series):
        try:
            ref_df = ref_series.reset_index()
            ref_df.columns = ["time", "aqi"]
            ref_layer = (
                alt.Chart(ref_df)
                .mark_line(strokeDash=[4, 3], color=IQAIR_GREEN, strokeWidth=2.2)
                .encode(x=alt.X("time:T", title=None), y=alt.Y("aqi:Q", title="AQI", scale=y_scale), tooltip=tooltip)
            )
            layers.append(ref_layer)
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
        f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="display:inline-block;width:18px;height:3px;background:{IQAIR_GREEN};border-radius:2px;border-top:2px dashed {IQAIR_GREEN};"></span> Live from Open-Meteo (US AQI\u202f\u200a)</span>'
        f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="display:inline-block;width:18px;height:6px;background:#8f2f12;border-radius:2px;"></span> Six/twelve-hour means (our model)</span>'
        "</div>"
    )
    st.markdown(swatches, unsafe_allow_html=True)


def comparison_frame(rows: pd.DataFrame, ref_series: pd.Series) -> pd.DataFrame:
    """Align our 30 outputs with the reference on the same window."""
    records = []
    for row in rows.itertuples():
        window = (
            f"{row.start_time:%m-%d %H:%M}"
            if row.kind == "point"
            else f"{row.start_time:%m-%d %H:%M} \u2192 {row.end_time:%m-%d %H:%M}"
        )
        if row.kind == "point":
            ref_value = ref_series.get(row.start_time, np.nan) if len(ref_series) else np.nan
        else:
            try:
                idx = pd.DatetimeIndex(ref_series.index)
                mask_ref = (idx >= row.start_time) & (idx <= row.end_time)
                ref_block = ref_series[mask_ref]
                ref_value = float(ref_block.mean()) if len(ref_block) else np.nan
            except Exception:
                ref_value = np.nan
        records.append(
            {
                "window": window,
                "kind": row.kind.replace("_", " "),
                "ours": float(row.value),
                "reference": ref_value,
            }
        )
    frame = pd.DataFrame(records)
    frame["diff_ref"] = frame["ours"] - frame["reference"]
    return frame


def render_comparison(rows: pd.DataFrame, ref_series: pd.Series) -> None:
    section_header("Comparison", "Our model vs live from Open-Meteo")
    st.markdown(
        f'<div style="font-size:13px; color:{INFO_BLUE_TEXT}; margin-bottom:8px;">'
        "A free, keyless hourly US AQI forecast for Karak -- "
        "the same US EPA AQI scale (categories, colors, breakpoints) this project's "
        "target uses, so the two are directly comparable. Mapped onto our exact 30 "
        "outputs with the same block-mean logic; diff = ours \u2212 reference.</div>",
        unsafe_allow_html=True,
    )
    frame = comparison_frame(rows, ref_series)
    if frame.empty or frame["reference"].isna().all():
        st.caption("Reference data unavailable. The forecast API may be temporarily unreachable.")
        return
    display = frame.copy()
    for col in ("ours", "reference", "diff_ref"):
        display[col] = display[col].round(1)
    display = display.rename(
        columns={
            "window": "Valid time",
            "kind": "Output",
            "ours": "Our model",
            "reference": "Live from Open-Meteo (US AQI\u202f\u200a)",
            "diff_ref": "\u0394 vs ref",
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


def render_model_history() -> None:
    eval_data = _load_model_eval()
    if not eval_data:
        st.info("Model evaluation data not available. Wait for the training pipeline to run.")
        return

    registry = eval_data.get("registry", [])
    if registry:
        st.subheader("Model registry (MLflow)")
        st.markdown(
            f'<div style="font-size:13px; color:{MUTED}; margin-bottom:10px; line-height:1.6;">'
            "MLflow tracks every training run and registers the best-performing model for each horizon. "
            "Each row is a registered model: the name identifies the target (e.g. <b>aqi-hourly-ridge</b> = our hourly Ridge model), "
            "the version is the training iteration, and the alias (e.g. <b>champion</b>) marks which version is currently served. "
            "When the daily training pipeline runs, it re-evaluates all models and promotes the winner to champion.</div>",
            unsafe_allow_html=True,
        )
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
        st.markdown(
            f'<div style="font-size:13px; color:{MUTED}; margin-bottom:10px; line-height:1.6;">'
            "A more realistic evaluation than a single train/test split. The model is trained on expanding windows "
            "of historical data and tested on the next 72 hours, then the window rolls forward. "
            "The 72-hour embargo prevents data leakage (no test data overlaps with training lag features). "
            "<b>RMSE</b> = root mean squared error (lower is better). "
            "<b>Category accuracy</b> = % of predictions in the correct EPA AQI band. "
            "<b>High AQI recall</b> = % of truly polluted hours the model correctly flags.</div>",
            unsafe_allow_html=True,
        )
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

    table = pd.DataFrame(features)[["feature", "value", "shap"]].copy()
    table["value"] = table["value"].round(3)
    table["shap"] = table["shap"].round(2)
    table = table.rename(columns={"feature": "Feature", "value": "Observed value", "shap": "SHAP contribution"})
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_weather_insights() -> None:
    """Display weather trend and seasonality images from the repo."""
    section_header("Weather insights", "Karak weather trends and AQI seasonality")

    st.markdown(
        f'<div style="font-size:14px; color:{MUTED}; margin-bottom:12px; line-height:1.6;">'
        "Long-term weather patterns in Karak influence air quality. These charts show "
        "temperature and precipitation trends from 2000 to present, and how AQI varies "
        "by season and time of day. Dust events (common in March\u2013June) and winter "
        "inversions (December\u2013February) are the primary drivers of poor air quality.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:12px; color:{MUTED}; margin-bottom:14px; line-height:1.5; '
        f'padding:8px 12px; background:#f6eee7; border-left:3px solid {KICKER}; border-radius:4px;">'
        "<b>How these charts were made:</b> The weather trends chart uses historical weather data "
        "(2000\u20132026) fetched from the Open-Meteo Archive API for Karak (33.13°N, 71.54°E) — "
        "temperature, precipitation, wind, and humidity only, no AQI. The seasonality chart was built "
        "by merging Open-Meteo hourly PM2.5, PM10, and ozone observations with weather variables, then "
        "computing US EPA AQI sub-indices from raw pollutant concentrations using the standard breakpoint "
        "tables (pollutant-specific averaging windows and unit conversions). Both charts were generated "
        "with matplotlib and stored in the karAQI-data repository.</div>",
        unsafe_allow_html=True,
    )

    img_urls = {
        "weather_trends": f"{IMAGES_REPO}/karak_weather_trends_2000_present.png",
        "seasonality": f"{IMAGES_REPO}/karak_aqi_open_meteo_seasonality.png",
    }
    # Fallback to local files
    img_paths = {
        "weather_trends": PROJECT_ROOT / "data" / "processed" / "karak_weather_trends_2000_present.png",
        "seasonality": PROJECT_ROOT / "data" / "processed" / "karak_aqi_open_meteo_seasonality.png",
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temperature & precipitation trends")
        st.markdown(
            f'<div style="font-size:13px; color:{MUTED}; margin-bottom:8px;">'
            "Karak\u2019s climate shows hot summers (40\u2009\u00b0C+) and mild winters. "
            "Precipitation is concentrated in the monsoon (July\u2013September). "
            "Dust storms during the dry pre-monsoon period (March\u2013June) "
            "correlate with PM10 spikes.</div>",
            unsafe_allow_html=True,
        )
        img_bytes = None
        try:
            resp = _requests.get(img_urls["weather_trends"], timeout=10)
            if resp.ok:
                img_bytes = io.BytesIO(resp.content)
        except Exception:
            pass
        if img_bytes is None and img_paths["weather_trends"].exists():
            img_bytes = io.BytesIO(img_paths["weather_trends"].read_bytes())
        if img_bytes is not None:
            st.image(img_bytes, use_column_width=True)
        else:
            st.info("Weather trends image not available yet.")

    with col2:
        st.subheader("AQI seasonality pattern")
        st.markdown(
            f'<div style="font-size:13px; color:{MUTED}; margin-bottom:8px;">'
            "Monthly and hourly AQI patterns reveal when air quality is worst. "
            "Summer months (May\u2013August) show higher PM2.5 due to dust and biomass burning. "
            "Nighttime inversions trap pollutants, leading to AQI peaks between 6\u201310 AM.</div>",
            unsafe_allow_html=True,
        )
        img_bytes = None
        try:
            resp = _requests.get(img_urls["seasonality"], timeout=10)
            if resp.ok:
                img_bytes = io.BytesIO(resp.content)
        except Exception:
            pass
        if img_bytes is None and img_paths["seasonality"].exists():
            img_bytes = io.BytesIO(img_paths["seasonality"].read_bytes())
        if img_bytes is not None:
            st.image(img_bytes, use_column_width=True)
        else:
            st.info("Seasonality image not available yet.")


def section_header(kicker: str, title: str) -> None:
    st.markdown(
        f'<div style="font-size:11px; letter-spacing:.18em; text-transform:uppercase; '
        f'color:{KICKER}; font-weight:700; margin-top:28px;">{kicker}</div>'
        f"<div style=\"font-family:'Poppins',sans-serif; font-size:20px; font-weight:700; "
        f"color:#241812; letter-spacing:-.03em; margin:3px 0 12px;\">{title}</div>",
        unsafe_allow_html=True,
    )


def render_topbar() -> dict:
    with st.container(border=True):
        col_brand, col_source, col_model = st.columns(
            [1.5, 1.6, 2.5], vertical_alignment="center", gap="small"
        )
        with col_brand:
            st.markdown(
                f"<div style=\"font-family:'Poppins',sans-serif; font-size:20px; font-weight:700; "
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
                format_func=lambda v: "My model" if v == "store" else "Live from Open-Meteo",
                index=0,
                horizontal=True,
                label_visibility="collapsed",
            )
        with col_model:
            parts = _model_label().split(" (", 1)
            model_name = parts[0]
            model_updated = f"({parts[1]}" if len(parts) > 1 else ""
            st.markdown(
                f'<div style="line-height:1.3;">'
                f'<span style="font-size:10px; color:{MUTED}; text-transform:uppercase; letter-spacing:.1em;">Model used for prediction</span><br>'
                f'<span style="display:inline-block; background:#e8f0fe; color:#1a56c9; '
                f'border:1px solid #c5d7f2; border-radius:999px; padding:2px 10px; '
                f'font-size:12px; font-weight:600;">{model_name}</span>'
                f' <span style="font-size:11px; color:{MUTED};">{model_updated}</span>'
                f'</div>',
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

    # Reference data: read from pre-fetched forecast JSON (zero runtime fetches)
    ref_series = _ref_series_from_forecast()
    ref_now = _ref_now_from_forecast()
    if ref_now is None:
        ref_now = forecast.get("ref_now")

    # Re-anchor reference series to the forecast origin so both lines
    # start at the same time on the chart.
    if len(ref_series) and origin is not None:
        ref_series = pd.Series(
            ref_series.values,
            index=pd.date_range(origin, periods=len(ref_series), freq="h"),
            name="aqi",
        )

    # Location + meta on the left; AQI hero card on the right.
    left_col, right_col = st.columns([1.0, 0.9], gap="medium")
    with left_col:
        st.markdown(
            "<div style=\"font-family:'Poppins',sans-serif; font-size:30px; font-weight:700; "
            "color:#241812; letter-spacing:-.03em; margin:6px 0 2px;\">Air quality in Karak</div>"
            f'<div style="font-size:14px; color:{MUTED}; line-height:1.5;">Air quality index (AQI) and PM2.5 air pollution '
            f"in Karak \u00b7 As of {origin:%d %b %Y, %H:00} \u00b7 Asia/Karachi</div>"
            f'<div style="font-size:13px; color:{MUTED}; margin-top:10px;">'
            f"Forecast origin \u00b7 {model_label}</div>",
            unsafe_allow_html=True,
        )
    with right_col:
        hero_slot = st.empty()
        hero_slot.markdown(HERO_SKELETON, unsafe_allow_html=True)

    with hero_slot.container():
        render_hero(source, rows, current_aqi, ref_now, model_label)

    render_metric_cards(origin, rows, ref_now)
    render_alerts(rows)

    section_header("Hourly forecast", "Next 24 hours, hour by hour")
    render_hourly_strip(rows)

    view = st.radio(
        "Compare",
        options=["all", "ours", "ref"],
        format_func=lambda v: {
            "all": "All sources",
            "ours": "Our model",
            "ref": "Live (Open-Meteo)",
        }[v],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    render_main_chart(origin, rows, ref_series, view)

    section_header("Extended forecast", "Beyond 24 hours \u2014 six- and twelve-hour means")
    render_block_means(rows)

    render_comparison(rows, ref_series)
    render_prediction_bar_chart(rows)

    with st.expander("Model comparison & evaluation"):
        render_model_history()

    with st.expander("SHAP explanations of the latest prediction"):
        render_shap()

    with st.expander("History / EDA"):
        render_eda()

    render_weather_insights()

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
        "<span>Reference: Open-Meteo AQ (US AQI\u202f\u200a)</span>"
        "</div>"
        '<div style="font-size:11px; color:'
        + MUTED
        + '; margin-top:4px;">'
        "Pre-computed forecasts served statically from karAQI-data. "
        "Auto-updates via GitHub Actions: feature pipeline (hourly) + training pipeline (daily 00:30 UTC)."
        "</div>"
    )
    st.markdown(status_html, unsafe_allow_html=True)

    # Debug panel
    with st.expander("Debug info", expanded=False):
        st.json({
            "data_source": DATA_REPO,
            "forecast_url": FORECAST_URL,
            "model_eval_url": MODEL_EVAL_URL,
            "generated_at": forecast.get("generated_at"),
            "source": forecast.get("source"),
            "model": forecast.get("model"),
            "outputs_count": len(forecast.get("outputs", [])),
            "ref_forecast_count": len(forecast.get("ref_forecast", [])),
            "ref_now": ref_now,
            "current_aqi": current_aqi,
            "origin": str(origin),
            "rows_count": len(rows),
            "ref_series_len": len(ref_series),
        })


if __name__ == "__main__":
    main()
