"""Streamlit dashboard for the Karak AQI Predictor -- Google Material / IQAir style.

A single scrolling page built like a Google Weather / Google Fit screen with an
IQAir-style hero widget: a category-colored panel with an AQI badge, status,
dominant pollutant and a white weather strip. Below it: an hourly forecast
strip, an Altair chart of the 72-hour window comparing our model with IQAir's
own hourly forecast for Karak, a comparison table, and expandable sections for
SHAP, model history and EDA.

The forecast window is anchored to the current hour: the origin is the most
recent completed hour, predictions run from the next hour (the current hour in
Google-Weather terms) out to +72h, and refreshing after the hour passes moves
the whole window forward by one hour. No model or prediction logic changes --
only the input data is kept current and the rendering is new.

Run from ``development``::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

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

PROCESSED = config.DATA_PROCESSED_DIR
API_URL = os.environ.get("AQI_API_URL", "http://127.0.0.1:8000")

# Semantic palette tailored from the portfolio design: green for environment,
# orange for warning, red for hazards, blue for information. The orange ramp
# is the portfolio's own terracotta scale.
CATEGORY_COLORS = {
    "Good": "#2e7d32",
    "Moderate": "#9ccc65",
    "Unhealthy for Sensitive Groups": "#f47a32",
    "Unhealthy": "#e26225",
    "Very Unhealthy": "#d93025",
    "Hazardous": "#8f2f12",
}
INK = "#241812"
MUTED = "#5c4a3f"  # darker warm brown: AA contrast for small text on cream
SURFACE = "#fffaf5"
CANVAS = "#f6eee7"
LINE = "#eadbd0"
ORANGE_700 = "#c84f1b"
KICKER = "#a83c10"  # brand terracotta darkened for AA contrast on cream
INFO_BLUE = "#4a7dd6"  # chart reference line
INFO_BLUE_TEXT = "#1a56c9"  # info text: AA contrast on cream
DISPLAY_FONT = "'Space Grotesk',sans-serif"
# Lucide-style inline SVG icons (stroke-based, 24x24 viewBox)
_LU = ("'none'" , "'currentColor'")  # not used as constants, just reference

# No emojis or SVGs -- Streamlit's markdown renderer strips <svg>/<span> tags
# so inline icons don't render.  Clean text labels only.

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
/* Let the main grid use the full viewport width, not a centered WordPress column */
.main .block-container { padding-left: 2.5rem; padding-right: 2.5rem; }
h1, h2, h3 {
    color: #241812; font-weight: 700; letter-spacing: -0.02em;
    font-family: 'Space Grotesk', 'DM Sans', sans-serif;
}
section[data-testid="stSidebar"] { display: none; }

/* Streamlit's own chrome header stays full-size (dark ink bar with Deploy/menu)
   so no info is cropped; the app content clears it via block-container padding. */
header[data-testid="stHeader"] {
    background: #241812 !important;
    border-bottom: 1px solid #eadbd0;
}
[data-testid="stDecoration"] { display: none !important; }

/* Top bar: brand, source pills, refresh, model chip in one row */
.topbar {
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    background: #fffaf5; border: 1px solid #eadbd0; border-radius: 16px;
    padding: 12px 18px; margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(91, 44, 18, 0.08);
}
/* Top bar: one slim toolbar row, like the black chrome bar above it */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fffaf5;
    border: 1px solid #eadbd0 !important;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(91, 44, 18, 0.06);
    padding: 6px 18px;
    margin-bottom: 14px;
}
/* Inside the top bar: no extra borders/shadows on children */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stRadio"] > div[role="radiogroup"] {
    background: #eadbd0;
}
/* Inside metric card row: remove duplicate card borders */
[data-testid="stVerticalBlockBorderWrapper"] > div > div > div > div > div {
    box-shadow: none !important;
    border: none !important;
}

/* Cards: Material-style 16px radius + soft elevation (Google L2) */
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

/* Captions keep AA contrast on the cream canvas */
div[data-testid="stCaptionContainer"] p { color: #5c4a3f !important; }

/* Segmented control: Google-style pills, equal-width, clearly distinct active state */
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
/* Streamlit renders the label text as nested Markdown and re-colors it -- force contrast */
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"],
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
    color: #5c4a3f !important;
}
div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"],
div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color: #c84f1b !important;
}
div[data-testid="stRadio"] label > div:first-child { display: none; }

/* Buttons: wide, short brand-ink pill (refresh stays on one line) */
div.stButton > button {
    background: #241812; color: #fffaf5 !important; border: none; border-radius: 999px;
    padding: 7px 24px; font-weight: 600; white-space: nowrap; height: 38px;
    box-shadow: 0 2px 6px rgba(36, 24, 18, 0.25);
}
div.stButton > button div[data-testid="stMarkdownContainer"] p { color: #fffaf5 !important; white-space: nowrap; }
div.stButton > button:hover { background: #3a2a1e; color: #ffffff !important; border: none; }
div.stButton > button:active, div.stButton > button:focus { border: none; outline: none; }

/* Toggle / checkbox: visible box even when off (BaseWeb renders st.toggle as a checkbox) */
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

/* Accessibility: visible focus ring */
:focus-visible { outline: 2px solid #c84f1b; outline-offset: 2px; }

hr { border-color: #eadbd0; }

/* Readable, on-brand text selection */
::selection { background: #f6ae76; color: #241812; }
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
    """Perceptual luminance > 140/255: use dark text, else white text."""
    r, g, b = _rgb(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 140


# --------------------------------------------------------------------------
# Pipeline metadata: last data fetch, last model training
# --------------------------------------------------------------------------
def _pipeline_metadata() -> dict:
    """Read timestamps from the feature store and model artifacts."""
    import os
    from datetime import datetime

    result = {"data_fetched": None, "model_trained": None}
    # Feature store mtime = last time data was ingested
    fs = config.FEATURE_STORE_DIR / "karak_feature_store.duckdb"
    if fs.exists():
        result["data_fetched"] = datetime.fromtimestamp(os.path.getmtime(fs))
    # Model manifest mtime = last time models were trained
    for name in ("aqi_forecast_hourly_models.json", "aqi_forecast_models.json"):
        p = config.PROJECT_ROOT / "models" / name
        if p.exists():
            mt = datetime.fromtimestamp(os.path.getmtime(p))
            if result["model_trained"] is None or mt > result["model_trained"]:
                result["model_trained"] = mt
    return result


# --------------------------------------------------------------------------
# Data fetching (same contract as before, plus current-hour anchoring)
# --------------------------------------------------------------------------
def _load_direct(source: str) -> tuple[pd.Timestamp, pd.DataFrame, list, dict]:
    """Forecast via direct function calls (no API server needed)."""
    from app.explain import explain_latest_origin
    from app.live_data import current_conditions, load_latest_hourly
    from src.inference_hourly import predict_latest
    from src.train_hourly import build_hourly_training_frame

    hourly = load_latest_hourly(source)
    forecast = predict_latest(hourly)
    origin = pd.Timestamp(forecast["forecast_origin"].iloc[0])
    rows = forecast[["kind", "start_time", "end_time", "value"]].copy()
    rows["category"] = rows["value"].map(aqi_category)
    features = build_hourly_training_frame(hourly, include_targets=False)
    refs = {"iqair": _iqair_reference(), "current": current_conditions(hourly)}
    return origin, rows, features, refs


def _load_via_api(source: str) -> tuple[pd.Timestamp, pd.DataFrame, dict, dict]:
    """Forecast through the FastAPI backend."""
    import requests

    response = requests.get(f"{API_URL}/forecast", params={"source": source}, timeout=90)
    response.raise_for_status()
    payload = response.json()
    origin = pd.Timestamp(payload["origin"])
    rows = pd.DataFrame(payload["outputs"])
    rows["start_time"] = pd.to_datetime(rows["start_time"])
    rows["end_time"] = pd.to_datetime(rows["end_time"])
    refs = {
        "iqair": _series_from_payload(payload.get("iqair_forecast") or []),
        "current": payload.get("current_conditions") or {},
    }
    return origin, rows, payload, refs


def _series_from_payload(items: list) -> pd.Series:
    if not items:
        return pd.Series(dtype=float, name="aqi")
    return pd.Series(
        [float(item["aqi"]) for item in items],
        index=pd.to_datetime([item["time"] for item in items]),
        name="aqi",
    )


def _iqair_reference() -> pd.Series:
    """Best-effort IQAir hourly forecast AQI (empty on failure)."""
    from app.live_data import iqair_forecast_aqi

    try:
        return iqair_forecast_aqi()
    except Exception:  # noqa: BLE001 - reference line is optional
        return pd.Series(dtype=float)


def _model_label() -> str:
    """Best-effort label of the registered champion (the live model)."""
    try:
        from src.model_registry import list_registered

        for model in list_registered():
            if model["name"] == "aqi-hourly-ridge":
                versions = model.get("latest_versions") or []
                if versions:
                    return f"aqi-hourly-ridge · v{versions[-1].get('version')} champion"
    except Exception:  # noqa: BLE001 - label is cosmetic
        pass
    return "aqi-hourly-ridge"


# --------------------------------------------------------------------------
# Rendering: Google Material components
# --------------------------------------------------------------------------
def _source_label(source: str) -> str:
    return "Feature store (scheduled)" if source == "store" else "Live (Open-Meteo pull)"


POLLUTANT_LABELS = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "ozone": "O₃",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO₂",
    "sulphur_dioxide": "SO₂",
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
    origin: pd.Timestamp,
    rows: pd.DataFrame,
    source: str,
    model_label: str,
    current: dict,
    iqair_now: float | None = None,
) -> None:
    """Hero AQI panel: the highlighted badge is the number that matters for
    the chosen mode.

    * ``store`` -- the big badge is our model's next-hour forecast (the
      dashboard's headline), with IQAir's current reading as the secondary
      line.
    * ``live``  -- the big badge is IQAir's live current-hour AQI (the live
      reference), with our model's next-hour forecast as the secondary line.

    ``current`` carries the dominant pollutant + concentration, and ``rows``
    the model outputs, so the two numbers are never confused.
    """
    first = rows.iloc[0]
    model_aqi = float(first["value"])
    model_category = first["category"] or "Good"

    live = source == "live" and iqair_now is not None
    if live:
        badge_aqi = iqair_now
        badge_category = aqi_category(iqair_now) or model_category
        badge_label = "US AQI⁺ · IQAir live"
        secondary_line = f"Ours (next hour): {model_aqi:.0f}"
    else:
        badge_aqi = model_aqi
        badge_category = model_category
        badge_label = "US AQI⁺ · next hour"
        secondary_line = (
            f"IQAir now: {iqair_now:.0f}" if iqair_now is not None else ""
        )

    color = category_color(badge_category)
    text_color = INK if is_light(color) else "#ffffff"
    panel = shade(color, 0.86)

    pollutant = _pollutant_label(current.get("main_pollutant"))
    concentration = current.get("concentration")
    concentration_html = (
        f"{concentration:.1f} µg/m³" if concentration is not None else "—"
    )
    secondary_html = (
        f'<div style="font-size:12px; margin-top:10px; opacity:.92; font-weight:600;">'
        f'{secondary_line}</div>'
        if secondary_line
        else ""
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
            {secondary_html}
          </div>
        </div>
      </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


def render_metric_cards(origin: pd.Timestamp, rows: pd.DataFrame, refs: dict) -> None:
    """Small stat tiles -- the AQI number itself lives in the hero, so these
    only carry the non-redundant forecast facts."""
    peak24 = float(rows[rows["kind"] == "point"]["value"].max())
    max72 = float(rows["value"].max())
    iqair = _ref_or_empty(refs, "iqair")
    iqair_now = float(iqair.iloc[0]) if len(iqair) else None
    tiles = [
        ("Forecast origin", origin.strftime("%m-%d %H:%M"), MUTED, ""),
        ("Peak hourly · next 24h", f"{peak24:.0f}", MUTED, ""),
        ("Max · full 72h", f"{max72:.0f}", MUTED, ""),
        ("IQAir now", f"{iqair_now:.0f}" if iqair_now is not None else "—", INFO_BLUE_TEXT, "US AQI⁺"),
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
            f'{note_html}</div>'
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
              <b>HAZARDOUS AQI (≥ 301) predicted</b> in the next 72 hours at: {windows}.
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
              <b>Very Unhealthy AQI (201–300) predicted</b> at: {windows}.
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
              <div style="font-size:11px; color:{MUTED};">{row.start_time:%d %b %H:%M} → {row.end_time:%d %b %H:%M}</div>
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


IQAIR_GREEN = "#2e7d32"  # environment -- IQAir reference line


def render_main_chart(
    origin: pd.Timestamp,
    rows: pd.DataFrame,
    refs: dict,
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

    iqair = _ref_or_empty(refs, "iqair")

    if view in ("all", "iqair") and len(iqair):
        iq_df = iqair.reset_index().rename(columns={"index": "time"})
        iq_layer = (
            alt.Chart(iq_df)
            .mark_line(strokeDash=[4, 3], color=IQAIR_GREEN, strokeWidth=2.2)
            .encode(x=alt.X("time:T", title=None), y=alt.Y("aqi:Q", title="AQI", scale=y_scale), tooltip=tooltip)
        )
        layers.append(iq_layer)

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
        f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="display:inline-block;width:18px;height:3px;background:{IQAIR_GREEN};border-radius:2px;border-top:2px dashed {IQAIR_GREEN};"></span> IQAir hourly forecast (US AQI⁺)</span>'
        f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="display:inline-block;width:18px;height:6px;background:#8f2f12;border-radius:2px;"></span> Six/twelve-hour means (our model)</span>'
        "</div>"
    )
    st.markdown(swatches, unsafe_allow_html=True)


def _ref_or_empty(refs: dict, key: str) -> pd.Series:
    series = refs.get(key)
    return series if series is not None and len(series) else pd.Series(dtype=float, name="aqi")


def comparison_frame(rows: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """Align our 30 outputs with IQAir on the same window.

    The same block-mean logic used for the model's own six/twelve-hour outputs
    is applied to the reference source: point outputs map to the reference
    value at that hour; block outputs map to the mean of the reference values
    inside the block's time window.
    """
    iqair = _ref_or_empty(refs, "iqair")
    records = []
    for row in rows.itertuples():
        window = (
            f"{row.start_time:%m-%d %H:%M}"
            if row.kind == "point"
            else f"{row.start_time:%m-%d %H:%M} → {row.end_time:%m-%d %H:%M}"
        )
        if row.kind == "point":
            iq_value = iqair.get(row.start_time, np.nan)
        else:
            mask_iq = (iqair.index >= row.start_time) & (iqair.index <= row.end_time)
            iq_block = iqair[mask_iq]
            iq_value = float(iq_block.mean()) if len(iq_block) else np.nan
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


def render_comparison(rows: pd.DataFrame, refs: dict) -> None:
    section_header("Comparison", "Our model vs IQAir")
    st.markdown(
        f'<div style="font-size:13px; color:{INFO_BLUE_TEXT}; margin-bottom:8px;">'
        "IQAir publishes its own hourly forecast for Karak labelled \"US AQI⁺\" -- "
        "the same US EPA AQI scale (categories, colors, breakpoints) this project's "
        "target uses, so the two are directly comparable. Mapped onto our exact 30 "
        "outputs with the same block-mean logic; diff = ours − IQAir.</div>",
        unsafe_allow_html=True,
    )
    frame = comparison_frame(rows, refs)
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
            "iqair": "IQAir (US AQI⁺)",
            "diff_iqair": "Δ vs IQAir",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_output_table(rows: pd.DataFrame) -> None:
    section_header("Full detail", "30-output forecast table")
    table = rows.copy()
    table["window"] = table.apply(
        lambda r: f"{r.start_time:%m-%d %H:%M}"
        if r["kind"] == "point"
        else f"{r.start_time:%m-%d %H:%M} → {r.end_time:%m-%d %H:%M}",
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


def render_shap(features, use_api: bool, source: str) -> None:
    output_choices = {
        "t+1h (hourly point)": 0,
        "t+24h (hourly point)": 23,
        "t+25..30h (six-hour mean)": 24,
        "t+49..60h (twelve-hour mean)": 28,
    }
    label = st.selectbox("Output to explain", list(output_choices.keys()), index=0)
    output_index = output_choices[label]
    try:
        if use_api:
            import requests

            response = requests.get(
                f"{API_URL}/explain", params={"output": output_index, "source": source}, timeout=90
            )
            response.raise_for_status()
            explanation = response.json()
        else:
            from app.explain import explain_latest_origin

            explanation = explain_latest_origin(features, output_index=output_index)
        top = pd.DataFrame(explanation["features"]).head(15)
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.42)))
        colors = ["#d93025" if v >= 0 else "#1a73e8" for v in top["shap"]]
        ax.barh(top["feature"][::-1], top["shap"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value → (positive pushes predicted AQI higher)")
        ax.set_title(f"SHAP attribution — {explanation['output_column']} (method: {explanation['method']})", fontsize=11)
        ax.grid(alpha=0.2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            f"Expected value: {explanation['expected_value']:.2f} · "
            f"model prediction: {explanation['prediction_base_plus_shap']:.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"SHAP explanation failed: {exc}")


def render_registry() -> None:
    """The MLflow registry's champion models (compact list)."""
    try:
        from src.model_registry import list_registered

        registered = {model["name"]: model["latest_versions"] for model in list_registered()}
    except Exception:  # noqa: BLE001 - best-effort
        registered = {}
    if not registered:
        st.caption("Registry unavailable (run `python -m src.model_registry register-hourly`).")
        return
    st.subheader("Model registry (MLflow, local)")
    rows = []
    for name, versions in list(registered.items())[:4]:
        latest = versions[-1] if versions else {}
        aliases = ",".join(latest.get("alias") or []) or "none"
        rows.append({"model": name, "version": latest.get("version"), "alias": aliases})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_model_history() -> None:
    hourly_path = PROCESSED / "hourly_model_comparison.csv"
    daily_path = PROCESSED / "model_comparison.csv"
    rolling_path = PROCESSED / "hourly_rolling_origin_comparison.csv"

    render_registry()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly holdout (72h purge)")
        if hourly_path.exists():
            comparison = pd.read_csv(hourly_path)
            grouped = (
                comparison.groupby(["model", "group"])[["rmse", "mae", "r2"]].mean().reset_index()
            )
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
            st.info("Run `python -m src.train_hourly`.")
    with col2:
        st.subheader("Daily holdout (+1/+2/+3 days)")
        if daily_path.exists():
            daily = pd.read_csv(daily_path)
            pivot = daily.pivot_table(index="model", columns="horizon_days", values="rmse").round(2)
            st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Run `python -m src.train`.")

    if rolling_path.exists():
        st.subheader("Rolling-origin evaluation (3 expanding folds, 72h embargo)")
        rolling = pd.read_csv(rolling_path)
        grouped = (
            rolling.groupby(["model", "group"])[
                ["mse", "rmse", "mae", "r2", "category_accuracy", "high_aqi_recall"]
            ]
            .mean()
            .round(3)
            .reset_index()
        )
        st.dataframe(grouped, use_container_width=True, hide_index=True)
    else:
        st.info("No rolling-origin CSV found — run `python -m src.train_hourly`.")


def render_eda() -> None:
    hourly_path = PROCESSED / "training_frame_hourly.csv"
    daily_path = PROCESSED / "training_frame.csv"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly rolling AQI — last 90 days")
        if hourly_path.exists():
            frame = pd.read_csv(hourly_path, parse_dates=["time"]).set_index("time").sort_index().tail(90 * 24)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(frame.index, frame["aqi_hourly_rolling"], lw=0.7, color="#1a73e8")
            ax.set_ylabel("AQI")
            ax.grid(alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Run `python -m src.train_hourly`.")
    with col2:
        st.subheader("Daily EPA AQI — last 2 years")
        if daily_path.exists():
            frame = pd.read_csv(daily_path, parse_dates=["time"]).set_index("time").sort_index().tail(730)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(frame.index, frame["aqi_us_epa"], lw=0.9, color="#d93025")
            ax.set_ylabel("AQI (US EPA)")
            ax.grid(alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Run `python -m src.train`.")

    st.subheader("Observed AQI category distribution (hourly)")
    if hourly_path.exists():
        frame = pd.read_csv(hourly_path, parse_dates=["time"])
        counts = frame["aqi_hourly_rolling"].map(aqi_category).value_counts()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        colors = [category_color(category) for category in counts.index]
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_ylabel("Hours")
        ax.tick_params(axis="x", rotation=15)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)


def section_header(kicker: str, title: str) -> None:
    """Portfolio-style section header: tiny tracked kicker + Space Grotesk title."""
    st.markdown(
        f'<div style="font-size:11px; letter-spacing:.18em; text-transform:uppercase; '
        f'color:{KICKER}; font-weight:700; margin-top:28px;">{kicker}</div>'
        f'<div style="font-family:\'Space Grotesk\',sans-serif; font-size:20px; font-weight:700; '
        f'color:#241812; letter-spacing:-.03em; margin:3px 0 12px;">{title}</div>',
        unsafe_allow_html=True,
    )


def render_topbar() -> dict:
    """One slim toolbar row (like the black chrome bar above it): brand,
    source pills, refresh, model chip and the API toggle -- nothing else."""
    with st.container(border=True):
        col_brand, col_source, col_refresh, col_model, col_api = st.columns(
            [1.5, 1.6, 1.0, 1.8, 1.4], vertical_alignment="center", gap="small"
        )
        with col_brand:
            st.markdown(
                f'<div style="font-family:\'Space Grotesk\',sans-serif; font-size:20px; font-weight:700; '
                f'letter-spacing:-.04em; background:linear-gradient(120deg,#8f2f12,#f47a32); '
                f'-webkit-background-clip:text; background-clip:text; color:transparent;">'
                f'Karak AQI</div>'
                f'<div style="font-size:11px; color:{MUTED};">{config.CITY_NAME} \u00b7 {config.LOCATION_LABEL}</div>',
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
        with col_api:
            use_api = st.toggle("FastAPI backend", value=False)
    return {"source": source, "use_api": use_api}


def main() -> None:
    inject_css()
    options = render_topbar()
    source, use_api = options["source"], options["use_api"]
    model_label = _model_label()

    # Location prominent on the left; the AQI metric card sits on the right
    # (the IQAir arrangement), with the toolbar already rendered above.
    left_col, right_col = st.columns([1.0, 0.9], gap="medium")
    with left_col:
        st.markdown(
            '<div style="font-family:\'Space Grotesk\',sans-serif; font-size:30px; font-weight:700; '
            'color:#241812; letter-spacing:-.03em; margin:6px 0 2px;">Air quality in Karak</div>'
            f'<div style="font-size:13px; color:{MUTED};">Air quality index (AQI) and PM2.5 air pollution '
            f'in Karak · {datetime.now():%d %b %Y, %H:%M} · Asia/Karachi</div>'
            f'<div style="font-size:12px; color:{MUTED}; margin-top:10px;">'
            f'Forecast starts from the current hour · {model_label}</div>',
            unsafe_allow_html=True,
        )
    with right_col:
        hero_slot = st.empty()
        hero_slot.markdown(HERO_SKELETON, unsafe_allow_html=True)

    features = None
    status = st.status("Fetching latest observations and running the 72-hour forecast…", expanded=False)
    try:
        if use_api:
            origin, rows, payload, refs = _load_via_api(source)
            alerts = pd.DataFrame(payload["alerts"]) if payload.get("alerts") else pd.DataFrame()
        else:
            origin, rows, features, refs = _load_direct(source)
            alerts = rows[rows["category"].isin(["Very Unhealthy", "Hazardous"])]
        status.update(label="Forecast ready", state="complete")
    except Exception as exc:  # noqa: BLE001 - show a readable error instead of a traceback
        status.update(label="Forecast failed", state="error")
        hero_slot.empty()
        st.error(f"Could not load a forecast: {exc}")
        st.info(
            "Check that the model artifacts exist (`python -m src.train_hourly`) and the "
            "feature store is populated (`python -m src.feature_store backfill-hourly --replace`)."
        )
        return

    # IQAir's series starts at the current hour; grab that "now" reading before
    # anchoring the reference to the forecast window (which starts at t+1h).
    iqair_now = None
    iqair_full = refs.get("iqair")
    if iqair_full is not None and len(iqair_full):
        iqair_now = float(iqair_full.iloc[0])

    # Anchor every reference line to the model's forecast window: the next hour
    # (the current hour when the data is fresh) through +72h.
    window_start, window_end = rows["start_time"].min(), rows["end_time"].max()
    for key in ("iqair",):
        series = refs.get(key)
        if series is not None and len(series):
            refs[key] = series[(series.index >= window_start) & (series.index <= window_end)]

    with hero_slot.container():
        render_hero(origin, rows, source, model_label, refs.get("current") or {}, iqair_now)

    render_metric_cards(origin, rows, refs)

    render_alerts(rows)

    section_header("Hourly forecast", "Next 24 hours, hour by hour")
    render_hourly_strip(rows)

    view = st.radio(
        "Compare",
        options=["all", "ours", "iqair"],
        format_func=lambda v: {
            "all": "All sources",
            "ours": "Our model",
            "iqair": "IQAir (US AQI⁺)",
        }[v],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    render_main_chart(origin, rows, refs, view)

    section_header("Extended forecast", "Beyond 24 hours — six- and twelve-hour means")
    render_block_means(rows)

    render_comparison(rows, refs)
    render_output_table(rows)

    with st.expander("SHAP explanations of the latest prediction"):
        render_shap(features, use_api, source)

    with st.expander("Model comparison & evaluation"):
        render_model_history()

    with st.expander("History / EDA"):
        render_eda()

    st.divider()
    meta = _pipeline_metadata()
    fetched = meta["data_fetched"].strftime("%d %b %Y, %H:%M") if meta["data_fetched"] else "—"
    trained = meta["model_trained"].strftime("%d %b %Y, %H:%M") if meta["model_trained"] else "—"
    status_html = (
        '<div style="display:flex; gap:24px; flex-wrap:wrap; font-size:12px; color:'
        + MUTED + '; padding:8px 0;">'
        '<span>Last data fetch: <b>' + fetched + '</b> (Open-Meteo, keyless)</span>'
        '<span>Last model training: <b>' + trained + '</b> (MLflow, file-backed)</span>'
        '<span>Feature store: DuckDB · Reference: IQAir (US AQI⁺)</span>'
        '</div>'
        '<div style="font-size:11px; color:' + MUTED + '; margin-top:4px;">'
        'Forecasts are estimates, not station measurements. '
        'Auto-updates via GitHub Actions: feature pipeline (hourly) + training pipeline (daily 01:15 UTC).'
        '</div>'
    )
    st.markdown(status_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
