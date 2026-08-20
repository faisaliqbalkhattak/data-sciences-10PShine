"""Fetch IQAir's hourly forecast for Karak and store as a static JSON file.

IQAir rate-limits anonymous reads aggressively, so this script:
1. Fetches with retries and backoff
2. On success, overwrites data/iqair_forecast.json
3. On failure, keeps the previous file untouched

The dashboard and forecast pipeline read from this file instead of
scraping IQAir at runtime, giving instant page loads.

Usage (from ``development``)::

    python -m src.fetch_iqair
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

logger = logging.getLogger(__name__)

#: Where the dashboard reads the IQAir forecast.
IQAIR_PATH = PROJECT_ROOT / "data" / "iqair_forecast.json"

IQAIR_URL = (
    "https://www.iqair.com/sg/air-quality/pakistan/khyber-pakhtunkhwa/karak"
)
IQAIR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _current_hour_local() -> pd.Timestamp:
    """Current hour in the project timezone (naive, aligned to the data grid)."""
    return pd.Timestamp.now(tz=config.TIMEZONE).floor("h").tz_localize(None)


def fetch_iqair_hourly() -> list[dict]:
    """Scrape IQAir's server-rendered hourly forecast table.

    Returns a list of dicts with 'time' and 'aqi' keys, or empty list on failure.
    """
    import re

    for attempt in range(4):
        try:
            response = requests.get(IQAIR_URL, headers=IQAIR_HEADERS, timeout=30)
            if response.status_code == 429:
                logger.warning("IQAir 429 rate-limited, attempt %d", attempt + 1)
                time.sleep(2**attempt * 3)
                continue
            response.raise_for_status()
            html = response.text

            table_start = html.find("Hourly forecast")
            if table_start == -1:
                raise RuntimeError("Hourly forecast table not found in HTML")
            table_end = html.find("</table>", table_start)
            if table_end == -1:
                raise RuntimeError("Table closing tag not found")

            section = html[table_start:table_end]
            values: list[float] = []
            for m in re.finditer(
                r"aqi-bg-[a-z-]+.*?<p[^>]*>\s*(\d+)\s*</p>", section, re.S
            ):
                values.append(float(m.group(1)))

            if not values:
                raise RuntimeError("No AQI values found in regex match")

            origin = _current_hour_local()
            index = pd.date_range(origin, periods=len(values), freq="h")
            result = []
            for ts, val in zip(index, values):
                result.append({
                    "time": pd.Timestamp(ts).isoformat(),
                    "aqi": round(float(val), 1),
                })
            logger.info("Fetched %d IQAir hourly values", len(result))
            return result

        except Exception as exc:
            logger.warning("IQAir fetch attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2**attempt * 3)

    logger.error("All IQAir fetch attempts failed")
    return []


def main() -> None:
    """Fetch IQAir and write to JSON (keeping previous data on failure)."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Fetch IQAir hourly forecast")
    parser.add_argument("--force", action="store_true", help="Overwrite even if file exists")
    args = parser.parse_args()

    IQAIR_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read previous data to keep on failure
    previous = []
    if IQAIR_PATH.exists():
        try:
            previous = json.loads(IQAIR_PATH.read_text(encoding="utf-8"))
        except Exception:
            previous = []

    new_data = fetch_iqair_hourly()

    if new_data:
        IQAIR_PATH.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
        logger.info("Wrote %d IQAir values to %s", len(new_data), IQAIR_PATH)
    elif previous:
        logger.warning("Keeping previous IQAir data (%d values)", len(previous))
    else:
        logger.error("No IQAir data available (first run failed)")


if __name__ == "__main__":
    main()
