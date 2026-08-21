"""Fetch IQAir's hourly forecast for Karak and store as a static JSON file.

This script runs ONCE PER DAY (via a daily GitHub Actions workflow).
It fetches IQAir's hourly forecast table and stores it so the dashboard
can read it without scraping at runtime.

IQAir rate-limits anonymous reads aggressively. The strategy is:
1. Fetch once per day with proper browser-like headers
2. On success, overwrite data/iqair_forecast.json
3. On failure, keep the previous file untouched (stale data is better than none)
4. Check freshness: skip fetch if existing data is less than 20 hours old

Usage (from ``development``)::

    python -m src.fetch_iqair
    python -m src.fetch_iqair --force   # bypass freshness check
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

# Use headers that closely mimic a real browser request.
# IQAir checks for bot-like patterns and blocks requests without
# proper Accept, Accept-Language, and Referer headers.
IQAIR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.iqair.com/",
    "Sec-Ch-Ua": '"Chromium";v="128", "Not_A Brand";v="24", "Google Chrome";v="128"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1",
}

# How old the existing data can be before we re-fetch (hours).
FRESHNESS_HOURS = 20


def _current_hour_local() -> pd.Timestamp:
    """Current hour in the project timezone (naive, aligned to the data grid)."""
    return pd.Timestamp.now(tz=config.TIMEZONE).floor("h").tz_localize(None)


def _is_data_fresh(path: Path, max_age_hours: int = FRESHNESS_HOURS) -> bool:
    """Check if the existing IQAir JSON is recent enough to skip fetching."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            return False
        # Check the timestamp of the first entry
        first_time = pd.Timestamp(data[0]["time"])
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        age_hours = (now - first_time).total_seconds() / 3600
        return age_hours < max_age_hours
    except Exception:
        return False


def fetch_iqair_hourly() -> list[dict]:
    """Scrape IQAir's server-rendered hourly forecast table.

    Returns a list of dicts with 'time' and 'aqi' keys, or empty list on failure.
    Uses proper browser headers and exponential backoff on 429 errors.
    """
    import re

    for attempt in range(4):
        try:
            response = requests.get(
                IQAIR_URL,
                headers=IQAIR_HEADERS,
                timeout=30,
                allow_redirects=True,
            )
            if response.status_code == 429:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
                logger.warning(
                    "IQAir 429 rate-limited, attempt %d — waiting %ds",
                    attempt + 1, wait,
                )
                time.sleep(wait)
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
            wait = 2 ** attempt * 3
            logger.warning("IQAir fetch attempt %d failed: %s", attempt + 1, exc)
            if attempt < 3:
                time.sleep(wait)

    logger.error("All IQAir fetch attempts failed")
    return []


def main() -> None:
    """Fetch IQAir and write to JSON (keeping previous data on failure)."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Fetch IQAir hourly forecast")
    parser.add_argument("--force", action="store_true", help="Bypass freshness check")
    args = parser.parse_args()

    IQAIR_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Skip if data is fresh enough
    if not args.force and _is_data_fresh(IQAIR_PATH):
        logger.info("IQAir data is fresh (< %dh old), skipping fetch", FRESHNESS_HOURS)
        return

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
