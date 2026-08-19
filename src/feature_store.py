"""Lightweight in-repo feature store for the Karak AQI pipeline.

Decision record
---------------
The assignment names Hopsworks or Vertex AI as the feature store. The project
notes documented real Hopsworks free-tier failures for this exact workflow
(hourly pipelines triggering billing freezes, ``imp`` module issues, RPC
disconnects), and the mentor session allowed documented alternatives. This
module is that documented alternative: a **DuckDB-backed store** that lives in
``data/feature_store/``, needs no server, no API key, and no paid tier, and
therefore satisfies the project's "100% serverless / free student stack"
criterion (C7 in ``Docs/model_selection_methodology.md``).

What it stores
--------------
``hourly_features``      -- the exact frame from
                           ``train_hourly.build_hourly_training_frame``:
                           processed features plus the 30-output target grid.
``hourly_observations``  -- the same processed features *without* future
                           targets (``include_targets=False``), the input
                           contract used by inference and the dashboard.
``hourly_raw``           -- the raw observed hourly columns (``time`` plus the
                           base pollutant/weather columns), used to feed
                           ``inference_hourly.predict_latest`` unchanged.
``daily_features``       -- the exact frame from ``train.build_training_frame``
                           (daily features plus the +1/+2/+3 day targets).
``meta``                 -- schema version, feature-column contract, source
                           file hashes, and row counts per table.

The store intentionally **consumes the validated feature builders rather than
re-implementing them**, so the schema contract recorded in the model manifests
cannot drift between training, inference, and the dashboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from . import config
from .train import build_training_frame
from .train_hourly import (
    BASE_FEATURES as HOURLY_BASE_FEATURES,
    TARGET_COLUMNS as HOURLY_TARGET_COLUMNS,
    build_hourly_training_frame,
)

SCHEMA_VERSION = 1

#: Table -> purpose mapping used by the CLI and by :func:`store_stats`.
TABLES = (
    "hourly_raw",
    "hourly_observations",
    "hourly_features",
    "daily_features",
)

_META_TABLE = "meta"


def _connect(store_path: Optional[Path] = None) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the DuckDB feature store."""
    path = Path(store_path or config.FEATURE_STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_META_TABLE} (
            table_name VARCHAR PRIMARY KEY,
            schema_version INTEGER,
            feature_columns VARCHAR,
            target_columns VARCHAR,
            n_rows INTEGER,
            source_file VARCHAR,
            source_sha256 VARCHAR,
            created_at TIMESTAMP
        )
        """
    )
    return connection


def _schema_checksum(columns: list[str]) -> str:
    """Stable hash of a column list so schema drift is detectable."""
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _write_frame(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    frame: pd.DataFrame,
    source_file: str,
    source_sha256: str,
    feature_columns: list[str],
    target_columns: list[str],
    replace: bool,
) -> None:
    """Write one frame to the store and record its metadata row."""
    if replace:
        connection.execute(f"DROP TABLE IF EXISTS {table}")
    connection.register("_frame", frame.reset_index())
    if replace:
        connection.execute(f"CREATE TABLE {table} AS SELECT * FROM _frame")
    else:
        connection.execute(f"INSERT INTO {table} SELECT * FROM _frame")
    connection.execute(
        f"""
        INSERT OR REPLACE INTO {_META_TABLE}
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            table,
            SCHEMA_VERSION,
            json.dumps(feature_columns),
            json.dumps(target_columns),
            len(frame),
            source_file,
            source_sha256,
            datetime.utcnow(),
        ],
    )
    connection.unregister("_frame")


def _read_frame(
    connection: duckdb.DuckDBPyConnection, table: str
) -> pd.DataFrame:
    frame = connection.execute(
        f"SELECT * FROM {table} ORDER BY time"
    ).df()
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.set_index("time")


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_hourly_source() -> Path:
    path = config.DATA_PROCESSED_DIR / "karak_aqi_open_meteo_hourly_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run notebook 02 before backfilling hourly features.")
    return path


def _default_daily_source() -> Path:
    matches = sorted(
        config.DATA_PROCESSED_DIR.glob("karak_aqi_open_meteo_daily_features.csv")
    )
    if not matches:
        raise FileNotFoundError("Run notebooks 01-02 before backfilling daily features.")
    return matches[-1]


def backfill_hourly(
    source_csv: Optional[Path] = None,
    store_path: Optional[Path] = None,
    replace: bool = False,
) -> dict:
    """Backfill the hourly tables from a processed hourly features CSV.

    The processed frame is rebuilt through ``build_hourly_training_frame`` so
    the stored schema is byte-for-byte the validated training/inference
    contract. ``replace=True`` (the normal backfill mode) makes the operation
    idempotent: the feature pipeline can run every hour and regenerate the
    store from the latest Open-Meteo data.
    """
    source = Path(source_csv) if source_csv else _default_hourly_source()
    if not source.exists():
        raise FileNotFoundError(f"Hourly source CSV not found: {source}")
    hourly = pd.read_csv(source, parse_dates=["time"])

    features = build_hourly_training_frame(hourly, include_targets=False)
    full = build_hourly_training_frame(hourly, include_targets=True)

    raw = hourly.copy()
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.set_index("time")
    raw = raw[[c for c in HOURLY_BASE_FEATURES if c in raw.columns]]

    expected = list(features.columns)
    missing = [c for c in expected if c not in full.columns]
    if missing:
        raise ValueError(f"Hourly feature store columns missing from full frame: {missing}")

    checksum = _source_sha256(source)
    with _connect(store_path) as connection:
        _write_frame(
            connection,
            "hourly_raw",
            raw,
            source.name,
            checksum,
            [c for c in HOURLY_BASE_FEATURES if c in raw.columns],
            [],
            replace,
        )
        _write_frame(
            connection,
            "hourly_observations",
            features,
            source.name,
            checksum,
            expected,
            [],
            replace,
        )
        _write_frame(
            connection,
            "hourly_features",
            full,
            source.name,
            checksum,
            expected,
            list(HOURLY_TARGET_COLUMNS),
            replace,
        )
    return {
        "table": "hourly",
        "rows": len(features),
        "target_rows": len(full),
        "source": source.name,
        "source_sha256": checksum,
        "replace": replace,
    }


def backfill_daily(
    source_csv: Optional[Path] = None,
    store_path: Optional[Path] = None,
    replace: bool = False,
) -> dict:
    """Backfill the daily table from a processed daily features CSV."""
    source = Path(source_csv) if source_csv else _default_daily_source()
    if not source.exists():
        raise FileNotFoundError(f"Daily source CSV not found: {source}")
    daily = pd.read_csv(source)
    frame = build_training_frame(daily)

    feature_columns = [
        c for c in frame.columns if not c.startswith("target_") and c != "aqi_us_epa"
    ]
    target_columns = [f"target_{h}d" for h in (1, 2, 3)]
    checksum = _source_sha256(source)
    with _connect(store_path) as connection:
        _write_frame(
            connection,
            "daily_features",
            frame,
            source.name,
            checksum,
            feature_columns,
            target_columns,
            replace,
        )
    return {
        "table": "daily",
        "rows": len(frame),
        "source": source.name,
        "source_sha256": checksum,
        "replace": replace,
    }


def get_hourly_features(
    start: Optional[str] = None,
    end: Optional[str] = None,
    store_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return stored hourly features plus targets as a DatetimeIndex frame."""
    with _connect(store_path) as connection:
        frame = _read_frame(connection, "hourly_features")
    return _slice(frame, start, end)


def get_hourly_observations(
    start: Optional[str] = None,
    end: Optional[str] = None,
    store_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return stored processed hourly features (no future targets)."""
    with _connect(store_path) as connection:
        frame = _read_frame(connection, "hourly_observations")
    return _slice(frame, start, end)


def get_hourly_raw(
    start: Optional[str] = None,
    end: Optional[str] = None,
    store_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return the raw observed hourly columns (inference input contract)."""
    with _connect(store_path) as connection:
        frame = _read_frame(connection, "hourly_raw")
    return _slice(frame, start, end)


def get_daily_features(
    start: Optional[str] = None,
    end: Optional[str] = None,
    store_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return stored daily features plus targets as a DatetimeIndex frame."""
    with _connect(store_path) as connection:
        frame = _read_frame(connection, "daily_features")
    return _slice(frame, start, end)


def _slice(frame: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]
    return frame


def latest_hourly_origin(store_path: Optional[Path] = None) -> Optional[pd.Timestamp]:
    """Latest timestamp stored in the hourly observations table."""
    with _connect(store_path) as connection:
        row = connection.execute(
            "SELECT max(time) AS latest FROM hourly_observations"
        ).fetchone()
    return pd.Timestamp(row[0]) if row and row[0] is not None else None


def validate_feature_schema(
    expected_columns: list[str],
    store_path: Optional[Path] = None,
    table: str = "hourly_observations",
) -> None:
    """Raise if the stored feature columns drift from the expected contract."""
    with _connect(store_path) as connection:
        row = connection.execute(
            f"SELECT feature_columns FROM {_META_TABLE} WHERE table_name = ?", [table]
        ).fetchone()
    if not row:
        raise ValueError(f"Feature store has no metadata for table {table!r}.")
    stored = json.loads(row[0])
    if stored != expected_columns:
        missing = sorted(set(expected_columns) - set(stored))
        extra = sorted(set(stored) - set(expected_columns))
        raise ValueError(
            f"Feature store schema drift on {table!r}: "
            f"missing={missing} extra={extra}"
        )


def store_stats(store_path: Optional[Path] = None) -> list[dict]:
    """Return a human-readable summary of every stored table."""
    with _connect(store_path) as connection:
        rows = connection.execute(
            f"""
            SELECT table_name, schema_version, feature_columns, target_columns,
                   n_rows, source_file, source_sha256, created_at
            FROM {_META_TABLE} ORDER BY table_name
            """
        ).fetchall()
    return [
        {
            "table": row[0],
            "schema_version": row[1],
            "feature_columns": json.loads(row[2]),
            "target_columns": json.loads(row[3]),
            "rows": row[4],
            "source_file": row[5],
            "source_sha256": row[6],
            "created_at": str(row[7]),
        }
        for row in rows
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.feature_store",
        description="Karak AQI DuckDB feature store: backfill, export, inspect.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text, func in (
        ("backfill-hourly", "Rebuild hourly tables from the processed CSV.", backfill_hourly),
        ("backfill-daily", "Rebuild the daily table from the processed CSV.", backfill_daily),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--source", type=Path, default=None, help="Source CSV (default: newest processed file).")
        p.add_argument("--replace", action="store_true", help="Drop and recreate the table (idempotent backfill).")
        p.set_defaults(func=func)

    export = sub.add_parser("export-hourly", help="Export the stored hourly frame to CSV.")
    export.add_argument("--out", type=Path, required=True, help="Destination CSV path.")
    export.add_argument("--start", default=None)
    export.add_argument("--end", default=None)
    export.set_defaults(func=lambda args: _export(get_hourly_features, args))

    export_daily = sub.add_parser("export-daily", help="Export the stored daily frame to CSV.")
    export_daily.add_argument("--out", type=Path, required=True, help="Destination CSV path.")
    export_daily.set_defaults(func=lambda args: _export(get_daily_features, args))

    sub.add_parser("stats", help="Print table metadata.").set_defaults(
        func=lambda args: print(json.dumps(store_stats(), indent=2))
    )
    return parser


def _export(getter, args) -> None:
    frame = getter(start=args.start, end=args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(args.out, index=False)
    print(f"Exported {len(frame)} rows to {args.out}")


def main() -> None:
    args = _build_parser().parse_args()
    if args.command in ("backfill-hourly", "backfill-daily"):
        kwargs = {"source_csv": args.source, "replace": args.replace}
        if args.command == "backfill-hourly":
            print(json.dumps(backfill_hourly(**kwargs), indent=2))
        else:
            print(json.dumps(backfill_daily(**kwargs), indent=2))
    elif args.command.startswith("export-"):
        args.func(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
