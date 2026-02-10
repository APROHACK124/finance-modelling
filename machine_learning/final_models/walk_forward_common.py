
from __future__ import annotations

"""
walk_forward_common.py

Shared, leakage-safe utilities for annual walk-forward (retrain once per year) pipelines.

Key design goals:
- Anti-leakage: training uses only examples whose labels are known "as-of" the training cut-off.
- Reproducible & idempotent persistence: OOS predictions are upserted into SQLite without duplication.
- Model-agnostic: TCN/CNN/XGBoost can reuse the same fold schedule + runner.

Public API:
- shift_trading_days(...)
- build_annual_fold_schedule(...)
- build_final_train_fold_from_schedule(...)
- get_table_info(...)
- upsert_dataframe(...)
- run_walk_forward_oos_to_sqlite(...)

Notes about the predictions table schema:
- If your UNIQUE constraint includes a nullable column (e.g. model_version), you MUST ensure the
  code always writes a non-null value; otherwise SQLite allows duplicates because NULL != NULL.
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# =============================================================================
# Trading-day utilities
# =============================================================================

def _normalize_trading_dates(trading_dates: Sequence[Union[str, pd.Timestamp]]) -> pd.DatetimeIndex:
    """
    Normalize a list/array of trading dates:
    - parse to datetime
    - drop NaT
    - drop timezone
    - unique + sort
    """
    idx = pd.Index(list(trading_dates))
    dt = pd.to_datetime(idx, errors="coerce")
    dt = dt[dt.notna()]
    if isinstance(dt, pd.DatetimeIndex) and dt.tz is not None:
        dt = dt.tz_convert(None)
    # Ensure DatetimeIndex
    dt = pd.DatetimeIndex(dt)
    dt = pd.DatetimeIndex(sorted(dt.unique()))
    return dt


def shift_trading_days(
    trading_dates: Sequence[Union[str, pd.Timestamp]],
    anchor_date: Union[str, pd.Timestamp],
    offset: int,
) -> pd.Timestamp:
    """
    Shift an anchor date by N trading days using the supplied trading calendar.

    - If anchor_date is not an exact trading date, it is padded to the last trading date <= anchor_date.
    - Raises if shift goes out of bounds.
    """
    dates = _normalize_trading_dates(trading_dates)
    if len(dates) == 0:
        raise ValueError("Empty trading_dates")

    anchor = pd.to_datetime(anchor_date, errors="coerce")
    if pd.isna(anchor):
        raise ValueError(f"Invalid anchor_date: {anchor_date!r}")
    if getattr(anchor, "tzinfo", None) is not None:
        # Convert tz-aware -> naive
        try:
            anchor = anchor.tz_convert(None)
        except Exception:
            anchor = pd.Timestamp(anchor).tz_convert(None)

    # pad to last date <= anchor
    pos = int(dates.searchsorted(anchor, side="right") - 1)
    if pos < 0:
        raise ValueError(f"anchor_date {anchor} is earlier than first trading date {dates[0]}")

    new_pos = int(pos + offset)
    if new_pos < 0 or new_pos >= len(dates):
        raise ValueError(f"shift out of bounds: pos={pos}, offset={offset}, len={len(dates)}")
    return pd.Timestamp(dates[new_pos])


def build_annual_fold_schedule(
    trading_dates: Sequence[Union[str, pd.Timestamp]],
    start_year: int,
    first_val_year: int,
    last_val_year: int,
    horizon_td: int = 22,
) -> pd.DataFrame:
    """
    Build annual walk-forward schedule.

    For validation year Y:
      - train_fit_end_date      = last trading day of year (Y-1)
      - train_last_labeled_date = shift_trading_days(train_fit_end_date, -horizon_td)
      - val_start_date          = first trading day of year Y
      - val_end_date            = last trading day of year Y
      - global_last_labeled_date= shift_trading_days(max_date_in_dataset, -horizon_td)
      - val_last_labeled_date   = min(val_end_date, global_last_labeled_date)

    Returns DataFrame with one row per fold (val_year).
    """
    if horizon_td <= 0:
        raise ValueError("horizon_td must be > 0")
    if last_val_year < first_val_year:
        raise ValueError("last_val_year < first_val_year")

    dates = _normalize_trading_dates(trading_dates)
    if len(dates) < horizon_td + 5:
        raise ValueError("Not enough trading_dates for given horizon_td")

    max_date = pd.Timestamp(dates[-1])
    global_last_labeled_date = shift_trading_days(dates, max_date, -horizon_td)

    # train_start_date: first trading date whose year >= start_year
    mask_start = dates.year >= int(start_year)
    if not mask_start.any():
        raise ValueError(f"No trading dates found with year >= start_year={start_year}")
    train_start_date = pd.Timestamp(dates[mask_start][0])

    rows: List[Dict[str, Any]] = []
    for y in range(int(first_val_year), int(last_val_year) + 1):
        val_mask = dates.year == y
        train_mask = dates.year == (y - 1)

        if not val_mask.any():
            continue
        if not train_mask.any():
            continue

        val_start_date = pd.Timestamp(dates[val_mask][0])
        val_end_date = pd.Timestamp(dates[val_mask][-1])

        train_fit_end_date = pd.Timestamp(dates[train_mask][-1])
        train_last_labeled_date = shift_trading_days(dates, train_fit_end_date, -horizon_td)

        # Skip folds where train window would be empty under start_year constraint
        if train_last_labeled_date < train_start_date:
            continue

        val_last_labeled_date = min(val_end_date, global_last_labeled_date)

        rows.append(
            dict(
                fold_id=int(y),
                val_year=int(y),
                horizon_td=int(horizon_td),
                train_start_date=train_start_date,
                train_fit_end_date=train_fit_end_date,
                train_last_labeled_date=train_last_labeled_date,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
                val_last_labeled_date=val_last_labeled_date,
                global_last_labeled_date=global_last_labeled_date,
                schedule_created_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Ensure datetime types
    for c in [
        "train_start_date",
        "train_fit_end_date",
        "train_last_labeled_date",
        "val_start_date",
        "val_end_date",
        "val_last_labeled_date",
        "global_last_labeled_date",
    ]:
        out[c] = pd.to_datetime(out[c], errors="coerce")

    out = out.sort_values(["val_year"]).reset_index(drop=True)
    return out


def validate_annual_fold_schedule(fold_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Validate basic consistency of the annual fold schedule.

    Checks (raises ValueError on failure):
    - required columns exist
    - train_fit_end_date < val_start_date
    - train_last_labeled_date <= train_fit_end_date
    - val_start_date <= val_end_date
    - val_last_labeled_date <= val_end_date
    - validation windows do not overlap (monotonic by year)

    Returns a sorted copy (by val_year).
    """
    if fold_schedule is None or fold_schedule.empty:
        raise ValueError("fold_schedule is empty")

    fs = fold_schedule.copy()

    required = [
        "val_year",
        "train_fit_end_date",
        "train_last_labeled_date",
        "val_start_date",
        "val_end_date",
        "val_last_labeled_date",
    ]
    missing = [c for c in required if c not in fs.columns]
    if missing:
        raise ValueError(f"fold_schedule missing required columns: {missing}")

    for c in [
        "train_fit_end_date",
        "train_last_labeled_date",
        "val_start_date",
        "val_end_date",
        "val_last_labeled_date",
    ]:
        fs[c] = pd.to_datetime(fs[c], errors="coerce")

    fs = fs.sort_values(["val_year"]).reset_index(drop=True)

    if not (fs["train_fit_end_date"] < fs["val_start_date"]).all():
        raise ValueError("Invalid schedule: train_fit_end_date must be < val_start_date for all folds.")
    if not (fs["train_last_labeled_date"] <= fs["train_fit_end_date"]).all():
        raise ValueError("Invalid schedule: train_last_labeled_date must be <= train_fit_end_date.")
    if not (fs["val_start_date"] <= fs["val_end_date"]).all():
        raise ValueError("Invalid schedule: val_start_date must be <= val_end_date.")
    if not (fs["val_last_labeled_date"] <= fs["val_end_date"]).all():
        raise ValueError("Invalid schedule: val_last_labeled_date must be <= val_end_date.")

    # No overlap across folds
    prev_end = None
    for _, r in fs.iterrows():
        if prev_end is not None:
            if pd.to_datetime(r["val_start_date"]) <= pd.to_datetime(prev_end):
                raise ValueError("Invalid schedule: validation windows overlap.")
        prev_end = r["val_end_date"]

    return fs

def build_final_train_fold_from_schedule(
    fold_schedule: pd.DataFrame,
    trading_dates: Sequence[Union[str, pd.Timestamp]],
    horizon_td: int = 22,
) -> pd.Series:
    """
    Build a "final training fold" for inference/deployment AFTER the last validation year.

    If your last validated year is Y (i.e. schedule ends at val_year=Y), the deploy model for year Y+1
    would be trained "as-of" the end of year Y, meaning:

      train_fit_end_date      = last trading day of year Y (== last fold's val_end_date)
      train_last_labeled_date = shift_trading_days(train_fit_end_date, -horizon_td)

    Validation window is not available inside historical data, so val_* are returned as NaT unless
    the trading calendar contains those dates.

    Returns: pd.Series with the same columns as build_annual_fold_schedule, plus is_final_fold=True.
    """
    if fold_schedule is None or fold_schedule.empty:
        raise ValueError("fold_schedule is empty")

    fs = fold_schedule.sort_values("val_year").reset_index(drop=True).copy()
    last_row = fs.iloc[-1]

    dates = _normalize_trading_dates(trading_dates)
    if len(dates) == 0:
        raise ValueError("Empty trading_dates")

    last_val_year = int(last_row["val_year"])
    deploy_year = int(last_val_year + 1)

    # Train ends at the end of the last validated year (Y)
    train_fit_end_date = pd.to_datetime(last_row.get("val_end_date"), errors="coerce")
    if pd.isna(train_fit_end_date):
        # Fallback: last trading date of last_val_year (if available)
        mask_y = dates.year == last_val_year
        train_fit_end_date = pd.Timestamp(dates[mask_y][-1]) if mask_y.any() else pd.Timestamp(dates[-1])

    train_last_labeled_date = shift_trading_days(dates, train_fit_end_date, -horizon_td)

    # Global last labeled date based on the available dataset max date
    global_last_labeled_date = shift_trading_days(dates, pd.Timestamp(dates[-1]), -horizon_td)

    # val window for deploy_year might not exist in the calendar
    mask_deploy = dates.year == deploy_year
    if mask_deploy.any():
        val_start_date = pd.Timestamp(dates[mask_deploy][0])
        val_end_date = pd.Timestamp(dates[mask_deploy][-1])
    else:
        val_start_date = pd.NaT
        val_end_date = pd.NaT

    train_start_date = pd.to_datetime(last_row.get("train_start_date"), errors="coerce")
    if pd.isna(train_start_date):
        train_start_date = pd.Timestamp(dates[0])

    out = dict(
        fold_id=int(deploy_year),
        val_year=int(deploy_year),
        horizon_td=int(horizon_td),
        train_start_date=train_start_date,
        train_fit_end_date=train_fit_end_date,
        train_last_labeled_date=train_last_labeled_date,
        val_start_date=val_start_date,
        val_end_date=val_end_date,
        val_last_labeled_date=pd.NaT,
        global_last_labeled_date=global_last_labeled_date,
        schedule_created_at=datetime.now(timezone.utc).isoformat(),
        is_final_fold=True,
        based_on_last_val_year=int(last_val_year),
    )
    return pd.Series(out)


# =============================================================================
# SQLite helpers (schema-safe)
# =============================================================================

_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sqlite_identifier(name: str) -> str:
    if not isinstance(name, str) or not _VALID_NAME_RE.match(name):
        raise ValueError(f"Invalid sqlite identifier: {name!r}")
    return name


def connect_sqlite(sqlite_path: str) -> sqlite3.Connection:
    if not isinstance(sqlite_path, str) or not sqlite_path:
        raise ValueError("sqlite_path must be a non-empty string")
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def get_table_info(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    table = _validate_sqlite_identifier(table)
    return pd.read_sql_query(f'PRAGMA table_info("{table}")', conn)


def _json_default(o: Any) -> Any:
    # JSON serialization for numpy/pandas types
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def _utc_now_sqlite_str() -> str:
    # Similar to SQLite datetime('now'): "YYYY-MM-DD HH:MM:SS"
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _infer_conflict_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    """
    Infer conflict columns for UPSERT.

    Heuristic (important for tables that have BOTH a surrogate PK and a business UNIQUE key):
    - If the table has at least one UNIQUE index/constraint, prefer the first UNIQUE index.
      (This avoids returning a surrogate INTEGER PRIMARY KEY like "pred_id", which you typically
       do not have when upserting.)
    - Otherwise, fall back to PRIMARY KEY columns.

    If you want strict control, pass conflict_cols explicitly to upsert/run_walk_forward.
    """
    table = _validate_sqlite_identifier(table)
    ti = get_table_info(conn, table)
    if ti.empty:
        raise ValueError(f"Table not found or has no columns: {table}")

    # 1) Prefer UNIQUE indexes/constraints (common for idempotent upserts)
    idx_list = pd.read_sql_query(f'PRAGMA index_list("{table}")', conn)
    if not idx_list.empty and "unique" in idx_list.columns:
        uniq = idx_list[idx_list["unique"].astype(int) == 1].copy()
        # Deterministic ordering: origin + name
        if "origin" in uniq.columns:
            uniq = uniq.sort_values(["origin", "name"])
        else:
            uniq = uniq.sort_values(["name"])
        for _, r in uniq.iterrows():
            idx_name = str(r["name"])
            idx_info = pd.read_sql_query(f'PRAGMA index_info("{idx_name}")', conn)
            if not idx_info.empty and "name" in idx_info.columns:
                cols = idx_info["name"].astype(str).tolist()
                if cols:
                    return cols

    # 2) Fallback: PRIMARY KEY
    pk = ti[ti["pk"].astype(int) > 0].sort_values("pk")
    if not pk.empty:
        return pk["name"].astype(str).tolist()

    raise ValueError(
        f"Cannot infer conflict cols for UPSERT in table {table}. "
        "Define a PRIMARY KEY or UNIQUE index, or pass conflict_cols explicitly."
    )

def upsert_dataframe(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    *,
    conflict_cols: Optional[Sequence[str]] = None,
    exclude_update_cols: Optional[Sequence[str]] = None,
    exclude_insert_cols: Optional[Sequence[str]] = None,
    chunk_size: int = 500,
) -> int:
    """
    Schema-safe UPSERT:
    - Inserts/updates only columns present both in df and in the table schema.
    - Idempotent when conflict_cols match a UNIQUE constraint / PK.
    - Avoids accidentally updating surrogate PKs (e.g. pred_id) by allowing exclusions.

    Returns number of rows attempted (inserted or updated).
    """
    if df is None or df.empty:
        return 0

    table = _validate_sqlite_identifier(table)
    ti = get_table_info(conn, table)
    if ti.empty:
        raise ValueError(f"Table not found or empty schema: {table}")

    table_cols = ti["name"].astype(str).tolist()

    df2 = df.copy()

    # Convert datetime columns in df2 to strings (keep user-provided formatting if already str)
    for c in df2.columns:
        if np.issubdtype(df2[c].dtype, np.datetime64):
            df2[c] = pd.to_datetime(df2[c], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")

    exclude_insert_cols = set(exclude_insert_cols or [])
    exclude_update_cols = set(exclude_update_cols or [])

    # Determine conflict columns
    conflict = list(conflict_cols) if conflict_cols is not None else _infer_conflict_cols(conn, table)
    conflict = [c for c in conflict if c in table_cols]
    if not conflict:
        raise ValueError(f"conflict_cols empty/invalid for table {table}: {conflict_cols}")

    # Heuristic: drop PK columns not used for conflict (common surrogate PK like pred_id)
    pk_cols = ti.loc[ti["pk"].astype(int) > 0, "name"].astype(str).tolist()
    for pk in pk_cols:
        if pk in conflict:
            continue
        if pk in df2.columns:
            # If df provides no meaningful values, drop it from insert/update.
            col = df2[pk]
            all_null = bool(col.isna().all())
            if all_null:
                exclude_insert_cols.add(pk)
                exclude_update_cols.add(pk)
            else:
                # Even if user provides values, updating a surrogate PK is almost never desired.
                exclude_update_cols.add(pk)

    # Columns to insert: intersection(df cols, table cols), excluding exclude_insert_cols
    insert_cols = [c for c in df2.columns if c in table_cols and c not in exclude_insert_cols]
    if not insert_cols:
        raise ValueError(f"No matching columns between df and table {table}")

    # Ensure all conflict cols are present in insert columns
    missing_conflict = [c for c in conflict if c not in insert_cols]
    if missing_conflict:
        raise ValueError(f"Missing conflict columns in df for table {table}: {missing_conflict}")

    update_cols = [c for c in insert_cols if c not in conflict and c not in exclude_update_cols]

    cols_sql = ", ".join([f'"{c}"' for c in insert_cols])
    placeholders = ", ".join(["?"] * len(insert_cols))
    conflict_sql = ", ".join([f'"{c}"' for c in conflict])

    if update_cols:
        update_sql = ", ".join([f'"{c}"=excluded."{c}"' for c in update_cols])
        sql = f'INSERT INTO "{table}" ({cols_sql}) VALUES ({placeholders}) ON CONFLICT({conflict_sql}) DO UPDATE SET {update_sql}'
    else:
        sql = f'INSERT INTO "{table}" ({cols_sql}) VALUES ({placeholders}) ON CONFLICT({conflict_sql}) DO NOTHING'

    df_ins = df2[insert_cols].copy()
    df_ins = df_ins.replace({np.nan: None})

    # Build rows as Python types
    values = []
    for row in df_ins.itertuples(index=False, name=None):
        cleaned = []
        for v in row:
            if isinstance(v, np.generic):
                v = v.item()
            cleaned.append(v)
        values.append(tuple(cleaned))

    cur = conn.cursor()
    total = 0
    for i in range(0, len(values), int(chunk_size)):
        chunk = values[i : i + int(chunk_size)]
        cur.executemany(sql, chunk)
        total += len(chunk)
    return total


# =============================================================================
# Walk-forward runner (model-agnostic)
# =============================================================================

def _as_naive_datetime(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, errors="coerce")
    # drop timezone if any
    if hasattr(x.dt, "tz") and x.dt.tz is not None:
        x = x.dt.tz_convert(None)
    return x


def _format_timestamp_for_id(ts: Union[pd.Series, pd.Timestamp], fmt: str) -> Union[pd.Series, str]:
    if isinstance(ts, pd.Series):
        x = _as_naive_datetime(ts)
        return x.dt.strftime(fmt)
    else:
        t = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t):
            return ""
        if getattr(t, "tzinfo", None) is not None:
            try:
                t = t.tz_convert(None)
            except Exception:
                t = pd.Timestamp(t).tz_convert(None)
        return pd.Timestamp(t).strftime(fmt)


def make_default_preds_schema_adapter(
    *,
    timestamp_format: str = "%Y-%m-%d",
    include_metadata: bool = True,
) -> Callable[..., pd.DataFrame]:
    """
    Default adapter for a simple predictions table like:

        CREATE TABLE PREDS_TABLE (
            pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_version TEXT,
            run_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            input_id TEXT NOT NULL,
            y_true REAL,
            y_pred REAL NOT NULL,
            prediction_time TEXT NOT NULL DEFAULT (datetime('now')),
            metadata TEXT,
            UNIQUE (model_name, model_version, run_id, input_id)
        )

    The adapter:
    - ensures required columns exist
    - formats timestamp consistently (timestamp_format)
    - builds input_id = symbol|timestamp
    - stores extra info (fold_id/horizon_td/etc) into `metadata` as JSON if include_metadata=True
    - DOES NOT include pred_id in output (keeps AUTOINCREMENT intact)
    """
    def adapter(
        preds_internal: pd.DataFrame,
        table_info: pd.DataFrame,
        *,
        kind: str,
        model_name: str,
        horizon_td: int,
        fold_row: pd.Series,
        run_meta: Dict[str, Any],
    ) -> pd.DataFrame:
        if kind != "preds":
            raise ValueError("make_default_preds_schema_adapter only supports kind='preds'")

        table_cols = table_info["name"].astype(str).tolist()

        df = preds_internal.copy()

        # Canonical expected columns from runner: symbol, timestamp, y_pred, (y_true optional)
        if "symbol" not in df.columns or "timestamp" not in df.columns or "y_pred" not in df.columns:
            raise ValueError("preds_internal must contain at least: ['symbol','timestamp','y_pred']")

        # Ensure strings
        df["symbol"] = df["symbol"].astype(str)

        # Timestamp formatted string (stable for input_id)
        df["timestamp"] = _format_timestamp_for_id(df["timestamp"], timestamp_format)

        # Fill tracking columns
        df["model_name"] = model_name
        model_version = (run_meta or {}).get("model_version")
        run_id = (run_meta or {}).get("run_id")
        if model_version is None or str(model_version).strip() == "":
            # Critical: model_version must not be NULL if it participates in UNIQUE constraints
            raise ValueError("run_meta['model_version'] must be a non-empty string (cannot be NULL).")
        if run_id is None or str(run_id).strip() == "":
            raise ValueError("run_meta['run_id'] must be a non-empty string.")

        df["model_version"] = str(model_version)
        df["run_id"] = str(run_id)

        # input_id deterministic
        df["input_id"] = df["symbol"].astype(str) + "|" + df["timestamp"].astype(str)

        # prediction_time optional (if column exists); useful to bump on reruns
        if "prediction_time" in table_cols and "prediction_time" not in df.columns:
            df["prediction_time"] = _utc_now_sqlite_str()

        # metadata JSON
        if "metadata" in table_cols:
            if include_metadata:
                base_meta = {
                    "fold_id": int(fold_row.get("fold_id", fold_row.get("val_year", -1))),
                    "val_year": int(fold_row.get("val_year", -1)) if pd.notna(fold_row.get("val_year", np.nan)) else None,
                    "horizon_td": int(horizon_td),
                    "train_fit_end_date": str(fold_row.get("train_fit_end_date")),
                    "train_last_labeled_date": str(fold_row.get("train_last_labeled_date")),
                    "val_start_date": str(fold_row.get("val_start_date")),
                    "val_end_date": str(fold_row.get("val_end_date")),
                    "val_last_labeled_date": str(fold_row.get("val_last_labeled_date")),
                }
                # Allow caller to pass an already-built metadata column with per-row dict/str
                if "metadata" in df.columns:
                    # Merge if dict-like
                    def _merge(m):
                        if m is None or (isinstance(m, float) and np.isnan(m)):
                            return base_meta
                        if isinstance(m, str):
                            try:
                                d = json.loads(m)
                                if isinstance(d, dict):
                                    out = dict(base_meta)
                                    out.update(d)
                                    return out
                            except Exception:
                                pass
                            out = dict(base_meta)
                            out["metadata_raw"] = m
                            return out
                        if isinstance(m, dict):
                            out = dict(base_meta)
                            out.update(m)
                            return out
                        out = dict(base_meta)
                        out["metadata_raw"] = str(m)
                        return out
                    meta_objs = df["metadata"].apply(_merge)
                else:
                    meta_objs = pd.Series([base_meta] * len(df))

                df["metadata"] = meta_objs.apply(lambda d: json.dumps(d, default=_json_default, ensure_ascii=False))
            else:
                df["metadata"] = None

        # Keep only columns that exist in table, EXCLUDING surrogate PK pred_id if present
        out_cols = [c for c in table_cols if c in df.columns and c != "pred_id"]
        out = df[out_cols].copy()

        # Basic type coercions
        if "y_pred" in out.columns:
            out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
        if "y_true" in out.columns:
            out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")

        # Enforce NOT NULL required columns where possible
        for req in ["model_name", "model_version", "run_id", "symbol", "timestamp", "input_id", "y_pred"]:
            if req in table_cols:
                if req not in out.columns:
                    raise ValueError(f"Adapter did not produce required column {req!r}")
                if out[req].isna().any():
                    bad = int(out[req].isna().sum())
                    raise ValueError(f"Adapter produced NULLs in required column {req!r}: {bad} rows")

        return out

    return adapter


def _ensure_metrics_dir(metrics_output_dir: Optional[str], run_meta: Dict[str, Any]) -> Optional[str]:
    if metrics_output_dir is None:
        return None
    run_id = (run_meta or {}).get("run_id", "run")
    out_dir = os.path.join(metrics_output_dir, str(run_id))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_walk_forward_oos_to_sqlite(
    *,
    model_name: str,
    horizon_td: int,
    fold_schedule: pd.DataFrame,
    build_train_dataset_fn: Callable[[pd.Series], Dict[str, Any]],
    build_infer_dataset_fn: Callable[[pd.Series], Dict[str, Any]],
    fit_model_fn: Callable[[Dict[str, Any], pd.Series], Any],
    predict_fn: Callable[[Any, Dict[str, Any], pd.Series], np.ndarray],
    sqlite_path: str,
    preds_table: str,
    schema_adapter: Callable[..., pd.DataFrame],
    run_meta: Dict[str, Any],
    eval_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    metrics_output_dir: Optional[str] = None,
    preds_conflict_cols: Optional[Sequence[str]] = None,
    preds_exclude_update_cols: Optional[Sequence[str]] = None,
    timestamp_format_for_ids: str = "%Y-%m-%d",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Annual walk-forward runner (model-agnostic).

    For each fold:
      - Train using cutoff train_last_labeled_date (dataset builder must enforce)
      - Predict for all timestamps in [val_start_date, val_end_date] (dataset builder must enforce)
      - UPSERT OOS predictions into SQLite
      - Evaluate only where label is available (timestamp <= val_last_labeled_date AND y_true finite)
      - Write fold metrics to JSON (NOT to SQLite)

    Returns a DataFrame with one row per fold and a column 'metrics_path' pointing to the JSON file (if enabled).
    """
    if fold_schedule is None or fold_schedule.empty:
        raise ValueError("fold_schedule is empty")
    if run_meta is None:
        run_meta = {}

    # Normalize & sort schedule
    fs = fold_schedule.copy()
    for c in [
        "train_start_date",
        "train_fit_end_date",
        "train_last_labeled_date",
        "val_start_date",
        "val_end_date",
        "val_last_labeled_date",
    ]:
        if c in fs.columns:
            fs[c] = pd.to_datetime(fs[c], errors="coerce")
    fs = fs.sort_values(["val_year"]).reset_index(drop=True)

    # Prepare metrics directory
    metrics_dir = _ensure_metrics_dir(metrics_output_dir, run_meta)

    # Connect DB and inspect schema
    conn = connect_sqlite(sqlite_path)
    try:
        preds_info = get_table_info(conn, preds_table)
        if preds_info.empty:
            raise ValueError(f"preds_table not found or empty schema: {preds_table}")

        # Default conflict cols if not supplied: infer from schema
        conflict_cols = list(preds_conflict_cols) if preds_conflict_cols is not None else _infer_conflict_cols(conn, preds_table)

        metrics_rows: List[Dict[str, Any]] = []

        for _, row in fs.iterrows():
            fold_id = int(row.get("fold_id", row.get("val_year")))
            if verbose:
                print(f"\n=== Fold {fold_id} (val_year={row.get('val_year')}) ===")
                print("train_fit_end_date:", row.get("train_fit_end_date"))
                print("train_last_labeled_date:", row.get("train_last_labeled_date"))
                print(
                    "val window:",
                    row.get("val_start_date"),
                    "->",
                    row.get("val_end_date"),
                    "eval_until:",
                    row.get("val_last_labeled_date"),
                )

            # 1) Build train dataset
            train_data = build_train_dataset_fn(row)
            # Optional sanity checks
            if isinstance(train_data, dict) and "meta" in train_data and isinstance(train_data["meta"], pd.DataFrame):
                if "timestamp" in train_data["meta"].columns and pd.notna(row.get("train_last_labeled_date", pd.NaT)):
                    max_ts = pd.to_datetime(train_data["meta"]["timestamp"], errors="coerce").max()
                    if pd.notna(max_ts) and max_ts > pd.to_datetime(row["train_last_labeled_date"]):
                        raise ValueError(
                            f"Train dataset leakage: max timestamp {max_ts} > train_last_labeled_date {row['train_last_labeled_date']}"
                        )

            # 2) Fit model
            model_obj = fit_model_fn(train_data, row)

            # 3) Build inference dataset (full val window)
            infer_data = build_infer_dataset_fn(row)
            if not isinstance(infer_data, dict):
                raise ValueError("build_infer_dataset_fn must return a dict")
            meta = infer_data.get("meta")
            if meta is None or not isinstance(meta, pd.DataFrame):
                raise ValueError("build_infer_dataset_fn must return dict with meta: pd.DataFrame")
            if "timestamp" not in meta.columns or "symbol" not in meta.columns:
                raise ValueError("infer meta must include columns: ['timestamp','symbol']")

            # 4) Predict
            y_pred = np.asarray(predict_fn(model_obj, infer_data, row)).reshape(-1)
            if len(y_pred) != len(meta):
                raise ValueError(f"predict_fn returned len={len(y_pred)} but meta len={len(meta)}")

            # 5) Assemble internal preds df (canonical)
            preds_internal = meta.copy()
            preds_internal["y_pred"] = y_pred
            if "y" in infer_data:
                y_true = np.asarray(infer_data["y"]).reshape(-1)
                if len(y_true) == len(meta):
                    preds_internal["y_true"] = y_true

            preds_internal["model_name"] = model_name
            preds_internal["horizon_td"] = int(horizon_td)
            preds_internal["fold_id"] = fold_id

            # Attach run_meta columns for adapter convenience
            for k, v in (run_meta or {}).items():
                preds_internal[k] = v

            # Normalize timestamp before adapter (helps consistent metadata and input_id)
            preds_internal["timestamp"] = _format_timestamp_for_id(preds_internal["timestamp"], timestamp_format_for_ids)

            # 6) Adapter -> exact table schema
            preds_out = schema_adapter(
                preds_internal,
                preds_info,
                kind="preds",
                model_name=model_name,
                horizon_td=horizon_td,
                fold_row=row,
                run_meta=run_meta,
            )

            # 7) UPSERT preds (idempotent)
            n_written = upsert_dataframe(
                conn,
                preds_table,
                preds_out,
                conflict_cols=conflict_cols,
                exclude_update_cols=preds_exclude_update_cols,
                # Important: don't insert pred_id even if adapter accidentally includes it
                exclude_insert_cols=["pred_id"],
            )
            conn.commit()
            if verbose:
                print("pred rows upserted:", n_written)

            # 8) Evaluate metrics subset and write JSON
            metrics_path = None
            metrics_dict: Dict[str, Any] = {}
            n_eval = 0

            if eval_fn is not None and "y_true" in preds_internal.columns:
                ts = pd.to_datetime(preds_internal["timestamp"], errors="coerce")
                y_true_all = pd.to_numeric(preds_internal["y_true"], errors="coerce").to_numpy(dtype=float)
                y_pred_all = pd.to_numeric(preds_internal["y_pred"], errors="coerce").to_numpy(dtype=float)

                val_last = row.get("val_last_labeled_date")
                if pd.isna(val_last):
                    val_last = ts.max()

                mask = (ts <= pd.to_datetime(val_last)) & np.isfinite(y_true_all) & np.isfinite(y_pred_all)
                n_eval = int(mask.sum())

                if n_eval > 0:
                    meta_eval = preds_internal.loc[mask, ["timestamp", "symbol"]].copy()
                    metrics_dict = eval_fn(
                        y_true_all[mask],
                        y_pred_all[mask],
                        meta=meta_eval,
                        time_col="timestamp",
                        group_col="symbol",
                    )
                else:
                    metrics_dict = {"N": 0}

                if metrics_dir is not None:
                    metrics_payload = {
                        "model_name": model_name,
                        "model_version": run_meta.get("model_version"),
                        "run_id": run_meta.get("run_id"),
                        "horizon_td": int(horizon_td),
                        "fold_id": fold_id,
                        "fold_row": {k: (str(v) if not pd.isna(v) else None) for k, v in row.items()},
                        "n_pred_rows": int(len(preds_internal)),
                        "n_eval_rows": int(n_eval),
                        "metrics": metrics_dict,
                        "written_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    metrics_path = os.path.join(metrics_dir, f"fold_{fold_id}_metrics.json")
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(metrics_payload, f, indent=2, ensure_ascii=False, default=_json_default)

                    if verbose:
                        print("metrics json:", metrics_path)

            metrics_rows.append(
                {
                    "fold_id": fold_id,
                    "val_year": int(row.get("val_year")) if pd.notna(row.get("val_year", np.nan)) else None,
                    "train_fit_end_date": row.get("train_fit_end_date"),
                    "train_last_labeled_date": row.get("train_last_labeled_date"),
                    "val_start_date": row.get("val_start_date"),
                    "val_end_date": row.get("val_end_date"),
                    "val_last_labeled_date": row.get("val_last_labeled_date"),
                    "n_pred_rows": int(len(preds_internal)),
                    "n_eval_rows": int(n_eval),
                    "metrics_path": metrics_path,
                }
            )

        # Also write an aggregated metrics index JSON for convenience
        if metrics_dir is not None:
            idx_path = os.path.join(metrics_dir, "metrics_index.json")
            with open(idx_path, "w", encoding="utf-8") as f:
                json.dump(metrics_rows, f, indent=2, ensure_ascii=False, default=_json_default)
            if verbose:
                print("metrics index:", idx_path)

        return pd.DataFrame(metrics_rows)

    finally:
        conn.close()
