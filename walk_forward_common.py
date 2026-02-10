
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# --------------------------
# Trading-day utilities
# --------------------------
def _normalize_trading_dates(trading_dates: Sequence[Union[str, pd.Timestamp]]) -> pd.DatetimeIndex:
    dt = pd.to_datetime(pd.Index(list(trading_dates)), errors="coerce")
    dt = dt[dt.notna()]
    # Drop tz for consistent comparisons
    if getattr(dt, "tz", None) is not None:
        dt = dt.tz_convert(None)
    else:
        # if individual timestamps are tz-aware, normalize via to_datetime again
        dt = pd.to_datetime(dt, errors="coerce").tz_localize(None) if hasattr(dt, "tz_localize") else pd.to_datetime(dt)
    dt = pd.DatetimeIndex(sorted(dt.unique()))
    return dt


def shift_trading_days(
    trading_dates: Sequence[Union[str, pd.Timestamp]],
    anchor_date: Union[str, pd.Timestamp],
    offset: int,
) -> pd.Timestamp:
    """
    Shift an anchor date by N trading days using the supplied trading calendar.

    - If anchor_date is not an exact trading date, it is "padded" to the last trading date <= anchor_date.
    - Raises if shift goes out of bounds.
    """
    dates = _normalize_trading_dates(trading_dates)
    if len(dates) == 0:
        raise ValueError("Empty trading_dates")

    anchor = pd.to_datetime(anchor_date)
    if getattr(anchor, "tzinfo", None) is not None:
        anchor = anchor.tz_convert(None)

    # pad to last date <= anchor
    pos = dates.searchsorted(anchor, side="right") - 1
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
    Build annual walk-forward schedule:

    For validation year Y:
      - train_fit_end_date = last trading day of year (Y-1)
      - train_last_labeled_date = shift_trading_days(train_fit_end_date, -horizon_td)
      - val_start_date = first trading day of year Y
      - val_end_date = last trading day of year Y
      - global_last_labeled_date = shift_trading_days(max_date_in_dataset, -horizon_td)
      - val_last_labeled_date = min(val_end_date, global_last_labeled_date)

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

        # skip folds where even the last train label is earlier than the configured train_start_date
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

    # Ensure types
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


# --------------------------
# SQLite helpers (schema-safe)
# --------------------------
_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sqlite_identifier(name: str) -> str:
    if not isinstance(name, str) or not _VALID_NAME_RE.match(name):
        raise ValueError(f"Invalid sqlite identifier: {name!r}")
    return name


def get_table_info(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    table = _validate_sqlite_identifier(table)
    return pd.read_sql_query(f'PRAGMA table_info("{table}")', conn)


def _infer_conflict_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    table = _validate_sqlite_identifier(table)
    ti = get_table_info(conn, table)
    if ti.empty:
        raise ValueError(f"Table not found or has no columns: {table}")

    # Prefer PK
    pk = ti[ti["pk"].astype(int) > 0].copy()
    if not pk.empty:
        pk = pk.sort_values("pk")
        return pk["name"].astype(str).tolist()

    # Else first UNIQUE index
    idx_list = pd.read_sql_query(f'PRAGMA index_list("{table}")', conn)
    if not idx_list.empty and "unique" in idx_list.columns:
        uniq = idx_list[idx_list["unique"].astype(int) == 1]
        if not uniq.empty:
            idx_name = str(uniq.iloc[0]["name"])
            idx_info = pd.read_sql_query(f'PRAGMA index_info("{idx_name}")', conn)
            if not idx_info.empty and "name" in idx_info.columns:
                cols = idx_info["name"].astype(str).tolist()
                if cols:
                    return cols

    raise ValueError(
        f"Cannot infer conflict cols for UPSERT in table {table}. "
        "Define a PRIMARY KEY or UNIQUE index, or handle conflict_cols externally."
    )


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


def upsert_dataframe(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    conflict_cols: Optional[Sequence[str]] = None,
    *,
    chunk_size: int = 500,
    exclude_update_cols: Optional[Sequence[str]] = None,
) -> int:
    """
    UPSERT df into SQLite table. Does NOT change schema.

    - conflict_cols: columns used in ON CONFLICT(...) target. If None, inferred from PK/UNIQUE.
    - exclude_update_cols: columns that should NOT be updated on conflict.
    """
    table = _validate_sqlite_identifier(table)
    if df is None or df.empty:
        return 0

    ti = get_table_info(conn, table)
    table_cols = ti["name"].astype(str).tolist()

    # Keep only table columns (adapter should already do this)
    use_cols = [c for c in table_cols if c in df.columns]
    if not use_cols:
        raise ValueError(f"No matching columns between df and table {table}")

    df2 = df[use_cols].copy()

    # Convert timestamps to ISO strings (SQLite-friendly)
    for c in df2.columns:
        if np.issubdtype(df2[c].dtype, np.datetime64):
            df2[c] = pd.to_datetime(df2[c], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%dT%H:%M:%S")

    conflict = list(conflict_cols) if conflict_cols is not None else _infer_conflict_cols(conn, table)
    conflict = [c for c in conflict if c in use_cols]
    if not conflict:
        raise ValueError(f"Conflict cols empty after intersection with df columns for table {table}")

    exclude_update_cols = set(exclude_update_cols or [])
    update_cols = [c for c in use_cols if c not in conflict and c not in exclude_update_cols]

    cols_sql = ", ".join([f'"{c}"' for c in use_cols])
    placeholders = ", ".join(["?"] * len(use_cols))
    conflict_sql = ", ".join([f'"{c}"' for c in conflict])

    if update_cols:
        update_sql = ", ".join([f'"{c}"=excluded."{c}"' for c in update_cols])
        sql = f'INSERT INTO "{table}" ({cols_sql}) VALUES ({placeholders}) ON CONFLICT({conflict_sql}) DO UPDATE SET {update_sql}'
    else:
        sql = f'INSERT INTO "{table}" ({cols_sql}) VALUES ({placeholders}) ON CONFLICT({conflict_sql}) DO NOTHING'

    # Prepare values
    vals = []
    for row in df2.itertuples(index=False, name=None):
        cleaned = []
        for v in row:
            if isinstance(v, (np.generic,)):
                v = v.item()
            cleaned.append(v)
        vals.append(tuple(cleaned))

    # Execute in chunks
    cur = conn.cursor()
    total = 0
    for i in range(0, len(vals), int(chunk_size)):
        chunk = vals[i : i + int(chunk_size)]
        cur.executemany(sql, chunk)
        total += len(chunk)
    return total


# --------------------------
# Walk-forward runner (generic)
# --------------------------
def run_walk_forward_oos_to_sqlite(
    model_name: str,
    horizon_td: int,
    fold_schedule: pd.DataFrame,
    build_train_dataset_fn: Callable[[pd.Series], Dict[str, Any]],
    build_infer_dataset_fn: Callable[[pd.Series], Dict[str, Any]],
    fit_model_fn: Callable[[Dict[str, Any], pd.Series], Any],
    predict_fn: Callable[[Any, Dict[str, Any], pd.Series], np.ndarray],
    sqlite_path: str,
    preds_table: str,
    metrics_table: str,
    schema_adapter: Callable[..., pd.DataFrame],
    eval_fn: Callable[..., Dict[str, Any]],
    run_meta: Dict[str, Any],
    *,
    preds_exclude_update_cols: Optional[Sequence[str]] = None,
    metrics_exclude_update_cols: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For each fold:
      - Train using cutoff train_last_labeled_date (and target_timestamp <= train_fit_end_date inside dataset builder)
      - Predict for all timestamps in [val_start_date, val_end_date]
      - UPSERT OOS predictions
      - Evaluate only where label is available (timestamp <= val_last_labeled_date and y_true finite)
      - UPSERT metrics per fold

    Returns: metrics DataFrame (one row per fold, internal format).
    """
    if fold_schedule is None or fold_schedule.empty:
        raise ValueError("fold_schedule is empty")

    fs = fold_schedule.copy()

    # Normalize datetime cols if present
    dt_cols = [
        "train_start_date",
        "train_fit_end_date",
        "train_last_labeled_date",
        "val_start_date",
        "val_end_date",
        "val_last_labeled_date",
    ]
    for c in dt_cols:
        if c in fs.columns:
            fs[c] = pd.to_datetime(fs[c], errors="coerce")

    fs = fs.sort_values(["val_year"]).reset_index(drop=True)

    preds_table = preds_table
    metrics_table = metrics_table

    metrics_rows: List[Dict[str, Any]] = []

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        preds_info = get_table_info(conn, preds_table)
        metrics_info = get_table_info(conn, metrics_table)
        if preds_info.empty:
            raise ValueError(f"preds_table not found or empty schema: {preds_table}")
        if metrics_info.empty:
            raise ValueError(f"metrics_table not found or empty schema: {metrics_table}")

        for i, row in fs.iterrows():
            fold_id = row.get("fold_id", row.get("val_year", i))
            if verbose:
                print(f"\n=== Fold {fold_id} (val_year={row.get('val_year')}) ===")
                print("train_fit_end_date:", row.get("train_fit_end_date"))
                print("train_last_labeled_date:", row.get("train_last_labeled_date"))
                print("val:", row.get("val_start_date"), "->", row.get("val_end_date"), "eval_until:", row.get("val_last_labeled_date"))

            # 1) Build train dataset
            train_data = build_train_dataset_fn(row)
            # 2) Fit model
            model_obj = fit_model_fn(train_data, row)

            # 3) Build inference dataset (full val window)
            infer_data = build_infer_dataset_fn(row)
            meta = infer_data.get("meta")
            if meta is None or not isinstance(meta, pd.DataFrame):
                raise ValueError("build_infer_dataset_fn must return dict with meta: pd.DataFrame")
            if "timestamp" not in meta.columns or "symbol" not in meta.columns:
                raise ValueError("infer meta must include columns: ['timestamp','symbol']")

            # 4) Predict
            y_pred = predict_fn(model_obj, infer_data, row)
            y_pred = np.asarray(y_pred).reshape(-1)
            if len(y_pred) != len(meta):
                raise ValueError(f"predict_fn returned len={len(y_pred)} but meta len={len(meta)}")

            # 5) Assemble internal preds df (canonical)
            preds_internal = meta.copy()
            preds_internal["y_pred"] = y_pred
            # include y_true if present (optional)
            if "y" in infer_data:
                y_true = np.asarray(infer_data["y"]).reshape(-1)
                if len(y_true) == len(meta):
                    preds_internal["y_true"] = y_true

            preds_internal["model_name"] = model_name
            preds_internal["horizon_td"] = int(horizon_td)
            preds_internal["fold_id"] = fold_id
            # attach run_meta to every row
            for k, v in (run_meta or {}).items():
                preds_internal[k] = v

            # 6) Adapter -> table schema exact
            preds_out = schema_adapter(
                preds_internal,
                preds_info,
                kind="preds",
                model_name=model_name,
                horizon_td=horizon_td,
                fold_row=row,
                run_meta=run_meta,
            )

            # 7) UPSERT preds
            n_written_preds = upsert_dataframe(
                conn,
                preds_table,
                preds_out,
                conflict_cols=None,
                exclude_update_cols=preds_exclude_update_cols,
            )
            conn.commit()
            if verbose:
                print("pred rows upserted:", n_written_preds)

            # 8) Evaluation subset (labels available)
            # Mask: timestamp <= val_last_labeled_date AND y_true finite
            metrics_dict: Dict[str, Any]
            eval_n = 0
            if "y" in infer_data:
                y_true_all = np.asarray(infer_data["y"]).reshape(-1)
                ts = pd.to_datetime(meta["timestamp"], errors="coerce")
                val_last = row.get("val_last_labeled_date")
                if pd.isna(val_last):
                    val_last = ts.max()

                m = (ts <= pd.to_datetime(val_last)) & np.isfinite(y_true_all) & np.isfinite(y_pred)
                eval_n = int(np.sum(m))
                if eval_n > 0:
                    meta_eval = meta.loc[m].copy()
                    metrics_dict = eval_fn(
                        y_true_all[m],
                        y_pred[m],
                        meta=meta_eval,
                        time_col="timestamp",
                        group_col="symbol",
                    )
                else:
                    metrics_dict = {"N": 0}
            else:
                metrics_dict = {"N": 0}

            # 9) Assemble internal metrics row
            metrics_internal = {
                "model_name": model_name,
                "horizon_td": int(horizon_td),
                "fold_id": fold_id,
                "val_year": row.get("val_year"),
                "train_fit_end_date": row.get("train_fit_end_date"),
                "train_last_labeled_date": row.get("train_last_labeled_date"),
                "val_start_date": row.get("val_start_date"),
                "val_end_date": row.get("val_end_date"),
                "val_last_labeled_date": row.get("val_last_labeled_date"),
                "n_eval": eval_n,
                "metrics_json": json.dumps(metrics_dict, default=_json_default),
            }
            for k, v in (run_meta or {}).items():
                metrics_internal[k] = v

            # Also expose top-level metrics keys for adapter to map if desired
            # (do not assume columns exist; adapter/table schema will decide)
            for k, v in metrics_dict.items():
                # avoid collisions
                if k not in metrics_internal:
                    metrics_internal[k] = v

            metrics_internal_df = pd.DataFrame([metrics_internal])

            metrics_out = schema_adapter(
                metrics_internal_df,
                metrics_info,
                kind="metrics",
                model_name=model_name,
                horizon_td=horizon_td,
                fold_row=row,
                run_meta=run_meta,
            )

            n_written_metrics = upsert_dataframe(
                conn,
                metrics_table,
                metrics_out,
                conflict_cols=None,
                exclude_update_cols=metrics_exclude_update_cols,
            )
            conn.commit()
            if verbose:
                print("metrics rows upserted:", n_written_metrics)

            metrics_rows.append(metrics_internal)

    return pd.DataFrame(metrics_rows)
