# data_collectors.py

from collections import defaultdict
import sqlite3
from typing import Sequence, Optional, Union, List, Any, Mapping, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Kept for backwards-compatibility with your existing imports.
# (If you don't use it anymore, you can remove it.)
from python_scripts.LLM_analysis.preprocess_store_database import get_connection  # noqa: F401

TARGET_HORIZONS = [1, 5, 20, 60]
TARGET_LOOKBACKS = [30, 60, 120, 252]


def _placeholders(n: int) -> str:
    return ",".join(["?"] * n)


def _to_iso_datetime(x: Union[str, pd.Timestamp]) -> str:
    """ISO string for SQLite TEXT columns that store timestamps."""
    return pd.to_datetime(x).isoformat(sep=" ")


def _to_iso_date(x: Union[str, pd.Timestamp]) -> str:
    """ISO date string for SQLite TEXT columns that store date-only values."""
    return pd.to_datetime(x).date().isoformat()


def fetch_stock_bars(
    conn: sqlite3.Connection,
    symbols: Sequence[str],
    timeframe: str,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    where = [f"symbol IN ({_placeholders(len(symbols))})", "timeframe = ?"]
    params: List[Any] = [*symbols, timeframe]

    if start is not None:
        where.append("timestamp >= ?")
        params.append(_to_iso_datetime(start))
    if end is not None:
        where.append("timestamp <= ?")
        params.append(_to_iso_datetime(end))

    sql = f"""
    SELECT *
    FROM stock_bars
    WHERE {' AND '.join(where)}
    ORDER BY symbol, timestamp
    """
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["timestamp"])
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def fetch_stock_indicators(
    conn: sqlite3.Connection,
    indicator_names: Sequence[str],
    symbols: Sequence[str],
    timeframe: str,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    where = [f"symbol IN ({_placeholders(len(symbols))})", "timeframe = ?"]
    params: List[Any] = [*symbols, timeframe]


    if start is not None:
        where.append("timestamp >= ?")
        params.append(_to_iso_datetime(start))
    if end is not None:
        where.append("timestamp <= ?")
        params.append(_to_iso_datetime(end))

    sql = f"""
    SELECT timestamp, symbol, timeframe, "{'", "'.join(indicator_names)}"
    FROM stock_indicators
    WHERE {' AND '.join(where)}
    ORDER BY symbol, timestamp
    """
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["timestamp"])
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def fetch_economic_indicators(
    conn: sqlite3.Connection,
    indicator_names: Sequence[str],
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    if not indicator_names:
        return pd.DataFrame()

    where = [f"indicator_name IN ({_placeholders(len(indicator_names))})"]
    params: List[Any] = [*indicator_names]

    if start is not None:
        where.append("date >= ?")
        params.append(_to_iso_date(start))
    if end is not None:
        where.append("date <= ?")
        params.append(_to_iso_date(end))

    sql = f"""
    SELECT *
    FROM economic_indicators
    WHERE {' AND '.join(where)}
    ORDER BY date
    """
    df = pd.read_sql_query(sql, conn, params=params, parse_dates=["date"])
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Joins the different indicators in a date into the same row with various columns (one per indicator_name)
    wide = (
        df.pivot_table(index="date", columns="indicator_name", values="value", aggfunc="last")  # type: ignore
        .sort_index()
    )
    return wide


# ---------------------------------------------------------------------
# FinancialModelingPrep features (fmp_features table)
# ---------------------------------------------------------------------

def fetch_fmp_features(
    conn: sqlite3.Connection,
    symbols: Sequence[str],
    feature_names: Optional[Sequence[str]] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Returns LONG df with columns: symbol, asof_date, feature, value, fetch_date
    from the `fmp_features` table.
    """
    if not symbols:
        return pd.DataFrame()

    where = [f"symbol IN ({_placeholders(len(symbols))})"]
    params: List[Any] = [*symbols]

    if feature_names is not None:
        if len(feature_names) == 0:
            return pd.DataFrame()
        where.append(f"feature IN ({_placeholders(len(feature_names))})")
        params.extend(feature_names)

    # NOTE: asof_date is TEXT; typically 'YYYY-MM-DD'
    if start is not None:
        where.append("asof_date >= ?")
        params.append(_to_iso_date(start))
    if end is not None:
        where.append("asof_date <= ?")
        params.append(_to_iso_date(end))

    sql = f"""
    SELECT symbol, asof_date, feature, value, fetch_date
    FROM fmp_features
    WHERE {' AND '.join(where)}
    ORDER BY symbol, asof_date
    """
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return pd.DataFrame()

    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")
    df["fetch_date"] = pd.to_datetime(df["fetch_date"], errors="coerce")
    return df


def fetch_fmp_features_wide(
    conn: sqlite3.Connection,
    symbols: Sequence[str],
    feature_names: Optional[Sequence[str]] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    prefix: str = "fmp_",
) -> pd.DataFrame:
    """
    Returns WIDE df with columns:
      symbol, asof_date, {prefix}{feature_1}, {prefix}{feature_2}, ...
    suitable for merge_asof into bars.
    """
    long = fetch_fmp_features(conn, symbols, feature_names=feature_names, start=start, end=end)
    if long.empty:
        return pd.DataFrame()
    
    long["asof_date"] = pd.to_datetime(long["asof_date"], errors="coerce")
    long["fetch_date"] = pd.to_datetime(long["fetch_date"], errors="coerce")

    long = long.sort_values(["symbol", "asof_date", "feature", "fetch_date"])

    # only the last version per key
    long = long.drop_duplicates(["symbol", "asof_date", "feature"], keep="last")

    wide = (
        long.pivot_table(
            index=["symbol", "asof_date"],
            columns="feature",
            values="value",
            aggfunc="last",  # type: ignore
        )
        .sort_index()
        .reset_index()
    )

    # Optional prefix to avoid collisions with other columns
    if prefix:
        wide.columns = ["symbol", "asof_date"] + [f"{prefix}{c}" for c in wide.columns[2:]]
    else:
        wide.columns = [str(c) for c in wide.columns]

    return wide


def build_ml_dataframe(
    conn: sqlite3.Connection,
    symbols: Sequence[str],
    timeframe: str,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    econ_indicator_names: Optional[Sequence[str]] = None,
    include_indicators: bool = False,
    indicator_names: Optional[Sequence[str]] = None,
    include_econ: bool = False,
    include_fmp: bool = False,
    fmp_feature_names: Optional[Sequence[str]] = None,
    fmp_prefix: str = "fmp_",
    keep_fmp_asof_date: bool = False,

) -> pd.DataFrame:
    """
    Convenience function: bars + (optional) tech indicators + (optional) economic indicators + (optional) FMP fundamentals.

    Returns long DataFrame with: symbol, timestamp, timeframe, OHLCV, indicators, econ..., fmp...
    """

    bars = fetch_stock_bars(conn, symbols, timeframe, start, end)
    if bars.empty:
        return pd.DataFrame()

    df = bars.copy()

    if include_indicators and indicator_names is not None:
        ind = fetch_stock_indicators(conn, indicator_names, symbols, timeframe, start, end)
        if not ind.empty:
            df = df.merge(ind, on=["symbol", "timestamp", "timeframe"], how="left", suffixes=("", "_ind"))

    if include_econ and econ_indicator_names is not None:
        econ = fetch_economic_indicators(conn, econ_indicator_names, start, end)
        if not econ.empty:
            econ_reset = econ.reset_index().rename(columns={"date": "econ_date"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            econ_reset["econ_date"] = pd.to_datetime(econ_reset["econ_date"], errors="coerce", utc=True)

            # merge_asof uses last known date <= timestamp
            df = df.sort_values(["symbol", "timestamp"])
            econ_reset = econ_reset.sort_values("econ_date")
            df = df[df["timestamp"].notna()]
            econ_reset = econ_reset[econ_reset["econ_date"].notna()]


            df = (
                pd.merge_asof(
                    df.sort_values("timestamp"),
                    econ_reset,
                    left_on="timestamp",
                    right_on="econ_date",
                    direction="backward",
                )
                .drop(columns=["econ_date"])
                .sort_values(["symbol", "timestamp"])
                .reset_index(drop=True)
            )

    if include_fmp:
        # IMPORTANT: don't cut by `start` by default; otherwise you can miss the last snapshot before `start`
        # which is needed for merge_asof to fill the first bars.
        fmp = fetch_fmp_features_wide(
            conn,
            symbols,
            feature_names=fmp_feature_names,
            start=None,
            end=end,
            prefix=fmp_prefix,
        )
        if not fmp.empty:
            df = df.sort_values(["symbol", "timestamp"])
            fmp = fmp.sort_values(["symbol", "asof_date"])

            merged = pd.merge_asof(
                df,
                fmp,
                by="symbol",
                left_on="timestamp",
                right_on="asof_date",
                direction="backward",
                allow_exact_matches=True,
            )
            if keep_fmp_asof_date:
                merged = merged.rename(columns={"asof_date": f"{fmp_prefix}asof_date"})
            else:
                merged = merged.drop(columns=["asof_date"])
            df = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# Supervised dataset creation
# ---------------------------------------------------------------------

def make_log_return_target(
    df: pd.DataFrame,
    horizon: int,
    price_col: str = "close",
    group_col: str = "symbol",
) -> pd.Series:
    """
    Log-return target with optional grouping by symbol (safe for multi-symbol dfs).
    """
    if group_col in df.columns:
        future = df.groupby(group_col, sort=False)[price_col].shift(-horizon)
    else:
        future = df[price_col].shift(-horizon)

    ratio = (future / df[price_col]).to_numpy()
    y = pd.Series(np.log(ratio), index=df.index, name=f"logret_t+{horizon}")
    return y


def make_lagged_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    group_col: str = "symbol",
) -> pd.DataFrame:
    """
    Lagged features with optional grouping by symbol (safe for multi-symbol dfs).
    """
    blocks = []
    if group_col in df.columns:
        gb = df.groupby(group_col, sort=False)
        for lag in range(lookback):
            block = gb[feature_cols].shift(lag).copy()
            block.columns = [f"{c}_lag{lag}" for c in feature_cols]
            blocks.append(block)
    else:
        for lag in range(lookback):
            block = df[feature_cols].shift(lag).copy()
            block.columns = [f"{c}_lag{lag}" for c in feature_cols]
            blocks.append(block)

    return pd.concat(blocks, axis=1)

LagSpec = Union[int, Sequence[int]]  # int => "n_lags", list[int] => explicit lags


def _normalize_lag_spec(spec: LagSpec) -> list[int]:
    """
    Normalize a per-feature lag specification into an explicit sorted list of lags.

    Rules:
      - int < 0  => exclude (returns empty list)
      - int 0 or 1 => [0]  (flag "no lag": include only lag0)
      - int n >= 2 => [0, 1, ..., n-1]
      - sequence of ints => sorted unique non-negative ints
    """
    if isinstance(spec, int):
        if spec < 0:
            return []
        if spec <= 1:
            return [0]
        return list(range(spec))

    # sequence of lags
    lags = sorted({int(x) for x in spec})
    lags = [x for x in lags if x >= 0]
    return lags


def build_supervised_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    lookback: int,
    horizon: int,
    *,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    lags_by_feature: Optional[Mapping[str, LagSpec]] = None,
    default_lags: Optional[LagSpec] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construye dataset supervisado para regresión:
        y(t) = log(P(t+h)/P(t))

    Lags:
      - Si lags_by_feature is None: comportamiento clásico -> TODOS feature_cols usan 0..lookback-1
      - Si lags_by_feature existe: puedes controlar lags por feature.
          - int 0/1 => solo lag0 (tu "no lag")
          - int n>=2 => lags 0..n-1
          - lista => lags explícitos
          - int <0 => feature excluida

      - default_lags:
          - si None y lags_by_feature existe => features no especificadas usan solo lag0
          - si no-None => fallback para features no especificadas (ej. default_lags=lookback)

    Importante:
      - shifts se calculan PER-SÍMBOLO (groupby shift)
      - meta incluye target_timestamp para split sin leakage por horizon
    """
    if group_col not in df.columns:
        raise ValueError(f"Falta columna {group_col}")
    if timestamp_col not in df.columns:
        raise ValueError(f"Falta columna {timestamp_col}")
    if price_col not in df.columns:
        raise ValueError(f"Falta columna {price_col}")

    if horizon <= 0:
        raise ValueError("horizon debe ser > 0")
    if lookback <= 0:
        raise ValueError("lookback debe ser > 0")

    # Validate feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan {len(missing)} features en df. Ej: {missing[:10]}")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Orden necesario para shift por grupo
    df = df.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    g = df.groupby(group_col, sort=False)

    # Target futuro y timestamp futuro (por símbolo)
    future_price = g[price_col].shift(-horizon)
    target = np.log(future_price / df[price_col])
    target_ts = g[timestamp_col].shift(-horizon)

    # ---------- Build lag plan ----------
    if lags_by_feature is None:
        # Classic behavior: every feature gets full lookback
        feat_to_lags: Dict[str, list[int]] = {c: list(range(lookback)) for c in feature_cols}
    else:
        # Custom behavior
        if default_lags is None:
            default_lags_norm = [0]  # default: "no lag" (only lag0)
        else:
            default_lags_norm = _normalize_lag_spec(default_lags)

        feat_to_lags = {}
        for c in feature_cols:
            spec = lags_by_feature.get(c, default_lags_norm)  # type: ignore[arg-type]
            # spec might already be a list if default_lags_norm used
            lags = _normalize_lag_spec(spec) if not isinstance(spec, list) else spec
            feat_to_lags[c] = lags

    # Build lag -> features map for efficient shifting
    lag_to_features: Dict[int, list[str]] = defaultdict(list)
    for feat, lags in feat_to_lags.items():
        for lag in lags:
            lag_to_features[int(lag)].append(feat)

    # If user excluded everything, error early
    if len(lag_to_features) == 0:
        raise ValueError("No features selected: all features were excluded by lags_by_feature.")

    # ---------- Compute lagged features (per lag, per group) ----------
    X_parts = []
    for lag in sorted(lag_to_features.keys()):
        cols = lag_to_features[lag]
        lag_df = g[cols].shift(lag)
        lag_df.columns = [f"{c}_lag{lag}" for c in cols]
        X_parts.append(lag_df)

    X = pd.concat(X_parts, axis=1)

    out = pd.concat(
        [
            df[[group_col, timestamp_col]],
            target_ts.rename("target_timestamp"),
            X,
            target.rename("target"),
        ],
        axis=1,
    ).dropna().reset_index(drop=True)

    # Orden global por tiempo (útil para CV / lectura)
    out = out.sort_values([timestamp_col, group_col]).reset_index(drop=True)

    meta = out[[group_col, timestamp_col, "target_timestamp"]].copy()
    X_out = out.drop(columns=[group_col, timestamp_col, "target_timestamp", "target"])
    y_out = out["target"].astype(np.float32)

    return X_out, y_out, meta

from collections import defaultdict
from typing import Mapping, Optional, Tuple
import numpy as np
import pandas as pd

def build_inference_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    *,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    lags_by_feature: Optional[Mapping[str, LagSpec]] = None,
    default_lags: Optional[LagSpec] = None,
    select: str = "latest",          # "latest" | "tail" | "all"
    tail_n: int = 1,                 # usado si select="tail"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds (X, meta) for inference without y_true.
    Drops only rows with missing lag features (not dropping last horizon rows).
    """
    if group_col not in df.columns:
        raise ValueError(f"Missing {group_col}")
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing {timestamp_col}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature cols in df. Example: {missing[:10]}")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df[df[timestamp_col].notna()].sort_values([group_col, timestamp_col]).reset_index(drop=True)

    g = df.groupby(group_col, sort=False)

    # --- lag plan ---
    if lags_by_feature is None:
        feat_to_lags = {c: list(range(lookback)) for c in feature_cols}
    else:
        if default_lags is None:
            default_lags_norm = [0]   # por defecto: solo lag0
        else:
            default_lags_norm = _normalize_lag_spec(default_lags)

        feat_to_lags = {}
        for c in feature_cols:
            spec = lags_by_feature.get(c, default_lags_norm)  # type: ignore[arg-type]
            lags = _normalize_lag_spec(spec) if not isinstance(spec, list) else spec
            feat_to_lags[c] = lags

    lag_to_features = defaultdict(list)
    for feat, lags in feat_to_lags.items():
        for lag in lags:
            lag_to_features[int(lag)].append(feat)

    if not lag_to_features:
        raise ValueError("No lag features selected (everything excluded).")

    # --- compute lagged X ---
    X_parts = []
    for lag in sorted(lag_to_features.keys()):
        cols = lag_to_features[lag]
        lag_df = g[cols].shift(lag)
        lag_df.columns = [f"{c}_lag{lag}" for c in cols]
        X_parts.append(lag_df)

    X = pd.concat(X_parts, axis=1)

    out = pd.concat([df[[group_col, timestamp_col]], X], axis=1).dropna().reset_index(drop=True)
    out = out.sort_values([timestamp_col, group_col]).reset_index(drop=True)

    # seleccionar filas para “live”
    if select == "latest":
        out = out.groupby(group_col, sort=False).tail(1).reset_index(drop=True)
    elif select == "tail":
        out = out.groupby(group_col, sort=False).tail(int(tail_n)).reset_index(drop=True)
    elif select == "all":
        pass
    else:
        raise ValueError("select must be 'latest', 'tail', or 'all'")

    meta = out[[group_col, timestamp_col]].copy()
    X_out = out.drop(columns=[group_col, timestamp_col])

    return X_out, meta




def time_split_indices(n: int, train_frac: float = 0.7, val_frac: float = 0.15):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)

def time_split_masks(
    meta: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    timestamp_col: str = "timestamp",
    target_ts_col: str = "target_timestamp",
):
    meta = meta.copy()
    meta[timestamp_col] = pd.to_datetime(meta[timestamp_col])
    meta[target_ts_col] = pd.to_datetime(meta[target_ts_col])

    unique_ts = np.sort(meta[timestamp_col].unique())
    n = len(unique_ts)
    if n < 10:
        raise ValueError("Muy pocos timestamps para split razonable")
    
    train_end = unique_ts[int(n * train_frac) - 1]
    val_end = unique_ts[int(n * (train_frac + val_frac)) - 1]
    # Train: base time <= train_end y el target tambien dentro de train

    train_mask = (meta[timestamp_col] <= train_end) & (meta[target_ts_col] <= train_end)

    val_mask = (
        (meta[timestamp_col] > train_end)
        & (meta[timestamp_col] <= val_end)
        & (meta[target_ts_col] <= val_end)
    )

    # Test
    test_mask = meta[timestamp_col] > val_end

    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy(), train_end, val_end

def time_split_mask_by_time(meta, train_start, train_end, val_end, *, timestamp_col="timestamp", target_col="target_timestamp"):
    ts  = pd.to_datetime(meta[timestamp_col])
    tgt = pd.to_datetime(meta[target_col])

    train_mask = (ts >= train_start) & (tgt < train_end)
    val_mask   = (ts >= train_end) & (tgt < val_end)
    test_mask  = (ts >= val_end)
    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()

def time_split_mask_by_time_purged(
    meta: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,     # embargo cut por timestamp
    val_start: pd.Timestamp,     # purge cut por target_timestamp
    val_end: pd.Timestamp,
    timestamp_col: str = "timestamp",
    target_col: str = "target_timestamp",
):
    """
    Purged walk-forward split:
    - train: timestamp in [train_start, train_end) AND target_timestamp < val_start
    - val:   timestamp >= val_start AND target_timestamp < val_end
    """

    ts = pd.to_datetime(meta[timestamp_col])
    tgt = pd.to_datetime(meta[target_col])

    train_mask = (ts >= train_start) & (ts < train_end) & (tgt < val_start)
    val_mask   = (ts >= val_start) & (tgt < val_end)

    return train_mask.to_numpy(), val_mask.to_numpy()

def make_test_mask(
    meta: pd.DataFrame,
    *,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp | None,
    timestamp_col: str = "timestamp",
    target_col: str = "target_timestamp",
):
    ts = pd.to_datetime(meta[timestamp_col])
    tgt = pd.to_datetime(meta[target_col])

    mask = (ts >= test_start)
    if test_end is not None:
        mask &= (tgt <= test_end)

    # si hay NaT en target_timestamp, esto los elimina también
    mask &= tgt.notna()

    return mask.to_numpy()



def purged_ts_cv_splits(
    meta: pd.DataFrame,
    n_splits: int = 5,
    timestamp_col: str = "timestamp",
    target_ts_col: str = "target_timestamp",
):
    """
    Creates splits of type TimeSeriesSplit but
    - splits by unique timestamps (not by rows)
    - purges train: target_timestamp <= train_end
    - purges test: target_timestamp <= test_end
    Returns list of (train_idx, test_idx) for GridSearchCV(cv=...)

    """

    m = meta.copy()
    m[timestamp_col] = pd.to_datetime(m[timestamp_col])
    m[target_ts_col] = pd.to_datetime(m[target_ts_col])

    unique_ts = np.array(sorted(m[timestamp_col].unique()))
    if len(unique_ts) < (n_splits + 2):
        raise ValueError(
            f"We need more unique timestamps ({len(unique_ts)}) for n_splits = {n_splits}."
        )
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []

    for train_ts_idx, test_ts_idx in tscv.split(unique_ts):
        train_end = unique_ts[train_ts_idx[-1]]
        test_end = unique_ts[test_ts_idx[-1]]

        train_mask = (m[timestamp_col] <= train_end) & (m[target_ts_col] <= train_end)

        test_mask = (
            (m[timestamp_col] > train_end)
            & (m[timestamp_col] <= test_end)
            & (m[target_ts_col] <= test_end)
        )

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        # If purging leads to a empty fold, we skip it
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        splits.append((train_idx, test_idx))

    if len(splits) == 0:
        raise ValueError(
            "Todos los folds quedaron vacios tras purgar. "
            "Reducir n_splits o revisar horizon/lookback."
        )
    
    return splits

# Walk-forward generator (3 folds)
def make_walk_forward_boundaries(
    meta: pd.DataFrame,
    *,
    n_folds: int,
    val_days: int,
    test_days: int,
    step_days: int | None = None,
    train_days: int | None = None,   # None => expanding
    timestamp_col: str = "timestamp",
):
    ts = pd.to_datetime(meta[timestamp_col])
    unique_days = np.array(sorted(ts.unique()))
    n = len(unique_days)

    step_days = val_days if step_days is None else step_days

    test_start_idx = n - test_days
    test_start = pd.Timestamp(unique_days[test_start_idx])
    test_end   = pd.Timestamp(unique_days[-1])

    folds = []
    for k in range(n_folds):
        # fold 0 es el más viejo
        val_start_idx = test_start_idx - (n_folds - k) * step_days
        val_end_idx   = val_start_idx + val_days

        if val_end_idx > test_start_idx:
            raise ValueError("Tus folds se pisan con el test. Baja n_folds/val/step o sube datos.")

        val_start = pd.Timestamp(unique_days[val_start_idx])
        val_end   = pd.Timestamp(unique_days[val_end_idx])

        if train_days is None:
            train_start = pd.Timestamp(unique_days[0])
        else:
            train_start_idx = max(0, val_start_idx - train_days)
            train_start = pd.Timestamp(unique_days[train_start_idx])

        train_end = val_start

        folds.append({
            "fold": k + 1,
            "train_start": train_start,
            "train_end": train_end,
            "val_end": val_end,
        })

    return {"folds": folds, "test_start": test_start, "test_end": test_end, "unique_days_n": n}

# updated
def make_walk_forward_plan(
    meta: pd.DataFrame,
    *,
    val_days: int = 126,          # 6 meses aprox
    step_days: int = 126,         # frecuencia de reentreno (semestral)
    test_days: int = 504,         # 2 años de test holdout
    train_days: int | None = 252*5,  # rolling 5 años (None => expanding)
    embargo_days: int = 0,        # opcional (normalmente 0 si purgas por target_timestamp)
    min_train_days: int = 252*3,  # mínimo 3 años entrenando para no hacer folds ridículos
    timestamp_col: str = "timestamp",
) -> dict:
    """
    Walk-forward purged plan:
    - Entrenamiento: [train_start, train_end) en timestamp, y además target_timestamp < val_start (purge)
    - Validación:    [val_start, val_end) en timestamp, y además target_timestamp < val_end
    - Test holdout:  [test_start, test_end]
    
    folds se construyen hacia atrás desde test_start con paso step_days
    """

    ts = pd.to_datetime(meta[timestamp_col])
    unique_days = np.array(sorted(ts.unique()))
    n = len(unique_days)

    if n <= (test_days + val_days + min_train_days + 10):
        raise ValueError(
            f"No hay suficientes días: n={n}. "
            f"Necesitas al menos ~{test_days + val_days + min_train_days} + margen."
        )

    test_start_idx = n - test_days
    test_start = pd.Timestamp(unique_days[test_start_idx])
    test_end = pd.Timestamp(unique_days[-1])

    # El fold más reciente termina justo en el arranque del test
    last_val_end_idx = test_start_idx
    folds_rev = []  # del más reciente al más viejo

    j = 0
    while True:
        val_end_idx = last_val_end_idx - j * step_days
        val_start_idx = val_end_idx - val_days

        if val_start_idx <= 0:
            break

        # embargo: recortas training por timestamp antes de val_start
        train_end_idx = val_start_idx - embargo_days
        if train_end_idx <= 0:
            break

        if train_days is None:
            train_start_idx = 0
        else:
            train_start_idx = train_end_idx - train_days

        if train_start_idx < 0:
            break

        if (train_end_idx - train_start_idx) < min_train_days:
            break

        fold = {
            "val_start": pd.Timestamp(unique_days[val_start_idx]),
            "val_end": pd.Timestamp(unique_days[val_end_idx]),
            "train_start": pd.Timestamp(unique_days[train_start_idx]),
            "train_end": pd.Timestamp(unique_days[train_end_idx]),  # por embargo (<= val_start)
        }
        folds_rev.append(fold)
        j += 1

    if len(folds_rev) == 0:
        raise ValueError("No pude construir ni un fold con esos parámetros. Baja min_train_days, test_days o train_days.")

    # Ordena de más viejo a más reciente
    folds = list(reversed(folds_rev))
    for k, f in enumerate(folds, start=1):
        f["fold"] = k

    return {
        "folds": folds,
        "test_start": test_start,
        "test_end": test_end,
        "unique_days_n": n,
    }



import re
import pandas as pd

def _parse_timeframe(timeframe: str) -> tuple[int, str]:
    m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", str(timeframe).strip())
    if not m:
        raise ValueError(f"Unrecognized timeframe: {timeframe}")
    return int(m.group(1)), m.group(2).lower()

def estimate_target_timestamp(timestamp: pd.Series, timeframe: str, horizon: int) -> pd.Series:
    n, unit = _parse_timeframe(timeframe)
    steps = int(n) * int(horizon)

    # Daily bars -> business days (aprox)
    if "day" in unit:
        return timestamp + pd.offsets.BDay(steps)
    if "hour" in unit:
        return timestamp + pd.Timedelta(hours=steps)
    if "min" in unit:
        return timestamp + pd.Timedelta(minutes=steps)

    # fallback
    return timestamp + pd.Timedelta(days=steps)

_LAG_PATTERNS = [
    re.compile(r"^(?P<feat>.+)_lag(?P<lag>\d+)$"),
    re.compile(r"^(?P<feat>.+)_lag_(?P<lag>\d+)$"),
    re.compile(r"^(?P<feat>.+)_t-(?P<lag>\d+)$"),
    re.compile(r"^(?P<feat>.+)_shift(?P<lag>\d+)$"),
]

# CNN feature normalization into vectors (C, T)

def parse_feat_lag(col: str) -> Optional[Tuple[str, int]]:
    for pat in _LAG_PATTERNS:
        m = pat.match(col)
        if m:
            return m.group("feat"), int(m.group("lag"))
    return None


