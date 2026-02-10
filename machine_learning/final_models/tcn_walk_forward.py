
"""
tcn_walk_forward.py

Annual walk-forward (retrain once/year) for a TCN time-series model with fixed horizon H=22 trading days.

What it does:
- Builds annual fold schedule (shared logic via walk_forward_common.py)
- For each fold:
  * trains TCN using data available "as-of" train cut-off (anti-leakage)
  * predicts OOS for the whole validation year
  * upserts OOS predictions into SQLite (idempotent)
  * evaluates fold metrics via eval_regression_extended and writes them to JSON (NOT SQLite)
  * saves fold artifacts (model + scaler + metadata)

It also exposes:
- build_final_inference_fold(...): build the "deploy" fold after the last validation year (shared by CNN/XGB too)
  so you can train a final model for inference using the same walk-forward logic.

Assumptions:
- You have project functions:
    - machine_learning.data_collectors.build_ml_dataframe
    - machine_learning.evaluators.eval_regression_extended
    - python_scripts.LLM_analysis.preprocess_store_database.get_connection (for data DB)
- Your features are already leakage-safe "as-of" each timestamp (e.g., econ/fmp merged backward).

Adjust the CONFIG section below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Project imports (adjust if your repo layout differs)
# ---------------------------------------------------------------------
import sys
import os
PROJECT_ROOT = os.path.abspath('../..')
sys.path.append(PROJECT_ROOT)

from machine_learning.data_collectors import build_ml_dataframe
from machine_learning.evaluators import eval_regression_extended
from python_scripts.LLM_analysis.preprocess_store_database import get_connection

from machine_learning.final_models.walk_forward_common import (
    build_annual_fold_schedule,
    build_final_train_fold_from_schedule,
    make_default_preds_schema_adapter,
    run_walk_forward_oos_to_sqlite,
    validate_annual_fold_schedule,
)

try:
    from database_tier1 import TARGET_STOCKS
except Exception:
    TARGET_STOCKS = []


# =============================================================================
# Reproducibility
# =============================================================================

def set_global_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic (can reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# =============================================================================
# TCN model
# =============================================================================

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, channels: List[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x: (B, L, F) -> Conv1d: (B, F, L)
        x = x.transpose(1, 2)
        z = self.tcn(x)        # (B, C, L)
        last = z[:, :, -1]     # causal "as-of" last step
        y = self.head(last)    # (B, 1)
        return y.squeeze(-1)


# =============================================================================
# Data -> TCN samples (anti-leakage)
# =============================================================================

def build_tcn_samples(
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    lookback: int,
    horizon_td: int,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    require_y: bool = True,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
) -> Dict[str, Any]:
    """
    Build TCN samples:
      - X: (N, lookback, n_features) using ONLY history <= t
      - y(t)=log(close[t+H]/close[t]) if available; NaN otherwise
      - meta: columns [symbol, timestamp, target_timestamp]

    Rules:
      - If not enough history for a timestamp -> skip (do NOT pad with zeros)
      - If any NaNs in the lookback window -> skip sample
      - If require_y=True -> keep only samples with y available
    """
    if price_col not in df.columns:
        raise ValueError(f"price_col={price_col!r} not in df columns")

    # ensure no duplicates in columns list
    cols = [group_col, timestamp_col, price_col] + [c for c in feature_cols if c not in {group_col, timestamp_col, price_col}]
    dd = df[cols].copy()
    dd[timestamp_col] = pd.to_datetime(dd[timestamp_col], errors="coerce")
    if getattr(dd[timestamp_col].dt, "tz", None) is not None:
        dd[timestamp_col] = dd[timestamp_col].dt.tz_convert(None)
    dd = dd.dropna(subset=[group_col, timestamp_col]).sort_values([group_col, timestamp_col])

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    X_list: List[np.ndarray] = []
    y_list: List[np.float32] = []
    meta_rows: List[Dict[str, Any]] = []

    for sym, g in dd.groupby(group_col, sort=False):
        g = g.sort_values(timestamp_col).reset_index(drop=True)

        ts = pd.to_datetime(g[timestamp_col], errors="coerce").to_numpy()
        close = g[price_col].to_numpy(dtype=np.float64)
        feats = g[feature_cols].to_numpy(dtype=np.float64)

        # close must be 1D
        if close.ndim == 2 and close.shape[1] == 1:
            close = close[:, 0]
        if close.ndim != 1:
            raise ValueError(f"`close` must be 1D (n,), got shape={close.shape}")

        n = len(g)
        if n < lookback:
            continue

        for i in range(lookback - 1, n):
            t = pd.Timestamp(ts[i])
            if start_date is not None and t < start_date:
                continue
            if end_date is not None and t > end_date:
                break

            seq = feats[i - lookback + 1 : i + 1, :]
            if not np.isfinite(seq).all():
                continue

            tgt_i = i + int(horizon_td)
            if tgt_i < n and np.isfinite(close[tgt_i]) and np.isfinite(close[i]) and close[i] > 0:
                y = float(np.log(close[tgt_i] / close[i]))
                target_ts = pd.Timestamp(ts[tgt_i])
            else:
                y = float("nan")
                target_ts = pd.NaT

            if require_y and not np.isfinite(y):
                continue

            X_list.append(seq.astype(np.float32))
            y_list.append(np.float32(y))
            meta_rows.append({"symbol": sym, "timestamp": t, "target_timestamp": target_ts})

    if not X_list:
        return {
            "X": np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "meta": pd.DataFrame(columns=["symbol", "timestamp", "target_timestamp"]),
        }

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], errors="coerce")
    meta["target_timestamp"] = pd.to_datetime(meta["target_timestamp"], errors="coerce")
    return {"X": X, "y": y, "meta": meta}


def purged_time_split_masks(meta: pd.DataFrame, val_frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal purged split for early stopping:
      - choose val_start by timestamp quantile
      - train: timestamp < val_start AND target_timestamp < val_start
      - val:   timestamp >= val_start AND target_timestamp finite
    """
    ts = pd.to_datetime(meta["timestamp"], errors="coerce")
    tgt = pd.to_datetime(meta["target_timestamp"], errors="coerce")

    uniq = np.sort(ts.dropna().unique())
    if len(uniq) < 20:
        m = np.ones(len(meta), dtype=bool)
        return m, ~m

    cut_idx = int((1.0 - val_frac) * len(uniq))
    cut_idx = max(1, min(cut_idx, len(uniq) - 1))
    val_start = pd.Timestamp(uniq[cut_idx])

    train_mask = (ts < val_start) & (tgt < val_start)
    val_mask = (ts >= val_start) & (tgt.notna())
    return train_mask.to_numpy(), val_mask.to_numpy()


def mean_daily_rank_ic(meta_ts: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean daily Spearman correlation across symbols for each day.
    This is a ranking-oriented metric (higher is better).
    """
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(meta_ts, errors="coerce"), "y_true": y_true, "y_pred": y_pred}
    ).dropna(subset=["timestamp"])

    def _ic(g: pd.DataFrame) -> float:
        if g.shape[0] < 3:
            return np.nan
        # Spearman rank correlation
        return float(g["y_true"].corr(g["y_pred"], method="spearman"))

    daily = df.groupby("timestamp", sort=True).apply(_ic)
    if daily.empty:
        return float("nan")
    return float(np.nanmean(daily.to_numpy(dtype=float)))


def train_tcn_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    cfg: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """
    Train TCN with scaler fit ONLY on train split (purged).
    Early stopping is based on ranking-oriented metric: val_mean_dailyIC (mean daily rank-IC).
    """
    train_mask, val_mask = purged_time_split_masks(meta, val_frac=float(cfg.get("val_frac", 0.1)))

    X_train = X[train_mask]
    y_train = y[train_mask]
    meta_train = meta.loc[train_mask].reset_index(drop=True)

    X_val = X[val_mask]
    y_val = y[val_mask]
    meta_val = meta.loc[val_mask].reset_index(drop=True)

    if X_train.shape[0] < int(cfg.get("min_train_samples", 200)):
        raise ValueError(f"Too few training samples: {X_train.shape[0]}")

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    def transform(arr: np.ndarray) -> np.ndarray:
        flat = scaler.transform(arr.reshape(-1, arr.shape[-1]))
        return flat.reshape(arr.shape).astype(np.float32)

    X_train_s = transform(X_train)
    X_val_s = transform(X_val) if X_val.shape[0] > 0 else X_val

    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    n_features = X.shape[-1]
    model = TCNRegressor(
        n_features=n_features,
        channels=list(cfg["channels"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss()

    # Loaders
    g = torch.Generator()
    g.manual_seed(int(cfg.get("loader_seed", 0)))

    train_loader = DataLoader(
        SeqDataset(X_train_s, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
        num_workers=0,
    )

    # Early stopping by ranking metric
    patience = int(cfg.get("patience", 8))
    min_delta = float(cfg.get("min_delta", 1e-4))
    best_metric = -np.inf
    best_state = None
    bad = 0

    def _predict_array(Xs: np.ndarray, bs: int = 4096) -> np.ndarray:
        if Xs.shape[0] == 0:
            return np.asarray([], dtype=np.float32)
        model.eval()
        loader = DataLoader(torch.from_numpy(Xs).float(), batch_size=int(bs), shuffle=False, drop_last=False)
        preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                yhat = model(xb).detach().cpu().numpy().astype(np.float32)
                preds.append(yhat)
        return np.concatenate(preds, axis=0)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")

        # Ranking-oriented validation metric
        val_metric = float("nan")
        if X_val_s.shape[0] > 0:
            yhat_val = _predict_array(X_val_s, bs=int(cfg.get("eval_batch_size", 8192)))
            val_metric = mean_daily_rank_ic(meta_val["timestamp"], y_val.astype(float), yhat_val.astype(float))

        if np.isfinite(val_metric):
            if ep % int(cfg.get("log_every", 1)) == 0:
                print(f"epoch={ep:03d} train_loss={train_loss:.6f} val_mean_dailyIC={val_metric:.6f}")

            if val_metric > (best_metric + min_delta):
                best_metric = val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print("Early stop (val_mean_dailyIC).")
                    break
        else:
            # If no valid val metric, keep training but log occasionally
            if ep % 5 == 0:
                print(f"epoch={ep:03d} train_loss={train_loss:.6f} val_mean_dailyIC=nan")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "scaler": scaler,
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "best_val_mean_dailyIC": float(best_metric) if np.isfinite(best_metric) else None,
        "cfg": cfg,
    }


@torch.no_grad()
def predict_tcn_bundle(bundle: Dict[str, Any], X: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    model = bundle["model"]
    scaler = bundle["scaler"]

    model.eval()
    flat = scaler.transform(X.reshape(-1, X.shape[-1]))
    Xs = flat.reshape(X.shape).astype(np.float32)

    loader = DataLoader(torch.from_numpy(Xs).float(), batch_size=int(batch_size), shuffle=False, drop_last=False)
    preds = []
    for xb in loader:
        xb = xb.to(device)
        yhat = model(xb).detach().cpu().numpy().astype(np.float32)
        preds.append(yhat)
    return np.concatenate(preds, axis=0) if preds else np.asarray([], dtype=np.float32)


def save_tcn_artifact(bundle: Dict[str, Any], out_dir: str, meta: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.pt")
    torch.save(bundle["model"].state_dict(), model_path)

    scaler_path = os.path.join(out_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(bundle["scaler"], f)

    info = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "bundle_info": {k: v for k, v in bundle.items() if k not in ["model", "scaler"]},
    }
    info_path = os.path.join(out_dir, "artifact.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    return out_dir


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # Repro
    seed: int = 0

    # Model
    model_name: str = "TCN"
    horizon_td: int = 22
    lookback: int = 60

    # Data
    timeframe: str = "1Day"
    symbols: Optional[List[str]] = None
    data_start: str = "2015-01-01"
    data_end: str = "2025-12-31"

    include_indicators: bool = False
    indicator_names: Optional[List[str]] = None

    include_econ: bool = False
    econ_names: Optional[List[str]] = None

    include_fmp: bool = False
    fmp_feature_names: Optional[List[str]] = None
    fmp_prefix: str = "fmp_"
    keep_fmp_asof_date: bool = False

    # Fold schedule
    start_year: int = 2016
    fold_schedule_path: str = "annual_fold_schedule_h22.csv"

    # Results
    results_sqlite_path: str = "results.sqlite"
    preds_table: str = "PREDS_TABLE"
    metrics_output_dir: str = "metrics_json"

    # Artifacts
    artifacts_root: str = "runs"

    # TCN hyperparams
    tcn_cfg: Optional[Dict[str, Any]] = None


def default_tcn_cfg() -> Dict[str, Any]:
    return {
        "kernel_size": 3,
        "channels": [32, 32, 32],
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 512,
        "epochs": 40,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        # early stopping
        "patience": 8,
        "min_delta": 1e-4,
        "val_frac": 0.1,
        "eval_batch_size": 8192,
        "log_every": 1,
    }


# =============================================================================
# Shared fold builder (for final inference model)
# =============================================================================

def build_final_inference_fold(fold_schedule: pd.DataFrame, trading_dates: pd.DatetimeIndex, horizon_td: int) -> pd.Series:
    """
    Returns a fold row (Series) representing the deploy training cut-off AFTER the last validated year.
    This is intended to be shared by CNN/XGB too (same logic).
    """
    return build_final_train_fold_from_schedule(fold_schedule, trading_dates, horizon_td=horizon_td)


# =============================================================================
# Walk-forward run
# =============================================================================

def load_raw_dataframe(cfg: Config) -> pd.DataFrame:
    symbols = cfg.symbols or (list(TARGET_STOCKS) if TARGET_STOCKS else [])
    if not symbols:
        raise ValueError("Config.symbols is empty and TARGET_STOCKS not available.")

    conn = get_connection()
    df_raw = build_ml_dataframe(
        conn,
        symbols=symbols,
        timeframe=cfg.timeframe,
        start=cfg.data_start,
        end=cfg.data_end,
        include_indicators=cfg.include_indicators,
        indicator_names=cfg.indicator_names or [],
        include_econ=cfg.include_econ,
        econ_indicator_names=cfg.econ_names or [],
        include_fmp=cfg.include_fmp,
        fmp_feature_names=cfg.fmp_feature_names or [],
        fmp_prefix=cfg.fmp_prefix,
        keep_fmp_asof_date=cfg.keep_fmp_asof_date,
    )
    if df_raw.empty:
        raise ValueError("df_raw is empty (check DB connection/symbols/date range).")

    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
    if getattr(df_raw["timestamp"].dt, "tz", None) is not None:
        df_raw["timestamp"] = df_raw["timestamp"].dt.tz_convert(None)

    df_raw = df_raw.dropna(subset=["symbol", "timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df_raw


def build_feature_cols(df_raw: pd.DataFrame) -> List[str]:
    non_feature_cols = {"symbol", "timestamp", "timeframe"}
    # Keep close as feature too (allowed as-of t)
    feature_cols = [c for c in df_raw.columns if c not in non_feature_cols]
    if "close" not in df_raw.columns:
        raise ValueError("Required column 'close' not found in df_raw.")
    # feature_cols can include 'close'; that's ok
    return feature_cols


def build_schedule(cfg: Config, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build OR load the annual fold schedule.

    IMPORTANT for multi-model consistency (TCN/CNN/XGB):
    - If cfg.fold_schedule_path exists, we LOAD it (do not rebuild).
    - Otherwise we build from trading_dates and save it.

    This ensures all models read an identical schedule for the same horizon.
    """
    if os.path.exists(cfg.fold_schedule_path):
        schedule = pd.read_csv(cfg.fold_schedule_path)
        for c in [
            "train_start_date",
            "train_fit_end_date",
            "train_last_labeled_date",
            "val_start_date",
            "val_end_date",
            "val_last_labeled_date",
            "global_last_labeled_date",
        ]:
            if c in schedule.columns:
                schedule[c] = pd.to_datetime(schedule[c], errors="coerce")
        return schedule

    min_year = int(trading_dates.min().year)
    max_year = int(trading_dates.max().year)

    start_year = max(cfg.start_year, min_year)
    first_val_year = max(start_year + 1, min_year + 1)
    last_val_year = max_year

    schedule = build_annual_fold_schedule(
        trading_dates=trading_dates,
        start_year=start_year,
        first_val_year=first_val_year,
        last_val_year=last_val_year,
        horizon_td=cfg.horizon_td,
    )
    schedule.to_csv(cfg.fold_schedule_path, index=False)
    return schedule

def make_run_meta(cfg: Config) -> Dict[str, Any]:
    tcn_cfg = cfg.tcn_cfg or default_tcn_cfg()
    run_cfg = {
        "model_name": cfg.model_name,
        "horizon_td": cfg.horizon_td,
        "lookback": cfg.lookback,
        "timeframe": cfg.timeframe,
        "symbols": cfg.symbols,
        "seed": cfg.seed,
        "tcn": tcn_cfg,
    }
    config_hash = stable_hash(run_cfg)[:16]
    model_version = f"{cfg.model_name}_h{cfg.horizon_td}_lb{cfg.lookback}_{config_hash}"
    run_id = f"{cfg.model_name}_annual_{model_version}"
    return {
        "run_id": run_id,
        "model_version": model_version,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run(cfg: Config) -> pd.DataFrame:
    set_global_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcn_cfg = cfg.tcn_cfg or default_tcn_cfg()
    run_meta = make_run_meta(cfg)

    artifacts_dir = os.path.join(cfg.artifacts_root, run_meta["run_id"])
    os.makedirs(artifacts_dir, exist_ok=True)

    print("DEVICE:", device)
    print("RUN_ID:", run_meta["run_id"])
    print("MODEL_VERSION:", run_meta["model_version"])
    print("ARTIFACTS_DIR:", artifacts_dir)

    df_raw = load_raw_dataframe(cfg)
    feature_cols = build_feature_cols(df_raw)

    trading_dates = pd.DatetimeIndex(sorted(df_raw["timestamp"].unique()))
    schedule = build_schedule(cfg, trading_dates)
    schedule = validate_annual_fold_schedule(schedule)
    print("Fold schedule (head):")
    print(schedule.head(3))
    print("Fold schedule (tail):")
    print(schedule.tail(3))
    if schedule.empty:
        raise ValueError("Fold schedule is empty.")

    # Default adapter for the given predictions table schema
    schema_adapter = make_default_preds_schema_adapter(timestamp_format="%Y-%m-%d", include_metadata=True)

    # Builders for common runner
    def build_train_dataset_fn(fold_row: pd.Series) -> Dict[str, Any]:
        train_last = pd.to_datetime(fold_row["train_last_labeled_date"])
        train_fit_end = pd.to_datetime(fold_row["train_fit_end_date"])

        data = build_tcn_samples(
            df_raw,
            feature_cols=feature_cols,
            lookback=cfg.lookback,
            horizon_td=cfg.horizon_td,
            start_date=None,
            end_date=train_last,
            require_y=True,
        )

        # Extra anti-leakage: target_timestamp <= train_fit_end_date
        meta = data["meta"].copy()
        ok = pd.to_datetime(meta["target_timestamp"], errors="coerce") <= train_fit_end
        ok &= pd.to_datetime(meta["timestamp"], errors="coerce") <= train_last

        X = data["X"][ok.to_numpy()]
        y = data["y"][ok.to_numpy()]
        meta = meta.loc[ok].reset_index(drop=True)

        if X.shape[0] == 0:
            raise ValueError(f"Fold {fold_row.get('fold_id')} has no train samples after leakage filters.")

        return {"X": X, "y": y, "meta": meta}

    def build_infer_dataset_fn(fold_row: pd.Series) -> Dict[str, Any]:
        val_start = pd.to_datetime(fold_row["val_start_date"])
        val_end = pd.to_datetime(fold_row["val_end_date"])
        data = build_tcn_samples(
            df_raw,
            feature_cols=feature_cols,
            lookback=cfg.lookback,
            horizon_td=cfg.horizon_td,
            start_date=val_start,
            end_date=val_end,
            require_y=False,  # keep last H days even if y is NaN
        )
        return data

    def fit_model_fn(train_data: Dict[str, Any], fold_row: pd.Series) -> Dict[str, Any]:
        bundle = train_tcn_one_fold(
            train_data["X"],
            train_data["y"],
            train_data["meta"],
            cfg=tcn_cfg,
            device=device,
        )

        fold_id = int(fold_row.get("fold_id", fold_row.get("val_year")))
        fold_dir = os.path.join(artifacts_dir, f"fold_{fold_id}")
        artifact_meta = {
            "run_meta": run_meta,
            "fold": {k: (str(fold_row.get(k)) if pd.notna(fold_row.get(k)) else None) for k in fold_row.index},
            "train_info": {
                "train_size": bundle["train_size"],
                "val_size": bundle["val_size"],
                "best_val_mean_dailyIC": bundle.get("best_val_mean_dailyIC"),
            },
            "model_cfg": tcn_cfg,
            "lookback": cfg.lookback,
            "horizon_td": cfg.horizon_td,
            "feature_cols": feature_cols,
        }
        save_tcn_artifact(bundle, fold_dir, artifact_meta)
        bundle["artifact_dir"] = fold_dir
        return bundle

    def predict_fn(model_obj: Dict[str, Any], infer_data: Dict[str, Any], fold_row: pd.Series) -> np.ndarray:
        X = infer_data["X"]
        return predict_tcn_bundle(model_obj, X, device=device, batch_size=int(tcn_cfg.get("infer_batch_size", 8192)))

    metrics_df = run_walk_forward_oos_to_sqlite(
        model_name=cfg.model_name,
        horizon_td=cfg.horizon_td,
        fold_schedule=schedule,
        build_train_dataset_fn=build_train_dataset_fn,
        build_infer_dataset_fn=build_infer_dataset_fn,
        fit_model_fn=fit_model_fn,
        predict_fn=predict_fn,
        sqlite_path=cfg.results_sqlite_path,
        preds_table=cfg.preds_table,
        schema_adapter=schema_adapter,
        run_meta=run_meta,
        eval_fn=eval_regression_extended,
        metrics_output_dir=cfg.metrics_output_dir,
        # IMPORTANT: use the UNIQUE constraint (not pred_id PK) for idempotent upsert
        preds_conflict_cols=["model_name", "model_version", "run_id", "input_id"],
        verbose=True,
    )

    # Save a copy of fold metrics index
    metrics_df_path = os.path.join(artifacts_dir, "walk_forward_metrics_index.csv")
    metrics_df.to_csv(metrics_df_path, index=False)
    print("Saved metrics index CSV:", metrics_df_path)

    return metrics_df


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-sqlite-path", type=str, default="results.sqlite")
    p.add_argument("--preds-table", type=str, default="PREDS_TABLE")
    p.add_argument("--metrics-output-dir", type=str, default="metrics_json")
    p.add_argument("--data-start", type=str, default="2015-01-01")
    p.add_argument("--data-end", type=str, default="2025-12-31")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols. If empty, uses TARGET_STOCKS.")
    return p.parse_args()


def main():
    args = parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    cfg = Config(
        seed=int(args.seed),
        lookback=int(args.lookback),
        data_start=args.data_start,
        data_end=args.data_end,
        results_sqlite_path=args.results_sqlite_path,
        preds_table=args.preds_table,
        metrics_output_dir=args.metrics_output_dir,
        symbols=symbols,
        tcn_cfg=default_tcn_cfg(),
        fold_schedule_path=f"annual_fold_schedule_h22.csv",
    )

    run(cfg)


if __name__ == "__main__":
    main()
