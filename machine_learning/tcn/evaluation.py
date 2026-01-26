from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return np.nan
    sa = np.std(a)
    sb = np.std(b)
    if sa == 0.0 or sb == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    b = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    return _safe_pearson(a, b)


@dataclass(frozen=True)
class CrossSectionMetrics:
    daily_ic_mean: float
    daily_rankic_mean: float
    quantile_spread_mean: float
    n_days: int


def daily_cross_sectional_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    min_cs_size: int = 20,
    quantile: float = 0.1,
) -> CrossSectionMetrics:
    """
    Calcula:
      - DailyIC mean (Pearson por timestamp)
      - DailyRankIC mean (Spearman por timestamp)
      - QuantileSpread mean: mean(y_true[top-q]) - mean(y_true[bottom-q]) usando score=y_pred
    Agrupa por meta[time_col] (decision time).
    """
    df = pd.DataFrame({
        "t": pd.to_datetime(meta[time_col]).values,
        "y": np.asarray(y_true, dtype=np.float64),
        "p": np.asarray(y_pred, dtype=np.float64),
    })
    df = df.dropna()
    if len(df) == 0:
        return CrossSectionMetrics(np.nan, np.nan, np.nan, 0)

    daily_ic = []
    daily_rankic = []
    qspread = []

    for t, g in df.groupby("t", sort=True):
        if len(g) < min_cs_size:
            continue
        y = g["y"].to_numpy()
        p = g["p"].to_numpy()

        ic = _safe_pearson(y, p)
        ric = _safe_spearman(y, p)

        n = len(g)
        k = max(1, int(np.floor(quantile * n)))
        # top/bottom por pred
        order = np.argsort(p)
        bot = order[:k]
        top = order[-k:]
        spread = float(np.mean(y[top]) - np.mean(y[bot]))

        if np.isfinite(ic):
            daily_ic.append(ic)
        if np.isfinite(ric):
            daily_rankic.append(ric)
        if np.isfinite(spread):
            qspread.append(spread)

    n_days = int(max(len(daily_ic), len(daily_rankic), len(qspread)))
    return CrossSectionMetrics(
        daily_ic_mean=float(np.mean(daily_ic)) if daily_ic else np.nan,
        daily_rankic_mean=float(np.mean(daily_rankic)) if daily_rankic else np.nan,
        quantile_spread_mean=float(np.mean(qspread)) if qspread else np.nan,
        n_days=n_days,
    )


@dataclass(frozen=True)
class BacktestConfig:
    time_col: str = "timestamp"
    group_col: str = "symbol"
    quantile: float = 0.1
    min_cs_size: int = 20
    cost_bps: float = 0.0
    periods_per_year: int = 252


def toy_long_short_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    cfg: BacktestConfig = BacktestConfig(),
) -> Dict[str, object]:
    """
    Toy backtest LS cross-sectional por timestamp (decision time):
      - long top-q, short bottom-q por score=y_pred
      - retorno LS = mean(y_true_long) - mean(y_true_short)
      - turnover aproximado por cambios en weights (equal-weight)
      - coste = cost_bps/10000 * turnover
    y_true debe ser el retorno futuro (log return) para el horizonte tradeado.

    Retorna dict con métricas + series (DataFrame) para inspección.
    """
    df = pd.DataFrame({
        "t": pd.to_datetime(meta[cfg.time_col]).values,
        "sym": meta[cfg.group_col].astype(str).values,
        "y": np.asarray(y_true, dtype=np.float64),
        "p": np.asarray(y_pred, dtype=np.float64),
    }).dropna()

    if len(df) == 0:
        return {"error": "empty_df"}

    df = df.sort_values("t", kind="mergesort")

    prev_w: Dict[str, float] = {}
    rows = []

    for t, g in df.groupby("t", sort=True):
        if len(g) < cfg.min_cs_size:
            continue

        y = g["y"].to_numpy()
        p = g["p"].to_numpy()
        syms = g["sym"].to_numpy()

        n = len(g)
        k = max(1, int(np.floor(cfg.quantile * n)))
        order = np.argsort(p)
        bot_idx = order[:k]
        top_idx = order[-k:]

        long_syms = syms[top_idx]
        short_syms = syms[bot_idx]

        # Equal weights, net 0
        w: Dict[str, float] = {}
        wl = 1.0 / len(long_syms)
        ws = -1.0 / len(short_syms)
        for s in long_syms:
            w[str(s)] = w.get(str(s), 0.0) + wl
        for s in short_syms:
            w[str(s)] = w.get(str(s), 0.0) + ws

        # LS return (log) = mean(long) - mean(short)
        ls_ret = float(np.mean(y[top_idx]) - np.mean(y[bot_idx]))

        # turnover aprox: 0.5*sum |w_t - w_{t-1}|
        keys = set(prev_w.keys()) | set(w.keys())
        turnover = 0.5 * float(sum(abs(w.get(k2, 0.0) - prev_w.get(k2, 0.0)) for k2 in keys))

        cost = float(cfg.cost_bps) / 10000.0 * turnover
        net_ret = ls_ret - cost

        rows.append({
            "timestamp": t,
            "ls_ret": ls_ret,
            "turnover": turnover,
            "cost": cost,
            "net_ret": net_ret,
            "n": n,
            "k": k,
        })
        prev_w = w

    if len(rows) == 0:
        return {"error": "no_days_after_filters"}

    ts = pd.DataFrame(rows).sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    rets = ts["net_ret"].to_numpy(dtype=np.float64)
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else float("nan")
    sharpe = float(mu / sd * np.sqrt(cfg.periods_per_year)) if sd and np.isfinite(sd) and sd > 0 else float("nan")

    equity = np.exp(np.cumsum(rets))
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(np.min(dd))

    out = {
        "n_days": int(len(ts)),
        "sharpe_ann": sharpe,
        "max_drawdown": max_dd,
        "turnover_avg": float(np.mean(ts["turnover"].to_numpy(dtype=np.float64))),
        "hit_rate": float(np.mean((ts["net_ret"].to_numpy(dtype=np.float64) > 0).astype(np.float64))),
        "cum_return": float(equity[-1] - 1.0),
        "series": ts,
    }
    return out
