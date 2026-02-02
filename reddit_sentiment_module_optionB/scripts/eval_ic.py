from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class ICSummary:
    horizon: int
    n_obs: int
    spearman_ic: float


def compute_forward_returns(
    prices: pd.DataFrame,
    horizon_days: int,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    prices: columns at least [date,ticker,close]
    returns: forward return over horizon_days: close(t+h)/close(t)-1
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df["fwd_close"] = df.groupby("ticker")[price_col].shift(-horizon_days)
    df[f"fwd_ret_{horizon_days}"] = df["fwd_close"] / df[price_col] - 1.0
    return df[["date", "ticker", f"fwd_ret_{horizon_days}"]]


def spearman_ic_per_day(
    feature_store: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    factor_col: str,
    ret_col: str,
) -> ICSummary:
    """Average daily Spearman correlation between factor and forward returns."""
    df = feature_store.merge(fwd_returns, on=["date", "ticker"], how="inner")
    if df.empty:
        return ICSummary(horizon=0, n_obs=0, spearman_ic=float("nan"))

    df = df.dropna(subset=[factor_col, ret_col])
    if df.empty:
        return ICSummary(horizon=0, n_obs=0, spearman_ic=float("nan"))

    # Compute daily spearman via ranks (corr of ranks)
    def _corr(g: pd.DataFrame) -> float:
        x = g[factor_col].rank(pct=False)
        y = g[ret_col].rank(pct=False)
        if len(g) < 3:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    daily = df.groupby("date", sort=False).apply(_corr)
    ic = float(np.nanmean(daily.to_numpy(dtype=float)))
    return ICSummary(horizon=0, n_obs=int(df.shape[0]), spearman_ic=ic)
