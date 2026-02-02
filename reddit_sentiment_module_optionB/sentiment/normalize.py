from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def percentile_rank_cs(values: np.ndarray) -> np.ndarray:
    """
    Cross-sectional percentile rank per date.
    values shape: (dates, tickers)
    """
    df = pd.DataFrame(values)
    return df.rank(axis=1, pct=True, method="average").to_numpy(dtype=np.float64)


def zscore_cs(values, axis=1, eps=0.0):
    values = np.asarray(values, dtype=float)

    mean = np.mean(values, axis=axis, keepdims=True)
    std  = np.std(values, axis=axis, keepdims=True)

    # salida inicializada en 0
    result = np.zeros_like(values, dtype=float)

    # divide solo donde std > 0 (o std > eps si quieres tolerancia)
    np.divide(values - mean, std, out=result, where=(std > eps))

    return result


def residualize_by_volume(
    sent_net: np.ndarray,
    sent_volume: np.ndarray,
    min_volume: float = 1e-6,
    min_obs: int = 20,
) -> np.ndarray:
    """
    Per-date cross-sectional OLS residuals:
      sent_net ~ a + b * log1p(sent_volume)

    sent_net, sent_volume: arrays shape (dates, tickers)
    """
    n_dates, _ = sent_net.shape
    resid = np.zeros_like(sent_net, dtype=np.float64)
    x_all = np.log1p(np.maximum(sent_volume, 0.0))

    for i in range(n_dates):
        y = sent_net[i].astype(np.float64, copy=False)
        x = x_all[i].astype(np.float64, copy=False)
        mask = sent_volume[i] > float(min_volume)
        n = int(mask.sum())
        if n < 2 or n < int(min_obs):
            resid[i] = y - y.mean()
            continue

        xm = x[mask].mean()
        ym = y[mask].mean()
        dx = x[mask] - xm
        dy = y[mask] - ym
        var = float((dx * dx).mean())
        if var <= 0.0:
            b = 0.0
        else:
            b = float((dx * dy).mean() / var)
        a = float(ym - b * xm)
        resid[i] = y - (a + b * x)

    return resid
