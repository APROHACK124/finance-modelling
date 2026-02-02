from __future__ import annotations

from datetime import date
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .aggregate import _date_range, compute_daily_adds, compute_decayed_running_sums, compute_step_decays
from .config import SentimentConfig


def assemble_market_index(
    global_events: pd.DataFrame,
    config: SentimentConfig,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Build daily market/macro indices for each horizon h.
    Output columns:
      date, market_sent_{h}, market_volume_{h}, market_disagree_{h}
    """
    config.validate()

    if global_events.empty:
        return pd.DataFrame(columns=["date"])

    min_d = min(global_events["feature_date"])
    max_d = max(global_events["feature_date"])
    start_date = min_d if start_date is None else start_date
    end_date = max_d if end_date is None else end_date

    date_index = _date_range(start_date, end_date)
    out = pd.DataFrame({"date": [d.isoformat() for d in date_index]})

    keys = ["market"]
    global_events = global_events.copy()
    global_events["key"] = "market"

    for h in config.horizons:
        half_life = float(config.half_life_days[h])
        step_decays = compute_step_decays(date_index=date_index, config=config, half_life_days=half_life)

        daily_adds = compute_daily_adds(
            global_events,
            horizon=h,
            config=config,
            date_index=date_index,
            keys=keys,
            key_col="key",
        )
        running = compute_decayed_running_sums(daily_adds, step_decays=step_decays)

        numer = running["numer_add"][:, 0]
        denom = running["denom_add"][:, 0]

        market_sent = numer / (denom + float(config.eps))
        market_volume = denom
        market_disagree = 1.0 - np.abs(market_sent)

        out[f"market_sent_{h}"] = market_sent
        out[f"market_volume_{h}"] = market_volume
        out[f"market_disagree_{h}"] = market_disagree

    return out


def compute_sector_indices(
    feature_store: pd.DataFrame,
    ticker_to_sector: pd.DataFrame,
    config: SentimentConfig,
    ticker_col: str = "ticker",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """
    Build sector sentiment indices from per-ticker feature store using:
      sector_numer = sum(sent_net_h * sent_volume_h)
      sector_denom = sum(sent_volume_h)
      sector_sent_h = numer / (denom + eps)

    Output: date, sector, sector_sent_{h}, sector_volume_{h}
    """
    if ticker_to_sector.empty:
        return pd.DataFrame(columns=["date", "sector"])

    df = feature_store.merge(
        ticker_to_sector[[ticker_col, sector_col]].drop_duplicates(),
        left_on="ticker",
        right_on=ticker_col,
        how="left",
    )
    df["sector"] = df[sector_col]
    df = df.dropna(subset=["sector"])
    if df.empty:
        return pd.DataFrame(columns=["date", "sector"])

    out_cols = {"date": df["date"], "sector": df["sector"].astype(str)}
    out = df[["date", "sector"]].drop_duplicates().sort_values(["date", "sector"]).reset_index(drop=True)

    for h in config.horizons:
        sn = df[f"sent_net_{h}"].astype(float)
        sv = df[f"sent_volume_{h}"].astype(float)
        numer = sn * sv
        tmp = df[["date", "sector"]].copy()
        tmp["numer"] = numer
        tmp["denom"] = sv
        g = tmp.groupby(["date", "sector"], sort=False).sum(numeric_only=True).reset_index()
        g[f"sector_sent_{h}"] = g["numer"] / (g["denom"] + float(config.eps))
        g = g.rename(columns={"denom": f"sector_volume_{h}"})
        keep = ["date", "sector", f"sector_sent_{h}", f"sector_volume_{h}"]
        if out.empty:
            out = g[keep]
        else:
            out = out.merge(g[keep], on=["date", "sector"], how="left")

    return out


def attach_sector_sent_to_tickers(
    feature_store: pd.DataFrame,
    ticker_to_sector: pd.DataFrame,
    sector_index: pd.DataFrame,
    config: SentimentConfig,
    ticker_col: str = "ticker",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Attach sector_sent_{h} to each ticker-date as sector_sent_{h}_of_ticker."""
    if ticker_to_sector.empty or sector_index.empty:
        return feature_store

    df = feature_store.merge(
        ticker_to_sector[[ticker_col, sector_col]].drop_duplicates(),
        left_on="ticker",
        right_on=ticker_col,
        how="left",
    )
    df["sector"] = df[sector_col]
    df = df.drop(columns=[ticker_col, sector_col], errors="ignore")

    merged = df.merge(sector_index, on=["date", "sector"], how="left", suffixes=("", "_sector"))

    for h in config.horizons:
        if f"sector_sent_{h}" in merged.columns:
            merged[f"sector_sent_{h}_of_ticker"] = merged[f"sector_sent_{h}"]
    return merged
