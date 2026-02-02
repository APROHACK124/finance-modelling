from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .aggregate import _date_range
from .config import SentimentConfig
from .features import AsOfCalendar, exp_time_decay


def compute_top_threads(
    events: pd.DataFrame,
    config: SentimentConfig,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    For each (date,ticker), compute:
      - top_positive_threads: list of up to K thread_id with largest positive contribution
      - top_negative_threads: list of up to K thread_id with most negative contribution

    Uses contributions computed with horizon = config.explain_horizon and half-life = half_life_days[h].
    Only considers events within the last config.explain_lookback_days.
    """
    k = int(config.top_k_threads)
    if k <= 0:
        return pd.DataFrame(columns=["date", "ticker", "top_positive_threads", "top_negative_threads"])

    if events.empty:
        return pd.DataFrame(columns=["date", "ticker", "top_positive_threads", "top_negative_threads"])

    explain_h = int(config.explain_horizon)
    if explain_h not in config.half_life_days:
        raise ValueError(f"half_life_days missing for explain_horizon={explain_h}")
    half_life = float(config.half_life_days[explain_h])
    lookback_days = int(config.explain_lookback_days)

    min_d = min(events["feature_date"])
    max_d = max(events["feature_date"])
    start_date = min_d if start_date is None else start_date
    end_date = max_d if end_date is None else end_date

    cal = AsOfCalendar.from_config(config)
    date_index = _date_range(start_date, end_date)

    # Sort events by created_dt_utc for fast slicing
    ev = events[["ticker", "thread_id", "created_dt_utc", "base_weight", "s"]].dropna().drop_duplicates(subset=["ticker", "thread_id", "created_dt_utc"])
    ev = ev.sort_values("created_dt_utc").reset_index(drop=True)

    # Convert times to numpy datetime64[ns] (UTC naive)
    times = ev["created_dt_utc"].apply(lambda x: x.replace(tzinfo=None)).to_numpy(dtype="datetime64[ns]")

    results = []
    for d in date_index:
        cutoff_dt = cal.cutoff_datetime_utc(d)
        cutoff_ns = np.datetime64(cutoff_dt.replace(tzinfo=None))
        start_dt = cutoff_dt - timedelta(days=lookback_days)
        start_ns = np.datetime64(start_dt.replace(tzinfo=None))

        left = int(np.searchsorted(times, start_ns, side="left"))
        right = int(np.searchsorted(times, cutoff_ns, side="right"))
        if right <= left:
            continue

        window = ev.iloc[left:right].copy()
        t_ns = times[left:right]
        age_days = (cutoff_ns - t_ns) / np.timedelta64(1, "D")
        decay = exp_time_decay(age_days.astype(float), half_life_days=half_life)
        w = window["base_weight"].to_numpy(dtype=float) * decay
        s = window["s"].to_numpy(dtype=int)
        contrib = w * s.astype(float)
        window["contrib"] = contrib

        # Positive
        pos = window[window["contrib"] > 0].copy()
        if not pos.empty:
            pos = pos.sort_values(["ticker", "contrib", "thread_id"], ascending=[True, False, True])
            pos = pos.groupby("ticker", sort=False).head(k)
            pos_lists = pos.groupby("ticker", sort=False)["thread_id"].apply(list).to_dict()
        else:
            pos_lists = {}

        # Negative
        neg = window[window["contrib"] < 0].copy()
        if not neg.empty:
            neg = neg.sort_values(["ticker", "contrib", "thread_id"], ascending=[True, True, True])
            neg = neg.groupby("ticker", sort=False).head(k)
            neg_lists = neg.groupby("ticker", sort=False)["thread_id"].apply(list).to_dict()
        else:
            neg_lists = {}

        tickers = set(pos_lists.keys()) | set(neg_lists.keys())
        for t in tickers:
            results.append(
                {
                    "date": d.isoformat(),
                    "ticker": t,
                    "top_positive_threads": json.dumps(pos_lists.get(t, [])),
                    "top_negative_threads": json.dumps(neg_lists.get(t, [])),
                }
            )

    if not results:
        return pd.DataFrame(columns=["date", "ticker", "top_positive_threads", "top_negative_threads"])

    return pd.DataFrame(results)
