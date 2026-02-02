from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from .config import SentimentConfig


def winsorize_series_by_group(
    values: pd.Series,
    groups: pd.Series,
    lower_q: float,
    upper_q: float,
) -> pd.Series:
    """Winsorize values within each group by quantiles."""
    df = pd.DataFrame({"v": values.astype(float), "g": groups.astype(str)})
    qs = df.groupby("g")["v"].quantile([lower_q, upper_q]).unstack()
    qs.columns = ["q_low", "q_high"]
    df = df.join(qs, on="g")
    return df["v"].clip(lower=df["q_low"], upper=df["q_high"])


def compute_engagement(
    post_score: pd.Series,
    top5_comment_score_sum_log1p: Optional[pd.Series],
    config: SentimentConfig,
) -> pd.Series:
    # Clip negatives to 0 before log1p
    post_score_pos = pd.to_numeric(post_score, errors="coerce").fillna(0.0).clip(lower=0.0)
    e_raw = np.log1p(post_score_pos.astype(float))
    if top5_comment_score_sum_log1p is not None:
        e_raw = e_raw + float(config.engagement_alpha) * pd.to_numeric(
            top5_comment_score_sum_log1p, errors="coerce"
        ).fillna(0.0).astype(float)
    return pd.Series(e_raw, index=post_score.index, dtype="float64")


@dataclass(frozen=True)
class AsOfCalendar:
    tz: ZoneInfo
    cutoff_t: time

    @staticmethod
    def from_config(config: SentimentConfig) -> "AsOfCalendar":
        tz = ZoneInfo(config.asof_timezone)
        hh, mm, ss = [int(x) for x in config.asof_cutoff_time.split(":")]
        return AsOfCalendar(tz=tz, cutoff_t=time(hour=hh, minute=mm, second=ss))

    def assign_feature_date(self, created_utc: float) -> date:
        dt_utc = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
        dt_local = dt_utc.astimezone(self.tz)
        d = dt_local.date()
        if dt_local.timetz().replace(tzinfo=None) <= self.cutoff_t:
            return d
        return d + timedelta(days=1)

    def cutoff_datetime_utc(self, d: date) -> datetime:
        dt_local = datetime.combine(d, self.cutoff_t, tzinfo=self.tz)
        return dt_local.astimezone(timezone.utc)


def exp_time_decay(age_days: np.ndarray, half_life_days: float) -> np.ndarray:
    # exp(-ln2 * age / half_life)
    return np.exp(-np.log(2.0) * (age_days.astype(float) / float(half_life_days)))


def compute_base_weight(
    conf: np.ndarray,
    rel: np.ndarray,
    engagement_E: np.ndarray,
    n_tickers: np.ndarray,
    config: SentimentConfig,
) -> np.ndarray:
    conf = np.clip(conf.astype(float), 0.0, 1.0)
    rel = np.clip(rel.astype(float), 0.0, 1.0)
    e = engagement_E.astype(float)

    w_quality = (conf ** float(config.gamma_conf)) * (rel ** float(config.delta_rel))
    w = w_quality * (1.0 + float(config.engagement_beta) * e)

    if config.divide_weight_by_num_tickers:
        n = np.maximum(n_tickers.astype(float), 1.0)
        w = w / n
    return w
