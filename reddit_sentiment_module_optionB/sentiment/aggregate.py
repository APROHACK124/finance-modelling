from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .cleaning import (
    apply_aliases,
    extract_ticker_level_fields,
    filter_valid_tickers,
    parse_tickers_json,
)
from .config import SentimentConfig
from .features import AsOfCalendar, compute_base_weight, compute_engagement, exp_time_decay, winsorize_series_by_group
from .normalize import percentile_rank_cs, residualize_by_volume, zscore_cs


def prepare_threads(
    predictions: pd.DataFrame,
    posts: pd.DataFrame,
    comment_scores: Optional[pd.DataFrame],
    config: SentimentConfig,
) -> pd.DataFrame:
    """
    Join predictions with reddit_posts (+ optional top5 comment scores),
    compute engagement E, and as-of assignment.
    """
    if "thread_id" not in predictions.columns:
        raise ValueError("predictions must include thread_id")
    if "thread_id" not in posts.columns:
        raise ValueError("posts must include thread_id")

    # Join to get created_utc, subreddit, post score (likes)
    cols_post = [
        c
        for c in ["thread_id", "created_utc", "subreddit", "score", "num_comments", "upvote_ratio", "url", "permalink"]
        if c in posts.columns
    ]
    # Optional snapshot timestamp columns (if present)
    for c in ("fetch_date", "fetched_at"):
        if c in posts.columns and c not in cols_post:
            cols_post.append(c)
    post_meta = posts[cols_post].drop_duplicates(subset=["thread_id"])
    df = predictions.merge(post_meta, on="thread_id", how="left", suffixes=("", "_post"))

    if "created_utc" not in df.columns:
        raise ValueError("posts table must provide created_utc")

    # Ensure created_utc is numeric and drop rows without it (cannot time-bucket without it)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df.dropna(subset=["created_utc"]).copy()

    # Optional comment score sum
    if comment_scores is not None and not comment_scores.empty:
        df = df.merge(comment_scores, on="thread_id", how="left")
    if "top5_comment_score_sum_log1p" not in df.columns:
        df["top5_comment_score_sum_log1p"] = 0.0

    # Engagement
    if "score" not in df.columns:
        df["score"] = 0.0
    df["E_raw"] = compute_engagement(df["score"], df["top5_comment_score_sum_log1p"], config)

    # Winsorize by subreddit to reduce viral domination
    if "subreddit" not in df.columns:
        df["subreddit"] = "unknown"
    df["subreddit"] = df["subreddit"].fillna("unknown").astype(str)

    if config.winsor_by_subreddit:
        df["E"] = winsorize_series_by_group(
            df["E_raw"],
            df["subreddit"],
            lower_q=float(config.winsor_lower_q),
            upper_q=float(config.winsor_upper_q),
        )
    else:
        df["E"] = df["E_raw"]

    # created datetime in UTC (true timestamp)
    df["created_dt_utc"] = df["created_utc"].apply(lambda x: datetime.fromtimestamp(float(x), tz=timezone.utc))

    # Option B: maturity gating. Delay eligibility by min_age_hours.
    min_age_h = float(getattr(config, "min_age_hours", 0.0) or 0.0)
    eligible_dt_utc = df["created_dt_utc"]
    if min_age_h > 0:
        eligible_dt_utc = eligible_dt_utc + pd.to_timedelta(min_age_h, unit="h")

    # Optional: enforce availability based on snapshot time (fetch_date/fetched_at) if present.
    if getattr(config, "use_fetch_date_as_availability", False):
        col = str(getattr(config, "fetch_date_column", "fetch_date"))
        fetch_col = col if col in df.columns else ("fetched_at" if "fetched_at" in df.columns else None)
        if fetch_col is not None:
            fetch_dt = pd.to_datetime(df[fetch_col], errors="coerce")
            # If naive timestamp and configured, assume UTC.
            if getattr(config, "fetch_date_assume_utc_if_naive", True) and getattr(fetch_dt.dt, "tz", None) is None:
                fetch_dt = fetch_dt.dt.tz_localize(timezone.utc)
            else:
                # Force to UTC if tz-aware
                try:
                    fetch_dt = fetch_dt.dt.tz_convert(timezone.utc)
                except Exception:
                    pass

            df["fetch_dt_utc"] = fetch_dt
            eligible_dt_utc = eligible_dt_utc.where(fetch_dt.isna() | (fetch_dt <= eligible_dt_utc), fetch_dt)
        else:
            df["fetch_dt_utc"] = pd.NaT
    else:
        df["fetch_dt_utc"] = pd.NaT

    # As-of bucketing: assign feature_date based on eligibility time (not the raw creation time).
    cal = AsOfCalendar.from_config(config)
    df["feature_date"] = eligible_dt_utc.apply(lambda ts: cal.assign_feature_date(ts.timestamp())).astype("object")

    # Store cutoff datetime in UTC for each feature_date (for reproducibility / explainability)
    unique_dates = sorted({d for d in df["feature_date"].tolist() if isinstance(d, date)})
    cutoff_map = {d: cal.cutoff_datetime_utc(d) for d in unique_dates}
    df["cutoff_dt_utc"] = df["feature_date"].map(cutoff_map)

    # intra-day age to cutoff (days) for its assigned feature_date
    df["intra_age_days"] = (
        (df["cutoff_dt_utc"] - df["created_dt_utc"]).dt.total_seconds().astype(float) / 86400.0
    ).clip(lower=0.0)

    return df


def build_events(
    threads: pd.DataFrame,
    universe: set[str],
    alias_to_ticker: Mapping[str, str],
    config: SentimentConfig,
) -> pd.DataFrame:
    """
    Explode threads into per-ticker events.

    Output columns:
      feature_date (date), ticker, thread_id, subreddit, scope,
      created_dt_utc, cutoff_dt_utc, intra_age_days,
      s, conf, rel, n_valid_tickers, base_weight
    """
    records: List[Dict[str, object]] = []

    def _payload_score(p: Mapping[str, object]) -> float:
        try:
            c = float(p.get("confidence", 0.0))  # type: ignore[arg-type]
        except Exception:
            c = 0.0
        try:
            r = float(p.get("relevance", 0.0))  # type: ignore[arg-type]
        except Exception:
            r = 0.0
        return c * r

    for _, row in threads.iterrows():
        row_d = row.to_dict()
        raw_payload = parse_tickers_json(row_d.get("tickers_json"))

        # Canonicalize tickers (aliases) while preserving per-ticker payload
        canon_payload: Dict[str, Dict[str, object]] = {}
        for raw_t, payload in raw_payload.items():
            canon = apply_aliases(raw_t, alias_to_ticker)
            if not canon or canon not in universe:
                continue
            p = payload if isinstance(payload, dict) else {}
            if canon in canon_payload:
                # Keep the "best" payload by confidence*relevance (if present)
                if _payload_score(p) > _payload_score(canon_payload[canon]):
                    canon_payload[canon] = dict(p)
            else:
                canon_payload[canon] = dict(p)

        tickers = list(canon_payload.keys())
        if not tickers:
            continue

        # Deterministic ordering within thread
        tickers = sorted(tickers)

        n = len(tickers)
        for t in tickers:
            payload = canon_payload.get(t, {})
            s, conf, rel = extract_ticker_level_fields(row_d, payload)

            if s is None:
                continue
            if (s == 0) and (not config.include_neutral):
                continue

            records.append(
                {
                    "feature_date": row_d.get("feature_date"),
                    "ticker": t,
                    "thread_id": row_d.get("thread_id"),
                    "subreddit": row_d.get("subreddit", "unknown"),
                    "scope": row_d.get("scope", "other"),
                    "created_dt_utc": row_d.get("created_dt_utc"),
                    "cutoff_dt_utc": row_d.get("cutoff_dt_utc"),
                    "intra_age_days": float(row_d.get("intra_age_days") or 0.0),
                    "s": int(s),
                    "conf": float(conf),
                    "rel": float(rel),
                    "n_valid_tickers": int(n),
                    "E": float(row_d.get("E") or 0.0),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "feature_date",
                "ticker",
                "thread_id",
                "subreddit",
                "scope",
                "created_dt_utc",
                "cutoff_dt_utc",
                "intra_age_days",
                "s",
                "conf",
                "rel",
                "n_valid_tickers",
                "E",
                "base_weight",
            ]
        )

    ev = pd.DataFrame.from_records(records)
    # base weight per ticker-event (no time decay yet)
    ev["base_weight"] = compute_base_weight(
        conf=ev["conf"].to_numpy(),
        rel=ev["rel"].to_numpy(),
        engagement_E=ev["E"].to_numpy(),
        n_tickers=ev["n_valid_tickers"].to_numpy(),
        config=config,
    )
    return ev


def build_global_events(threads: pd.DataFrame, config: SentimentConfig) -> pd.DataFrame:
    """
    Thread-level events for global indices (market/macro).
    Uses thread-level sentiment/conf/rel.
    """
    from .cleaning import map_sentiment_to_s, to_unit_interval

    records: List[Dict[str, object]] = []
    market_scopes = set(config.market_scopes)

    for _, row in threads.iterrows():
        scope = str(row.get("scope", "other"))
        if scope not in market_scopes:
            continue

        s = map_sentiment_to_s(row.get("sentiment"))
        if s is None:
            continue
        if (s == 0) and (not config.include_neutral):
            continue

        conf = to_unit_interval(row.get("confidence_10"))
        rel = to_unit_interval(row.get("relevance_10"))

        records.append(
            {
                "feature_date": row.get("feature_date"),
                "thread_id": row.get("thread_id"),
                "subreddit": row.get("subreddit", "unknown"),
                "scope": scope,
                "created_dt_utc": row.get("created_dt_utc"),
                "cutoff_dt_utc": row.get("cutoff_dt_utc"),
                "intra_age_days": float(row.get("intra_age_days") or 0.0),
                "s": int(s),
                "conf": float(conf),
                "rel": float(rel),
                "E": float(row.get("E") or 0.0),
                "n_valid_tickers": 1,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "feature_date",
                "thread_id",
                "subreddit",
                "scope",
                "created_dt_utc",
                "cutoff_dt_utc",
                "intra_age_days",
                "s",
                "conf",
                "rel",
                "E",
                "n_valid_tickers",
                "base_weight",
            ]
        )

    df = pd.DataFrame.from_records(records)
    df["base_weight"] = compute_base_weight(
        conf=df["conf"].to_numpy(),
        rel=df["rel"].to_numpy(),
        engagement_E=df["E"].to_numpy(),
        n_tickers=df["n_valid_tickers"].to_numpy(),
        config=config,
    )
    return df


def _date_range(start: date, end: date) -> List[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def compute_daily_adds(
    events: pd.DataFrame,
    horizon: int,
    config: SentimentConfig,
    date_index: Sequence[date],
    keys: Sequence[str],
    key_col: str,
) -> Dict[str, np.ndarray]:
    """
    For a given horizon, compute per-day 'adds' already decayed from intra-day timestamp to cutoff.

    Returns dict of arrays with shape (num_dates, num_keys).
    Variables:
      - numer_add: sum(s * base_weight * intra_decay)
      - denom_add: sum(base_weight * intra_decay)
      - bull_add: sum(base_weight * intra_decay where s=+1)
      - bear_add: sum(base_weight * intra_decay where s=-1)
      - conf_num_add: sum(conf * base_weight * intra_decay)
      - rel_num_add: sum(rel * base_weight * intra_decay)
    """
    n_dates = len(date_index)
    n_keys = len(keys)

    if events.empty:
        zeros = np.zeros((n_dates, n_keys), dtype=np.float64)
        return {
            "numer_add": zeros.copy(),
            "denom_add": zeros.copy(),
            "bull_add": zeros.copy(),
            "bear_add": zeros.copy(),
            "conf_num_add": zeros.copy(),
            "rel_num_add": zeros.copy(),
        }

    half_life = float(config.half_life_days[horizon])
    intra_decay = exp_time_decay(events["intra_age_days"].to_numpy(dtype=float), half_life_days=half_life)

    w = events["base_weight"].to_numpy(dtype=float) * intra_decay
    s = events["s"].to_numpy(dtype=int)

    numer = w * s.astype(float)
    denom = w
    bull = np.where(s == 1, w, 0.0)
    bear = np.where(s == -1, w, 0.0)
    conf_num = w * events["conf"].to_numpy(dtype=float)
    rel_num = w * events["rel"].to_numpy(dtype=float)

    df = events[["feature_date", key_col]].copy()
    df["numer"] = numer
    df["denom"] = denom
    df["bull"] = bull
    df["bear"] = bear
    df["conf_num"] = conf_num
    df["rel_num"] = rel_num

    agg = df.groupby(["feature_date", key_col], sort=False).sum(numeric_only=True).reset_index()

    date_pos = {d: i for i, d in enumerate(date_index)}
    key_pos = {k: j for j, k in enumerate(keys)}

    out = {name: np.zeros((n_dates, n_keys), dtype=np.float64) for name in ["numer", "denom", "bull", "bear", "conf_num", "rel_num"]}

    for row in agg.itertuples(index=False):
        d = getattr(row, "feature_date")
        k = getattr(row, key_col)
        if d not in date_pos or k not in key_pos:
            continue
        i = date_pos[d]
        j = key_pos[k]
        out["numer"][i, j] = float(getattr(row, "numer"))
        out["denom"][i, j] = float(getattr(row, "denom"))
        out["bull"][i, j] = float(getattr(row, "bull"))
        out["bear"][i, j] = float(getattr(row, "bear"))
        out["conf_num"][i, j] = float(getattr(row, "conf_num"))
        out["rel_num"][i, j] = float(getattr(row, "rel_num"))

    return {
        "numer_add": out["numer"],
        "denom_add": out["denom"],
        "bull_add": out["bull"],
        "bear_add": out["bear"],
        "conf_num_add": out["conf_num"],
        "rel_num_add": out["rel_num"],
    }


def compute_step_decays(date_index: Sequence[date], config: SentimentConfig, half_life_days: float) -> np.ndarray:
    """
    Step-wise decay factors between consecutive cutoffs in UTC.
    Handles DST when asof_timezone != UTC.
    Returns array length n_dates where decays[i] is decay from i-1 -> i (decays[0]=1).
    """
    cal = AsOfCalendar.from_config(config)
    cutoffs = [cal.cutoff_datetime_utc(d) for d in date_index]
    decays = np.ones((len(date_index),), dtype=np.float64)
    for i in range(1, len(date_index)):
        dt = (cutoffs[i] - cutoffs[i - 1]).total_seconds() / 86400.0
        decays[i] = float(np.exp(-np.log(2.0) * (dt / float(half_life_days))))
    return decays


def compute_decayed_running_sums(
    daily_adds: Dict[str, np.ndarray],
    step_decays: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Apply recursive decay across days: run_t = decay_t*run_{t-1} + add_t."""
    n_dates, n_keys = daily_adds["denom_add"].shape
    out = {k: np.zeros((n_dates, n_keys), dtype=np.float64) for k in daily_adds.keys()}
    run = {k: np.zeros((n_keys,), dtype=np.float64) for k in daily_adds.keys()}
    for i in range(n_dates):
        decay = float(step_decays[i])
        for k in run.keys():
            run[k] *= decay
            run[k] += daily_adds[k][i]
            out[k][i] = run[k]
    return out


def compute_daily_thread_counts(
    events: pd.DataFrame,
    date_index: Sequence[date],
    tickers: Sequence[str],
) -> np.ndarray:
    """Matrix (dates x tickers) with daily count of threads for each date/ticker."""
    n_dates = len(date_index)
    n_tickers = len(tickers)
    if events.empty:
        return np.zeros((n_dates, n_tickers), dtype=np.int32)

    tmp = events[["feature_date", "ticker", "thread_id"]].drop_duplicates()
    cnt = tmp.groupby(["feature_date", "ticker"], sort=False)["thread_id"].count().reset_index(name="cnt")
    date_pos = {d: i for i, d in enumerate(date_index)}
    t_pos = {t: j for j, t in enumerate(tickers)}

    out = np.zeros((n_dates, n_tickers), dtype=np.int32)
    for row in cnt.itertuples(index=False):
        d = getattr(row, "feature_date")
        t = getattr(row, "ticker")
        if d not in date_pos or t not in t_pos:
            continue
        out[date_pos[d], t_pos[t]] = int(getattr(row, "cnt"))
    return out


def rolling_sum_counts(counts: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum over axis=0 (dates). counts shape: dates x keys"""
    if window <= 1:
        return counts.astype(np.float64)
    csum = np.cumsum(counts, axis=0, dtype=np.int64)
    out = csum.copy()
    out[window:] = csum[window:] - csum[:-window]
    return out.astype(np.float64)


def compute_source_diversity(
    events: pd.DataFrame,
    date_index: Sequence[date],
    tickers: Sequence[str],
    window_days: int,
) -> np.ndarray:
    """
    Rolling number of unique subreddits in the last `window_days` (inclusive), per ticker-date.

    Efficient interval-union approach:
      For each (ticker, subreddit), occurrences at dates d imply it is counted for dates in [d, d+window_days-1].
      We union intervals per pair and accumulate via difference arrays per ticker.
    """
    n_dates = len(date_index)
    n_tickers = len(tickers)
    if events.empty:
        return np.zeros((n_dates, n_tickers), dtype=np.int32)

    date_pos = {d: i for i, d in enumerate(date_index)}
    t_pos = {t: j for j, t in enumerate(tickers)}

    occ = (
        events[["ticker", "subreddit", "feature_date"]]
        .dropna()
        .drop_duplicates()
        .groupby(["ticker", "subreddit"], sort=False)["feature_date"]
        .apply(list)
    )

    diff = np.zeros((n_tickers, n_dates + 1), dtype=np.int32)

    for (ticker, subreddit), dates_list in occ.items():
        if ticker not in t_pos:
            continue
        col = t_pos[ticker]
        positions = sorted(date_pos[d] for d in dates_list if d in date_pos)
        if not positions:
            continue

        cur_start = None
        cur_end = None
        for p in positions:
            start = p
            end = min(p + window_days - 1, n_dates - 1)
            if cur_start is None:
                cur_start, cur_end = start, end
            elif start <= (cur_end + 1):
                cur_end = max(cur_end, end)
            else:
                diff[col, cur_start] += 1
                if cur_end + 1 < n_dates:
                    diff[col, cur_end + 1] -= 1
                cur_start, cur_end = start, end

        if cur_start is not None and cur_end is not None:
            diff[col, cur_start] += 1
            if cur_end + 1 < n_dates:
                diff[col, cur_end + 1] -= 1

    src = np.cumsum(diff[:, :n_dates], axis=1).T  # dates x tickers
    return src


def compute_sentiment_momentum(sent_net: np.ndarray, step_decays: np.ndarray) -> np.ndarray:
    """Momentum vs EMA with step-wise decay factors (per day)."""
    n_dates, n_tickers = sent_net.shape
    ema = np.zeros_like(sent_net, dtype=np.float64)
    prev = np.zeros((n_tickers,), dtype=np.float64)
    for i in range(n_dates):
        if i == 0:
            prev = sent_net[i].copy()
        else:
            decay = float(step_decays[i])
            alpha = 1.0 - decay
            prev = decay * prev + alpha * sent_net[i]
        ema[i] = prev
    return sent_net - ema


def assemble_feature_store(
    events: pd.DataFrame,
    universe_tickers: Sequence[str],
    config: SentimentConfig,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Compute daily per-ticker features for all horizons.

    Returns a long dataframe with 1 row per (date,ticker).
    """
    config.validate()

    tickers = sorted({t for t in universe_tickers}) if config.sort_tickers else list(universe_tickers)

    if events.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    min_d = min(events["feature_date"])
    max_d = max(events["feature_date"])
    start_date = min_d if start_date is None else start_date
    end_date = max_d if end_date is None else end_date
    if start_date > end_date:
        raise ValueError("start_date > end_date")

    date_index = _date_range(start_date, end_date)
    n_dates = len(date_index)
    n_tickers = len(tickers)

    # Discrete counts/diversity inputs once
    counts_daily = compute_daily_thread_counts(events, date_index=date_index, tickers=tickers)

    base = pd.MultiIndex.from_product(
        [[d.isoformat() for d in date_index], tickers],
        names=["date", "ticker"],
    )
    out_df = pd.DataFrame(index=base)

    for h in config.horizons:
        half_life = float(config.half_life_days[h])
        step_decays = compute_step_decays(date_index=date_index, config=config, half_life_days=half_life)

        daily_adds = compute_daily_adds(
            events,
            horizon=h,
            config=config,
            date_index=date_index,
            keys=tickers,
            key_col="ticker",
        )
        running = compute_decayed_running_sums(daily_adds, step_decays=step_decays)

        numer = running["numer_add"]
        denom = running["denom_add"]
        bull = running["bull_add"]
        bear = running["bear_add"]
        conf_num = running["conf_num_add"]
        rel_num = running["rel_num_add"]

        sent_net = numer / (denom + float(config.eps))
        bull_ratio = bull / (bull + bear + float(config.eps))
        sent_volume = denom
        avg_conf = conf_num / (denom + float(config.eps))
        avg_rel = rel_num / (denom + float(config.eps))
        disagree = 1.0 - np.abs(sent_net)

        # Momentum vs EMA(sent_net)
        sent_mom = compute_sentiment_momentum(sent_net, step_decays=step_decays)

        # Mention count rolling sum
        window = int(config.mention_window_days[h])
        mention_count = rolling_sum_counts(counts_daily, window=window)

        # Source diversity (unique subreddits) rolling
        source_div = compute_source_diversity(events, date_index=date_index, tickers=tickers, window_days=window).astype(np.float64)

        # Cross-section normalization
        sent_rank = percentile_rank_cs(sent_net)
        sent_z = zscore_cs(sent_net)
        sent_resid = residualize_by_volume(
            sent_net=sent_net,
            sent_volume=sent_volume,
            min_volume=float(config.resid_min_volume),
            min_obs=int(config.resid_min_obs),
        )

        def _flat(a: np.ndarray) -> np.ndarray:
            return a.reshape(n_dates * n_tickers)

        out_df[f"sent_net_{h}"] = _flat(sent_net)
        out_df[f"bull_ratio_{h}"] = _flat(bull_ratio)
        out_df[f"sent_volume_{h}"] = _flat(sent_volume)
        out_df[f"mention_count_{h}"] = _flat(mention_count)
        out_df[f"source_diversity_{h}"] = _flat(source_div)
        out_df[f"avg_conf_{h}"] = _flat(avg_conf)
        out_df[f"avg_rel_{h}"] = _flat(avg_rel)
        out_df[f"disagree_{h}"] = _flat(disagree)
        out_df[f"sent_mom_{h}"] = _flat(sent_mom)
        out_df[f"sent_rank_{h}"] = _flat(sent_rank)
        out_df[f"sent_z_{h}"] = _flat(sent_z)
        out_df[f"sent_resid_{h}"] = _flat(sent_resid)

    out_df = out_df.reset_index()
    return out_df
