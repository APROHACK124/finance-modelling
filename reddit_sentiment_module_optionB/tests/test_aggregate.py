from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from sentiment.aggregate import assemble_feature_store
from sentiment.config import SentimentConfig


def test_assemble_feature_store_small():
    cfg = SentimentConfig(horizons=(5,), half_life_days={5: 3.0}, mention_window_days={5: 5}, asof_timezone="UTC", asof_cutoff_time="23:59:59")
    cfg.resid_min_obs = 1  # allow tiny regression
    cfg.resid_min_volume = 0.0

    d1 = date(2026, 1, 1)
    d2 = date(2026, 1, 2)

    ev = pd.DataFrame(
        [
            {
                "feature_date": d1,
                "ticker": "A",
                "thread_id": "t1",
                "subreddit": "s1",
                "created_dt_utc": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
                "cutoff_dt_utc": datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
                "intra_age_days": 0.0,
                "s": 1,
                "conf": 1.0,
                "rel": 1.0,
                "n_valid_tickers": 1,
                "E": 0.0,
                "base_weight": 1.0,
            },
            {
                "feature_date": d1,
                "ticker": "B",
                "thread_id": "t2",
                "subreddit": "s2",
                "created_dt_utc": datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc),
                "cutoff_dt_utc": datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
                "intra_age_days": 0.0,
                "s": -1,
                "conf": 1.0,
                "rel": 1.0,
                "n_valid_tickers": 1,
                "E": 0.0,
                "base_weight": 1.0,
            },
        ]
    )

    fs = assemble_feature_store(ev, universe_tickers=["A", "B"], config=cfg, start_date=d1, end_date=d2)
    # Two dates * 2 tickers = 4 rows
    assert len(fs) == 4

    # Check date1
    a1 = fs[(fs["date"] == "2026-01-01") & (fs["ticker"] == "A")].iloc[0]
    b1 = fs[(fs["date"] == "2026-01-01") & (fs["ticker"] == "B")].iloc[0]
    assert np.isclose(a1["sent_net_5"], 1.0)
    assert np.isclose(b1["sent_net_5"], -1.0)
    assert np.isclose(a1["bull_ratio_5"], 1.0)
    assert np.isclose(b1["bull_ratio_5"], 0.0)

    # Rank percentiles with 2 tickers: bottom=0.5, top=1.0
    assert np.isclose(a1["sent_rank_5"], 1.0)
    assert np.isclose(b1["sent_rank_5"], 0.5)

    # date2 should keep same sign (decay cancels in ratio)
    a2 = fs[(fs["date"] == "2026-01-02") & (fs["ticker"] == "A")].iloc[0]
    b2 = fs[(fs["date"] == "2026-01-02") & (fs["ticker"] == "B")].iloc[0]
    assert np.isclose(a2["sent_net_5"], 1.0, atol=1e-6)
    assert np.isclose(b2["sent_net_5"], -1.0, atol=1e-6)

    # Mention count: last 5 days includes date1 events, so still 1 on date2
    assert np.isclose(a2["mention_count_5"], 1.0)
    assert np.isclose(b2["mention_count_5"], 1.0)

    # Source diversity: last 5 days includes 1 subreddit each
    assert np.isclose(a2["source_diversity_5"], 1.0)
    assert np.isclose(b2["source_diversity_5"], 1.0)
