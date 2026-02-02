import numpy as np

from sentiment.config import SentimentConfig
from sentiment.features import compute_base_weight, exp_time_decay


def test_exp_time_decay_half_life():
    # at age==half_life => 0.5
    d = exp_time_decay(np.array([0.0, 3.0]), half_life_days=3.0)
    assert np.isclose(d[0], 1.0)
    assert np.isclose(d[1], 0.5)


def test_compute_base_weight_divide_by_n():
    cfg = SentimentConfig()
    conf = np.array([1.0, 1.0])
    rel = np.array([1.0, 1.0])
    E = np.array([0.0, 0.0])
    n = np.array([1, 2])

    w = compute_base_weight(conf=conf, rel=rel, engagement_E=E, n_tickers=n, config=cfg)
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], 0.5)
