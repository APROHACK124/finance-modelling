from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import json
from pathlib import Path


def _as_int_tuple(x: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(i) for i in x)


@dataclass
class SentimentConfig:
    # Core
    horizons: Tuple[int, ...] = (5, 20, 60)
    half_life_days: Dict[int, float] = field(default_factory=lambda: {5: 3.0, 20: 10.0, 60: 30.0})

    # Engagement and weights
    engagement_alpha: float = 0.5
    engagement_beta: float = 0.15
    gamma_conf: float = 1.2
    delta_rel: float = 1.5
    eps: float = 1e-9

    # Robustness
    winsor_by_subreddit: bool = True
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    include_neutral: bool = False  # if True: neutral contributes s=0 (only volume/count); if False: drop neutral rows
    divide_weight_by_num_tickers: bool = True

    # "As-of" bucketing (avoid leakage)
    # Cutoff time is interpreted in asof_timezone, then converted to UTC internally.
    asof_timezone: str = "UTC"
    asof_cutoff_time: str = "23:59:59"  # HH:MM:SS in asof_timezone

    # Option B: maturity gating (reduce leakage from votes / "top comments")
    # The idea is to delay when a thread becomes eligible to contribute until it is at least
    # `min_age_hours` old (measured at the cutoff).
    #
    # Implementation detail: we shift the timestamp used for bucketing to
    #   created_utc + min_age_hours
    # while keeping time-decay based on the true created_utc.
    min_age_hours: float = 24.0

    # If your tables have a fetch timestamp (snapshot time), you can prevent using rows
    # before they were actually collected.
    use_fetch_date_as_availability: bool = True
    fetch_date_column: str = "fetch_date"
    fetch_date_assume_utc_if_naive: bool = True

    # Windows for discrete counts (mention_count/source_diversity)
    mention_window_days: Dict[int, int] = field(default_factory=lambda: {5: 5, 20: 20, 60: 60})

    # Explainability
    explain_horizon: int = 20
    explain_lookback_days: int = 30
    top_k_threads: int = 3

    # Residualization
    resid_min_obs: int = 20
    resid_min_volume: float = 1e-6

    # Market (global) index
    market_scopes: Tuple[str, ...] = ("macro", "rates_fx", "policy_regulation", "sector")
    # If a thread has 0 valid tickers, it can still contribute to global indices when scope is in:
    market_scopes_allow_no_ticker: Tuple[str, ...] = ("macro", "rates_fx", "policy_regulation", "crypto", "other")

    # Sorting / determinism
    sort_tickers: bool = True

    # Table names (sqlite)
    table_predictions: str = "_02_sentiment_predictions"
    table_posts: str = "reddit_posts"
    table_comments: str = "reddit_comments"
    table_universe: str = "_ref_ticker_universe"
    table_aliases: str = "_ref_ticker_aliases"

    # Output table names (sqlite)
    table_feature_store: str = "_03_sentiment_feature_store"
    table_market_index: str = "_03_sentiment_market_index"
    table_sector_index: str = "_03_sentiment_sector_sent"

    # Optional sector mapping (sqlite table)
    table_ticker_sector: Optional[str] = None
    column_ticker_sector_ticker: str = "ticker"
    column_ticker_sector_sector: str = "sector"

    def validate(self) -> None:
        if not self.horizons:
            raise ValueError("horizons cannot be empty")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"invalid horizons: {self.horizons}")
        for h in self.horizons:
            if h not in self.half_life_days:
                raise ValueError(f"half_life_days missing for horizon={h}")
            if self.half_life_days[h] <= 0:
                raise ValueError(f"half_life_days[{h}] must be > 0")
            if h not in self.mention_window_days:
                raise ValueError(f"mention_window_days missing for horizon={h}")
            if self.mention_window_days[h] <= 0:
                raise ValueError(f"mention_window_days[{h}] must be > 0")

        if self.engagement_alpha < 0:
            raise ValueError("engagement_alpha must be >= 0")
        if self.engagement_beta < 0:
            raise ValueError("engagement_beta must be >= 0")
        if self.gamma_conf <= 0:
            raise ValueError("gamma_conf must be > 0")
        if self.delta_rel <= 0:
            raise ValueError("delta_rel must be > 0")
        if not (0.0 <= self.winsor_lower_q < self.winsor_upper_q <= 1.0):
            raise ValueError("winsor quantiles must satisfy 0<=lower<upper<=1")
        if self.top_k_threads < 0:
            raise ValueError("top_k_threads must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["horizons"] = list(self.horizons)
        d["market_scopes"] = list(self.market_scopes)
        d["market_scopes_allow_no_ticker"] = list(self.market_scopes_allow_no_ticker)
        return d

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> "SentimentConfig":
        cfg = SentimentConfig()
        for k, v in m.items():
            if not hasattr(cfg, k):
                continue
            setattr(cfg, k, v)
        # Coerce some fields
        cfg.horizons = _as_int_tuple(cfg.horizons)
        cfg.market_scopes = tuple(cfg.market_scopes)
        cfg.market_scopes_allow_no_ticker = tuple(cfg.market_scopes_allow_no_ticker)
        cfg.validate()
        return cfg


def load_config(path: str | Path) -> SentimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config json must be an object/dict")
        return SentimentConfig.from_mapping(data)

    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML config requires PyYAML. Install pyyaml or use JSON/dict config."
            ) from e
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config yaml must be a mapping/dict")
        return SentimentConfig.from_mapping(data)

    raise ValueError(f"unsupported config extension: {path.suffix}")


def load_config_or_default(path: Optional[str | Path]) -> SentimentConfig:
    cfg = SentimentConfig() if path is None else load_config(path)
    cfg.validate()
    return cfg
