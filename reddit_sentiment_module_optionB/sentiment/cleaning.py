from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


_ID_PREFIX_RE = re.compile(r"^(t[0-9]_)")


def normalize_reddit_id(x: Any) -> str:
    """Normalize reddit fullname ids like 't3_abc123' -> 'abc123'."""
    if x is None:
        return ""
    s = str(x).strip()
    s = _ID_PREFIX_RE.sub("", s)
    return s


def parse_jsonish(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    text = str(s).strip()
    if text == "" or text.lower() == "null":
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def to_unit_interval(x: Any) -> float:
    """Coerce confidence/relevance to [0,1]."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except Exception:
        return 0.0
    if np.isnan(v) or np.isinf(v):
        return 0.0
    # Heuristics: accept 0..1, 0..10, 0..100
    if v <= 1.0:
        return float(np.clip(v, 0.0, 1.0))
    if v <= 10.0:
        return float(np.clip(v / 10.0, 0.0, 1.0))
    if v <= 100.0:
        return float(np.clip(v / 100.0, 0.0, 1.0))
    return 1.0


def map_sentiment_to_s(sent: Any) -> Optional[int]:
    """Map sentiment label to +1/-1/0. Returns None if unknown."""
    if sent is None:
        return None
    s = str(sent).strip().lower()
    if s in {"buy", "bull", "bullish", "long", "positive", "pos", "up"}:
        return 1
    if s in {"sell", "bear", "bearish", "short", "negative", "neg", "down"}:
        return -1
    if s in {"neutral", "hold", "flat", "mixed", "none"}:
        return 0
    return None


def clean_ticker_symbol(sym: Any) -> str:
    if sym is None:
        return ""
    s = str(sym).strip().upper()
    # Remove leading '$' and other common noise
    s = s.lstrip("$")
    # Keep only A-Z0-9 and dots (BRK.B) and dashes (RDS-A), strip others
    s = re.sub(r"[^A-Z0-9\.\-]", "", s)
    return s


def apply_aliases(ticker: str, alias_to_ticker: Mapping[str, str]) -> str:
    if not ticker:
        return ticker
    return alias_to_ticker.get(ticker, ticker)


def filter_valid_tickers(
    tickers: Iterable[str],
    universe: Optional[set[str]],
) -> List[str]:
    out: List[str] = []
    for t in tickers:
        if not t:
            continue
        if universe is None or t in universe:
            out.append(t)
    # Keep unique in order
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def parse_tickers_json(tickers_json: Any) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict[ticker] -> {sentiment, confidence, relevance, ...}
    Supports:
      - dict mapping ticker->dict
      - list of tickers (then empty dict per ticker)
    """
    obj = parse_jsonish(tickers_json)
    if obj is None:
        return {}

    if isinstance(obj, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in obj.items():
            tk = clean_ticker_symbol(k)
            if tk == "":
                continue
            if isinstance(v, dict):
                out[tk] = dict(v)
            else:
                out[tk] = {"value": v}
        return out

    if isinstance(obj, list):
        out = {}
        for item in obj:
            tk = clean_ticker_symbol(item)
            if tk:
                out[tk] = {}
        return out

    return {}


def extract_ticker_level_fields(
    row: Mapping[str, Any],
    per_ticker_payload: Mapping[str, Any],
) -> Tuple[Optional[int], float, float]:
    """Returns (s, conf, rel) for a specific ticker."""
    # sentiment precedence: per-ticker sentiment -> row['sentiment']
    sent = per_ticker_payload.get("sentiment", None)
    if sent is None and "sentiment" in row:
        sent = row.get("sentiment")
    s = map_sentiment_to_s(sent)

    conf = per_ticker_payload.get("confidence", None)
    if conf is None:
        conf = row.get("confidence_10", None)
    rel = per_ticker_payload.get("relevance", None)
    if rel is None:
        rel = row.get("relevance_10", None)

    return s, to_unit_interval(conf), to_unit_interval(rel)
