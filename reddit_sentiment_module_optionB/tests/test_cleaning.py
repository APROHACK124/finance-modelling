import pytest

from sentiment.cleaning import parse_tickers_json, clean_ticker_symbol, map_sentiment_to_s, to_unit_interval


def test_parse_tickers_json_dict():
    s = '{"AAPL": {"sentiment": "buy", "confidence": 0.8, "relevance": 0.9}, "msft": {"sentiment": "sell"}}'
    d = parse_tickers_json(s)
    assert "AAPL" in d
    assert "MSFT" in d
    assert d["AAPL"]["sentiment"] == "buy"


def test_parse_tickers_json_list():
    s = '["AAPL", "$msft", " brk.b "]'
    d = parse_tickers_json(s)
    assert set(d.keys()) == {"AAPL", "MSFT", "BRK.B"}


def test_map_sentiment():
    assert map_sentiment_to_s("buy") == 1
    assert map_sentiment_to_s("sell") == -1
    assert map_sentiment_to_s("neutral") == 0
    assert map_sentiment_to_s("positive") == 1
    assert map_sentiment_to_s("negative") == -1
    assert map_sentiment_to_s("???") is None


def test_to_unit_interval():
    assert to_unit_interval(0.5) == 0.5
    assert to_unit_interval(5) == 0.5
    assert to_unit_interval(50) == 0.5
    assert to_unit_interval(500) == 1.0
