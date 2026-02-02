from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .aggregate import assemble_feature_store, build_events, build_global_events, prepare_threads
from .config import SentimentConfig, load_config_or_default
from .explain import compute_top_threads
from .indices import assemble_market_index, attach_sector_sent_to_tickers, compute_sector_indices
from .io import (
    connect_sqlite,
    load_posts,
    load_predictions,
    load_ticker_aliases,
    load_ticker_to_sector,
    load_ticker_universe,
    load_top5_comment_scores,
    table_exists,
    write_dataframe,
)


def _parse_date(s: Optional[str]) -> Optional[date]:
    if s is None or s == "":
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Reddit-only sentiment feature store (daily).")
    p.add_argument("--db", required=True, help="Path to input sqlite db")
    p.add_argument("--out-db", default=None, help="Path to output sqlite db (default: overwrite input db)")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config (optional)")
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--if-exists", default="replace", choices=["replace", "append", "fail"], help="sqlite to_sql if_exists")
    p.add_argument("--no-top-threads", action="store_true", help="Disable top threads explainability columns")
    return p


def run(db_path: str, out_db_path: Optional[str], cfg_path: Optional[str], start_date: Optional[date], end_date: Optional[date], if_exists: str, no_top_threads: bool) -> None:
    cfg: SentimentConfig = load_config_or_default(cfg_path)
    if no_top_threads:
        cfg.top_k_threads = 0

    with connect_sqlite(db_path) as conn:
        if not table_exists(conn, cfg.table_predictions):
            raise RuntimeError(f"missing table: {cfg.table_predictions}")
        if not table_exists(conn, cfg.table_posts):
            raise RuntimeError(f"missing table: {cfg.table_posts}")
        if not table_exists(conn, cfg.table_universe):
            raise RuntimeError(f"missing table: {cfg.table_universe}")

        pred = load_predictions(conn, cfg.table_predictions)
        posts = load_posts(conn, cfg.table_posts)
        uni_df = load_ticker_universe(conn, cfg.table_universe)
        aliases_df = load_ticker_aliases(conn, cfg.table_aliases)

        universe = set(uni_df["ticker"].astype(str).str.upper())
        alias_map = {a: t for a, t in zip(aliases_df["alias"], aliases_df["ticker"])} if not aliases_df.empty else {}

        # Top-5 comment scores (optional)
        comment_scores = None
        if table_exists(conn, cfg.table_comments):
            comment_scores = load_top5_comment_scores(conn, cfg.table_comments, thread_ids=pred["thread_id"].tolist())
        else:
            comment_scores = None

        threads = prepare_threads(pred, posts, comment_scores=comment_scores, config=cfg)

        events = build_events(threads, universe=universe, alias_to_ticker=alias_map, config=cfg)
        global_events = build_global_events(threads, config=cfg)

    fs = assemble_feature_store(
        events,
        universe_tickers=sorted(universe) if cfg.sort_tickers else list(universe),
        config=cfg,
        start_date=start_date,
        end_date=end_date,
    )

    # Explainability: top threads
    if int(cfg.top_k_threads) > 0:
        top_df = compute_top_threads(events, config=cfg, start_date=start_date, end_date=end_date)
        if not top_df.empty:
            fs = fs.merge(top_df, on=["date", "ticker"], how="left")
        fs["top_positive_threads"] = fs.get("top_positive_threads").fillna("[]")
        fs["top_negative_threads"] = fs.get("top_negative_threads").fillna("[]")

    # Market index
    market = assemble_market_index(global_events, config=cfg, start_date=start_date, end_date=end_date)

    # Sector indices (optional)
    sector_index = pd.DataFrame()
    if cfg.table_ticker_sector:
        with connect_sqlite(db_path) as conn:
            t2s = load_ticker_to_sector(
                conn,
                cfg.table_ticker_sector,
                ticker_col=cfg.column_ticker_sector_ticker,
                sector_col=cfg.column_ticker_sector_sector,
            )
        if not t2s.empty:
            sector_index = compute_sector_indices(fs, t2s, config=cfg, ticker_col=cfg.column_ticker_sector_ticker, sector_col=cfg.column_ticker_sector_sector)
            fs = attach_sector_sent_to_tickers(fs, t2s, sector_index, config=cfg, ticker_col=cfg.column_ticker_sector_ticker, sector_col=cfg.column_ticker_sector_sector)

    # Deterministic ordering
    fs = fs.sort_values(["date", "ticker"]).reset_index(drop=True)
    market = market.sort_values(["date"]).reset_index(drop=True)
    if not sector_index.empty:
        sector_index = sector_index.sort_values(["date", "sector"]).reset_index(drop=True)

    out_db = db_path if out_db_path is None else out_db_path
    with connect_sqlite(out_db) as conn:
        write_dataframe(conn, fs, cfg.table_feature_store, if_exists=if_exists, index=False)
        write_dataframe(conn, market, cfg.table_market_index, if_exists=if_exists, index=False)
        if not sector_index.empty:
            write_dataframe(conn, sector_index, cfg.table_sector_index, if_exists=if_exists, index=False)

        # Save config snapshot (optional)
        cfg_snapshot = pd.DataFrame([{"key": "sentiment_feature_store_config", "value": json.dumps(cfg.to_dict(), sort_keys=True)}])
        cfg_snapshot.to_sql("_meta", conn, if_exists="append", index=False)


def main() -> None:
    p = build_arg_parser()
    args = p.parse_args()
    run(
        db_path=args.db,
        out_db_path=args.out_db,
        cfg_path=args.config,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        if_exists=args.if_exists,
        no_top_threads=bool(args.no_top_threads),
    )


if __name__ == "__main__":
    main()
