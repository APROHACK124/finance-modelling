from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cleaning import normalize_reddit_id


def connect_sqlite(db_path: str | Path) -> sqlite3.Connection:
    db_path = str(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    cur = conn.execute(q, (table_name,))
    return cur.fetchone() is not None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return [r["name"] for r in rows]


def read_table(
    conn: sqlite3.Connection,
    table_name: str,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    params: Optional[Sequence[object]] = None,
) -> pd.DataFrame:
    cols = "*" if columns is None else ", ".join(columns)
    q = f"SELECT {cols} FROM {table_name}"
    if where:
        q += f" WHERE {where}"
    return pd.read_sql_query(q, conn, params=params)


def _chunked(iterable: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


@dataclass
class RedditTables:
    predictions: str
    posts: str
    comments: str
    universe: str
    aliases: str


def load_predictions(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    cols = get_table_columns(conn, table)
    need = [
        "doc_id",
        "sentiment",
        "confidence_10",
        "relevance_10",
        "scope",
        "tickers_json",
        "evidence_json",
        "explanation",
        "subject",
    ]
    use_cols = [c for c in need if c in cols]
    df = read_table(conn, table, columns=use_cols, where="schema_version='thread_v2'")
    if "doc_id" not in df.columns:
        raise ValueError(f"{table} missing required column doc_id")
    df["thread_id"] = df["doc_id"].astype(str).map(normalize_reddit_id)
    return df


def load_posts(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    cols = get_table_columns(conn, table)
    need = [
        "post_id",
        "score",
        "created_utc",
        "subreddit",
        "num_comments",
        "upvote_ratio",
        "url",
        "permalink",
        # Optional snapshot timestamp (to avoid using info before it was collected)
        "fetch_date",
        "fetched_at",
    ]
    use_cols = [c for c in need if c in cols]
    df = read_table(conn, table, columns=use_cols)
    if "post_id" not in df.columns:
        raise ValueError(f"{table} missing required column post_id")
    df["thread_id"] = df["post_id"].astype(str).map(normalize_reddit_id)
    return df


def detect_comment_post_id_col(comment_cols: Sequence[str]) -> Optional[str]:
    # Common schemas: post_id, link_id, submission_id, parent_post_id
    for c in ("post_id", "link_id", "submission_id", "parent_post_id"):
        if c in comment_cols:
            return c
    return None


def load_top5_comment_scores(
    conn: sqlite3.Connection,
    table: str,
    thread_ids: Sequence[str],
    chunk_size: int = 900,
) -> pd.DataFrame:
    """
    Returns a dataframe: thread_id, top5_comment_score_sum_log1p

    Uses only comments linked to the given thread_ids. If the comments table schema
    doesn't contain a post-id column or score, returns empty.
    """
    if not thread_ids:
        return pd.DataFrame({"thread_id": [], "top5_comment_score_sum_log1p": []})

    cols = get_table_columns(conn, table)
    post_col = detect_comment_post_id_col(cols)
    if post_col is None or "score" not in cols:
        return pd.DataFrame({"thread_id": [], "top5_comment_score_sum_log1p": []})

    # Build both raw ids and t3_ prefixed ids for better match.
    raw_ids: List[str] = []
    for tid in thread_ids:
        tid = str(tid)
        raw_ids.append(tid)
        raw_ids.append(f"t3_{tid}")

    raw_ids = sorted(set(raw_ids))

    out_parts: List[pd.DataFrame] = []
    for chunk in _chunked(raw_ids, chunk_size):
        placeholders = ",".join(["?"] * len(chunk))
        q = f"SELECT {post_col} AS post_id_raw, score FROM {table} WHERE {post_col} IN ({placeholders})"
        part = pd.read_sql_query(q, conn, params=list(chunk))
        out_parts.append(part)

    if not out_parts:
        return pd.DataFrame({"thread_id": [], "top5_comment_score_sum_log1p": []})

    df = pd.concat(out_parts, ignore_index=True)
    if df.empty:
        return pd.DataFrame({"thread_id": [], "top5_comment_score_sum_log1p": []})

    df["thread_id"] = df["post_id_raw"].astype(str).map(normalize_reddit_id)

    # Keep only top-5 comment scores per thread.
    df["score_pos"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df.sort_values(["thread_id", "score_pos"], ascending=[True, False])
    df = df.groupby("thread_id", sort=False).head(5)

    # Engagement uses sum log1p(score_pos)
    df["log1p_score"] = np.log1p(df["score_pos"].astype(float))
    agg = df.groupby("thread_id", sort=False)["log1p_score"].sum().reset_index()
    agg = agg.rename(columns={"log1p_score": "top5_comment_score_sum_log1p"})
    return agg


def load_ticker_universe(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    cols = get_table_columns(conn, table)
    if "symbol" not in cols:
        raise ValueError(f"{table} missing required column symbol")
    df = read_table(conn, table, columns=["symbol"])
    df["ticker"] = df[['symbol']]
    df = df[['ticker']]
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna().drop_duplicates().sort_values("ticker")
    return df


def load_ticker_aliases(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    if not table_exists(conn, table):
        return pd.DataFrame(columns=["alias", "ticker"])

    cols = get_table_columns(conn, table)
    # Try to infer alias/ticker columns
    alias_col = None
    ticker_col = None
    for c in ("alias", "from", "src", "symbol_alias"):
        if c in cols:
            alias_col = c
            break
    for c in ("ticker", "to", "dst", "symbol"):
        if c in cols:
            ticker_col = c
            break
    if alias_col is None or ticker_col is None:
        # Fall back to first two columns
        if len(cols) >= 2:
            alias_col, ticker_col = cols[0], cols[1]
        else:
            return pd.DataFrame(columns=["alias", "ticker"])

    df = read_table(conn, table, columns=[alias_col, ticker_col])
    df = df.rename(columns={alias_col: "alias", ticker_col: "ticker"})
    df["alias"] = df["alias"].astype(str).str.upper().str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna().drop_duplicates()
    return df


def load_ticker_to_sector(
    conn: sqlite3.Connection,
    table: str,
    ticker_col: str = "ticker",
    sector_col: str = "sector",
) -> pd.DataFrame:
    if not table_exists(conn, table):
        return pd.DataFrame(columns=[ticker_col, sector_col])
    cols = get_table_columns(conn, table)
    if ticker_col not in cols or sector_col not in cols:
        return pd.DataFrame(columns=[ticker_col, sector_col])
    df = read_table(conn, table, columns=[ticker_col, sector_col])
    df[ticker_col] = df[ticker_col].astype(str).str.upper().str.strip()
    df[sector_col] = df[sector_col].astype(str).str.strip()
    df = df.dropna().drop_duplicates(subset=[ticker_col])
    return df


def write_dataframe(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table: str,
    if_exists: str = "replace",
    index: bool = False,
) -> None:
    df.to_sql(table, conn, if_exists=if_exists, index=index)
