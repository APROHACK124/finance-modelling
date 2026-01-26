#!/usr/bin/env python3
"""
create_sentiment_views.py

Crea (si no existen) vistas para ver:
- posts + predicción
- comments + predicción
- news + predicción
- threads + predicción (doc_id=post_id)
- unificada (UNION ALL) con columnas comunes

Checa existencia en sqlite_master y permite --force para recrear.
"""

import os
import sqlite3
import argparse
from pathlib import Path
from typing import Optional, Dict


# =========================
# Path utils (robusto)
# =========================

def guess_runtime_dir() -> Path:
    """Directorio base del runtime: script dir si existe __file__, si no cwd (notebook/REPL)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def find_project_root(start: Optional[Path] = None,
                      markers=("app.py", "Dockerfile", "requirements.txt", "python_scripts", "data")) -> Path:
    """
    Sube directorios hasta hallar "marcadores" del proyecto.
    """
    start = (start or guess_runtime_dir()).resolve()
    for parent in [start] + list(start.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return start


def resolve_db_path(db_relative: str = "data/stock_data.db",
                    env_var: str = "INVESTINGS_DB_PATH") -> Path:
    """
    Resuelve la DB así:
    1) env INVESTINGS_DB_PATH si existe
    2) si db_relative es absoluta -> esa
    3) si es relativa -> project_root/db_relative
    """
    env = os.getenv(env_var)
    if env:
        return Path(os.path.expanduser(env)).resolve()

    p = Path(db_relative).expanduser()
    if p.is_absolute():
        return p.resolve()

    root = find_project_root()
    return (root / db_relative).resolve()


# =========================
# SQLite helpers
# =========================

def get_connection(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def object_exists(con: sqlite3.Connection, obj_type: str, name: str) -> bool:
    """
    obj_type: 'view' | 'table' | 'index'
    """
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type=? AND name=? LIMIT 1;",
        (obj_type, name)
    ).fetchone()
    return row is not None


def require_tables(con: sqlite3.Connection, required: list[str]) -> None:
    missing = [t for t in required if not object_exists(con, "table", t)]
    if missing:
        raise RuntimeError(f"Faltan tablas requeridas en la DB: {missing}")


# =========================
# View definitions
# =========================

PRED_TABLE = "_02_sentiment_predictions"

VIEWS_SQL: Dict[str, str] = {
    "v_reddit_posts_with_sentiment": f"""
    CREATE VIEW v_reddit_posts_with_sentiment AS
    SELECT
      p.post_id,
      p.subreddit,
      p.title,
      p.body,
      p.score,
      p.num_comments,
      p.url,
      p.created_utc,
      p.fetch_date,
      sp.model,
      sp.schema_version,
      sp.sentiment,
      sp.confidence_10,
      sp.relevance_10,
      sp.explanation,
      sp.usage_input_tokens,
      sp.usage_output_tokens,
      sp.usage_total_tokens,
      sp.cost_usd,
      sp.created_at AS predicted_at
    FROM reddit_posts p
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source = 'reddit_post'
     AND sp.doc_id  = p.post_id;
    """,

    "v_reddit_comments_with_sentiment": f"""
    CREATE VIEW v_reddit_comments_with_sentiment AS
    SELECT
      c.comment_id,
      c.post_id,
      c.body,
      c.score,
      c.created_utc,
      c.fetch_date,
      sp.model,
      sp.schema_version,
      sp.sentiment,
      sp.confidence_10,
      sp.relevance_10,
      sp.explanation,
      sp.usage_input_tokens,
      sp.usage_output_tokens,
      sp.usage_total_tokens,
      sp.cost_usd,
      sp.created_at AS predicted_at
    FROM reddit_comments c
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source = 'reddit_comment'
     AND sp.doc_id  = c.comment_id;
    """,

    "v_news_with_sentiment": f"""
    CREATE VIEW v_news_with_sentiment AS
    SELECT
      n.url,
      n.query_term,
      n.source_name,
      n.author,
      n.title,
      n.description,
      n.content,
      n.published_at,
      n.fetch_date,
      sp.model,
      sp.schema_version,
      sp.sentiment,
      sp.confidence_10,
      sp.relevance_10,
      sp.explanation,
      sp.usage_input_tokens,
      sp.usage_output_tokens,
      sp.usage_total_tokens,
      sp.cost_usd,
      sp.created_at AS predicted_at
    FROM news_articles n
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source = 'news'
     AND sp.doc_id  = n.url;
    """,

    "v_reddit_threads_with_sentiment": f"""
    CREATE VIEW v_reddit_threads_with_sentiment AS
    SELECT
      p.post_id,
      p.subreddit,
      p.title,
      p.body,
      p.score,
      p.num_comments,
      p.created_utc,
      p.fetch_date,
      sp.model,
      sp.schema_version,
      sp.sentiment,
      sp.confidence_10,
      sp.relevance_10,
      sp.explanation,
      sp.usage_input_tokens,
      sp.usage_output_tokens,
      sp.usage_total_tokens,
      sp.cost_usd,
      sp.created_at AS predicted_at
    FROM reddit_posts p
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source = 'reddit_thread'
     AND sp.doc_id  = p.post_id;
    """,

    "v_sentiment_unified": f"""
    CREATE VIEW v_sentiment_unified AS

    -- POSTS
    SELECT
      'reddit_post' AS source,
      p.post_id     AS doc_id,
      p.post_id     AS post_id,
      p.subreddit   AS subreddit,
      p.title       AS title,
      p.body        AS text,
      p.score       AS score,
      p.num_comments AS num_comments,
      p.created_utc AS time_utc,
      p.fetch_date  AS fetch_date,
      NULL          AS published_at,
      sp.model,
      sp.schema_version,
      sp.sentiment,
      sp.confidence_10,
      sp.relevance_10,
      sp.explanation,
      sp.usage_input_tokens,
      sp.usage_output_tokens,
      sp.usage_total_tokens,
      sp.cost_usd,
      sp.created_at AS predicted_at
    FROM reddit_posts p
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source='reddit_post' AND sp.doc_id=p.post_id

    UNION ALL

    -- COMMENTS
    SELECT
      'reddit_comment' AS source,
      c.comment_id     AS doc_id,
      c.post_id        AS post_id,
      NULL             AS subreddit,
      NULL             AS title,
      c.body           AS text,
      c.score          AS score,
      NULL             AS num_comments,
      c.created_utc    AS time_utc,
      c.fetch_date     AS fetch_date,
      NULL             AS published_at,
      sp.model,
      sp.schema_version,
      sp.sentiment,
      sp.confidence_10,
      sp.relevance_10,
      sp.explanation,
      sp.usage_input_tokens,
      sp.usage_output_tokens,
      sp.usage_total_tokens,
      sp.cost_usd,
      sp.created_at AS predicted_at
    FROM reddit_comments c
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source='reddit_comment' AND sp.doc_id=c.comment_id

    UNION ALL

    -- NEWS
    SELECT
      'news'          AS source,
      n.url           AS doc_id,
      NULL            AS post_id,
      NULL            AS subreddit,
      n.title         AS title,
      COALESCE(n.content, n.description) AS text,
      NULL            AS score,
      NULL            AS num_comments,
      CAST(strftime('%s', n.published_at) AS REAL) AS time_utc,
      n.fetch_date    AS fetch_date,
      n.published_at  AS published_at,
      sp.model,
      sp.schema_version,
      sp.sentiment,
      sp.confidence_10,
      sp.relevance_10,
      sp.explanation,
      sp.usage_input_tokens,
      sp.usage_output_tokens,
      sp.usage_total_tokens,
      sp.cost_usd,
      sp.created_at AS predicted_at
    FROM news_articles n
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source='news' AND sp.doc_id=n.url

    UNION ALL

    -- THREADS (doc_id = post_id)
    SELECT
      'reddit_thread' AS source,
      p.post_id       AS doc_id,
      p.post_id       AS post_id,
      p.subreddit     AS subreddit,
      p.title         AS title,
      p.body          AS text,
      p.score         AS score,
      p.num_comments  AS num_comments,
      p.created_utc   AS time_utc,
      p.fetch_date    AS fetch_date,
      NULL            AS published_at,
      sp.model,
      sp.schema_version,
      sp.sentiment,
      sp.confidence_10,
      sp.relevance_10,
      sp.explanation,
      sp.usage_input_tokens,
      sp.usage_output_tokens,
      sp.usage_total_tokens,
      sp.cost_usd,
      sp.created_at AS predicted_at
    FROM reddit_posts p
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source='reddit_thread' AND sp.doc_id=p.post_id;
    """,
}

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_pred_source_doc_model_ver_time
ON _02_sentiment_predictions(source, doc_id, model, schema_version, created_at);
"""


def create_or_verify_views(db_path: str, force: bool = False) -> None:
    con = get_connection(db_path)
    try:
        # Verifica que existan tablas base
        require_tables(con, [PRED_TABLE, "reddit_posts", "reddit_comments", "news_articles"])

        # Índice recomendado (idempotente)
        con.execute(INDEX_SQL)

        for view_name, create_sql in VIEWS_SQL.items():
            exists = object_exists(con, "view", view_name)
            if exists and not force:
                print(f"✅ View ya existe: {view_name} (skip)")
                continue

            if exists and force:
                print(f"♻️  Re-creando view: {view_name}")
                con.execute(f"DROP VIEW IF EXISTS {view_name};")
            else:
                print(f"➕ Creando view: {view_name}")

            con.execute(create_sql)

        con.commit()
        print("✅ Listo: vistas verificadas/creadas.")
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description="Create/verify sentiment views in SQLite.")
    parser.add_argument("--db", default="data/stock_data.db", help="Ruta a la DB (relativa al proyecto o absoluta).")
    parser.add_argument("--force", action="store_true", help="Drop & recreate views si ya existen.")
    args = parser.parse_args()

    db_path = str(resolve_db_path(args.db))
    print(f"[DB] {db_path}")

    create_or_verify_views(db_path, force=args.force)


if __name__ == "__main__":
    main()
