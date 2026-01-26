# preprocess_store_database.py

# %% [markdown]
# ### 02_SQLite_IO_and_Batching
# 
# Este bloque crea utilidades para:
# - Conectarse a SQLite con **WAL** y `synchronous=NORMAL`.
# - Crear tablas **_01_reddit_posts_preprocessed**, **_01_reddit_comments_preprocessed**, **_01_news_articles_preprocessed**.
# - Procesar texto en **lotes** con **checkpoint** (`_etl_checkpoints`) para reanudar.
# - Hacer un **smoke test** que migra N filas por tabla.
# 
# **Notas de modelado**
# - Las tablas preprocesadas **no duplican** los metadatos crudos (p. ej. `score`, `url` de post), solo almacenan salidas de NLP + trazabilidad mínima.
# - `lang/lang_conf`: se detectan sobre el **texto combinado** (p. ej., `title + body`).
# - `content_hash`: hash estable del `combined_ml` para *dedup*/auditoría futura.
# - PK de preprocesadas = PK de crudas + `FOREIGN KEY ... ON DELETE CASCADE`, así mantienes integridad referencial.
# 

# %%
import os
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from python_scripts.utilities_text_processing import (
    text_preprocessing,
    text_preprocessing_llm,
    text_preprocessing_ml,
)


# Default route
DB_PATH_DEFAULT = os.path.expanduser(
    "~/Desktop/all_folders/Investings_project/app/data/stock_data.db"
)

PROC_VERSION = "01_preprocess_v0.1"



# %%
from pathlib import Path
import os

# chatGPT part

def guess_runtime_dir() -> Path:
    """
    Devuelve el directorio 'base' desde el cual debemos resolver rutas relativas.
    - Si el código corre como script (.py): usa la carpeta de ese archivo (__file__).
    - Si corre en notebook (no existe __file__): usa el directorio de trabajo actual (Path.cwd()).
      En Jupyter/Lab esto suele ser la carpeta donde está el .ipynb, salvo que hayas cambiado el cwd.
    """
    try:
        # Modo script / módulo
        return Path(__file__).resolve().parent
    except NameError:
        # Modo notebook / consola interactiva
        return Path.cwd().resolve()


def find_project_root(start: Optional[Path] = None,
                      markers = ("app.py", "Dockerfile", "requirements.txt", "python_scripts", "data")) -> Path:
    """
    Sube por la jerarquía de carpetas hasta encontrar un directorio que contenga
    al menos uno de los 'markers' típicos de tu proyecto (p. ej., app.py, data/).
    Si no encuentra un candidato, devuelve 'start'.
    """
    start = (start or guess_runtime_dir()).resolve()
    for parent in [start] + list(start.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return start


def resolve_db_path(db_relative: str = "data/stock_data.db",
                    create_dirs: bool = False,
                    env_var: str = "INVESTINGS_DB_PATH") -> Path:
    """
    Resuelve la ruta absoluta de la DB, con esta prioridad:
    1) Variable de entorno INVESTINGS_DB_PATH (si está definida).
    2) Si 'db_relative' es absoluto, úsalo tal cual.
    3) Une el 'project_root' (descubierto) con 'db_relative' (por defecto data/stock_data.db).

    Parámetros:
    - db_relative: ruta relativa a la raíz del proyecto (o absoluta si quieres forzar).
    - create_dirs: si True, crea la carpeta padre de la DB si no existe.
    - env_var: variable de entorno para sobre-escribir la ruta cuando corras en otras máquinas.

    Devuelve:
    - Path absoluto a la base de datos.
    """
    # 1) Override por variable de entorno
    env = os.getenv(env_var)
    if env:
        p = Path(os.path.expanduser(env)).resolve()
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 2) ¿db_relative ya es absoluto?
    p = Path(db_relative).expanduser()
    if p.is_absolute():
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    # 3) Resolver relativo a la raíz del proyecto
    root = find_project_root()
    full = (root / db_relative).resolve()
    if create_dirs:
        full.parent.mkdir(parents=True, exist_ok=True)
    return full


# 👉 Define/actualiza el valor por defecto que usa el resto del notebook:
DB_PATH_DEFAULT = str(resolve_db_path("data/stock_data.db"))
print("[DB_PATH_DEFAULT]", DB_PATH_DEFAULT)


# %%

import time
import sqlite3
from typing import Any, Callable, List, Optional, Tuple

def get_connection(db_path: Optional[str] = None,
                   *,
                   timeout_s: float = 30.0,
                   busy_timeout_ms: int = 30_000) -> sqlite3.Connection:
    """
    - timeout_s: cuánto esperar cuando hay locks (a nivel sqlite3.connect)
    - busy_timeout_ms: equivalente PRAGMA busy_timeout
    """
    if db_path is None:
        path = resolve_db_path("data/stock_data.db")
    else:
        path = resolve_db_path(db_path)

    con = sqlite3.connect(str(path), timeout=timeout_s)

    # Espera cuando SQLite esté ocupado / locked
    con.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)};")

    # PRAGMAs de integridad + concurrencia
    con.execute("PRAGMA foreign_keys = ON;")
    mode = con.execute("PRAGMA journal_mode;").fetchone()[0]
    if str(mode).upper() != "WAL":
        con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con



def _is_lock_error(e: BaseException) -> bool:
    if not isinstance(e, sqlite3.OperationalError):
        return False
    msg = str(e).lower()
    return ("locked" in msg) or ("busy" in msg)

def _commit_with_retry(con: sqlite3.Connection,
                       write_fn: Callable[[], Any],
                       *,
                       max_retries: int = 5,
                       base_sleep_s: float = 0.25) -> Any:
    """
    Ejecuta write_fn() dentro de una transacción de escritura.
    Si hay lock/busy, reintenta con backoff exponencial.
    """
    for attempt in range(max_retries):
        try:
            con.execute("BEGIN IMMEDIATE;")  # toma el lock de escritura al inicio
            out = write_fn()
            con.commit()
            return out
        except sqlite3.OperationalError as e:
            con.rollback()
            if (not _is_lock_error(e)) or (attempt == max_retries - 1):
                raise
            time.sleep(base_sleep_s * (2 ** attempt))
        except Exception:
            con.rollback()
            raise

        

def check_pragmas(con: sqlite3.Connection) -> Dict[str, str]:
    return {
        'foreign_keys': con.execute("PRAGMA foreign_keys;").fetchone()[0],
        "journal_mode": con.execute("PRAGMA journal_mode;").fetchone()[0],
        "synchronous": con.execute("PRAGMA synchronous;").fetchone()[0],
    }

con = get_connection()
print(check_pragmas(con))
con.close()


# %% [markdown]
# # Table schemes + Indices + Views

# %%
DDL_PREPROCESSED = {
    "_01_reddit_posts_preprocessed": """
    CREATE TABLE IF NOT EXISTS _01_reddit_posts_preprocessed (
        post_id TEXT PRIMARY KEY,
        lang TEXT,
        lang_conf REAL,
        combined_raw TEXT,        -- title + body
        combined_ml TEXT,
        combined_llm TEXT,
        content_hash TEXT,         -- hash combined_ml normalized
        processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        proc_version TEXT,
        FOREIGN KEY (post_id) REFERENCES reddit_posts (post_id) ON DELETE CASCADE
    );
    """,

    "_01_reddit_comments_preprocessed": """
    CREATE TABLE IF NOT EXISTS _01_reddit_comments_preprocessed (
        comment_id TEXT PRIMARY KEY,
        post_id TEXT NOT NULL,
        lang TEXT,
        lang_conf REAL,
        body_raw TEXT, 
        body_ml TEXT,
        body_llm TEXT,
        content_hash TEXT,
        processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        proc_version TEXT,
        FOREIGN KEY (comment_id) REFERENCES reddit_comments (comment_id) ON DELETE CASCADE,
        FOREIGN KEY (post_id) REFERENCES reddit_posts (post_id) ON DELETE CASCADE
    );
    """,
    "_01_news_articles_preprocessed": """
        CREATE TABLE IF NOT EXISTS _01_news_articles_preprocessed (
            url TEXT PRIMARY KEY,
            lang TEXT,
            lang_conf REAL,
            combined_raw TEXT,     -- title + description + content
            combined_ml TEXT,
            combined_llm TEXT,
            content_hash TEXT,
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            proc_version TEXT,
            FOREIGN KEY (url) REFERENCES news_articles (url) ON DELETE CASCADE
        );
    """
}

DDL_INDEXES = {
    "_01_reddit_posts_preprocessed": [
        "CREATE INDEX IF NOT EXISTS idx_rpp_hash ON _01_reddit_posts_preprocessed (content_hash);",
        "CREATE INDEX IF NOT EXISTS idx_rpp_proc_at ON _01_reddit_posts_preprocessed (processed_at);",
    ],
    "_01_reddit_comments_preprocessed": [
        "CREATE INDEX IF NOT EXISTS idx_rcp_hash ON _01_reddit_comments_preprocessed (content_hash);",
        "CREATE INDEX IF NOT EXISTS idx_rcp_post ON _01_reddit_comments_preprocessed (post_id);",
        "CREATE INDEX IF NOT EXISTS idx_rcp_proc_at ON _01_reddit_comments_preprocessed (processed_at);",
    ],
    "_01_news_articles_preprocessed": [
        "CREATE INDEX IF NOT EXISTS idx_nap_hash ON _01_news_articles_preprocessed (content_hash);",
        "CREATE INDEX IF NOT EXISTS idx_nap_proc_at ON _01_news_articles_preprocessed (processed_at);",
    ],
}

DDL_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS _etl_checkpoints (
    target_table TEXT PRIMARY KEY,    -- p.ej. '_01_reddit_posts_preprocessed'
    source_table TEXT NOT NULL,       -- p.ej. 'reddit_posts'
    pk_col       TEXT NOT NULL,       -- p.ej. 'post_id'
    time_col     TEXT NOT NULL,       -- p.ej. 'created_utc' | 'published_at'
    last_pk      TEXT,                -- último PK procesado
    last_time    REAL,                -- último timestamp procesado (segundos UNIX)
    updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Vistas de conveniencia (joins listos para usar)
DDL_VIEWS = {
    "v_reddit_posts_enriched": """
    CREATE VIEW IF NOT EXISTS v_reddit_posts_enriched AS
    SELECT p.*, pp.lang, pp.lang_conf, pp.combined_ml, pp.combined_llm, pp.content_hash, pp.processed_at
    FROM reddit_posts p
    LEFT JOIN _01_reddit_posts_preprocessed pp ON pp.post_id = p.post_id;
    """,
    "v_reddit_comments_enriched": """
    CREATE VIEW IF NOT EXISTS v_reddit_comments_enriched AS
    SELECT c.*, cp.lang, cp.lang_conf, cp.body_ml, cp.body_llm, cp.content_hash, cp.processed_at
    FROM reddit_comments c
    LEFT JOIN _01_reddit_comments_preprocessed cp ON cp.comment_id = c.comment_id;
    """,
    "v_news_articles_enriched": """
    CREATE VIEW IF NOT EXISTS v_news_articles_enriched AS
    SELECT n.*, np.lang, np.lang_conf, np.combined_ml, np.combined_llm, np.content_hash, np.processed_at
    FROM news_articles n
    LEFT JOIN _01_news_articles_preprocessed np ON np.url = n.url;
    """
}

# %%

def create_processed_schema(db_path: Optional[str] = None):
    con = get_connection(db_path)
    try:
        cur = con.cursor()
        for ddl in DDL_PREPROCESSED.values():
            cur.execute(ddl)
        for tbl, idxs in DDL_INDEXES.items():
            for idx in idxs:
                cur.execute(idx)
        cur.execute(DDL_CHECKPOINTS)
        for ddl in DDL_VIEWS.values():
            cur.execute(ddl)
        con.commit()
    finally:
        con.close()

create_processed_schema()

# %%
def _ensure_str(x) -> str:
    return "" if x is None else str(x)

def compute_content_hash(text: str) -> str:
    norm = (_ensure_str(text)).strip()
    h = hashlib.blake2b(norm.encode('utf-8'), digest_size = 20)
    return h.hexdigest()
def combine_post_text(title: Optional[str], body: Optional[str]) -> str:
    parts = [p.strip() for p in [title, body] if p and p.strip()]
    return " - ".join(parts) if parts else ""

def combine_news_text(title: Optional[str], desc: Optional[str], content: Optional[str]) -> str:
    parts = [p.strip() for p in [title, desc, content] if p and p.strip()]
    return " - ".join(parts) if parts else ""

# %%
def upsert_checkpoint(con: sqlite3.Connection,
                      target_table: str,
                      source_table: str,
                      pk_col: str,
                      time_col: str,
                      last_pk: Optional[str],
                      last_time: Optional[float]
                      ) -> None:
    con.execute("""
        INSERT INTO _etl_checkpoints(target_table, source_table, pk_col, time_col, last_pk, last_time, updated_at)
        VALUES (?, ?, ?, ?, ? , ?, CURRENT_TIMESTAMP)
        ON CONFLICT(target_table) DO UPDATE SET
            last_pk = excluded.last_pk,
            last_time = excluded.last_time,
            updated_at = CURRENT_TIMESTAMP;
        """, (target_table, source_table, pk_col, time_col, last_pk, last_time))
    
def get_checkpoint(con: sqlite3.Connection, target_table: str) -> Optional[Tuple[str, str, str, str, Optional[str], Optional[float]]]:
    row = con.execute("""
                      SELECT target_table, source_table, pk_col, time_col, last_pk, last_time
                      FROM _etl_checkpoints WHERE target_table = ?;
                      """, (target_table, )).fetchone()
    return row

def _order_time_sql_posts_comments(created_col: str = "created_utc") -> str:
    # Soporta stored as TEXT o REAL/INTEGER
    return f"COALESCE(CAST({created_col} AS REAL), 0.0)"

def _order_time_sql_news(published_col: str = "published_at",
                         fetch_col: str = "fetch_date") -> str:
    # Soporta ISO strings o epoch numérico
    def _as_epoch(col: str) -> str:
        return (
            f"CASE "
            f"WHEN typeof({col}) IN ('integer','real') THEN CAST({col} AS REAL) "
            f"ELSE CAST(strftime('%s', {col}) AS REAL) "
            f"END"
        )
    return (
        "COALESCE("
        f"{_as_epoch(published_col)}, "
        f"{_as_epoch(fetch_col)}, "
        "0.0)"
    )


def fetch_missing_batch_from_view(con: sqlite3.Connection,
                                  view_name: str,
                                  select_cols: List[str],
                                  *,
                                  pk_col: str,
                                  order_time_sql: str,
                                  batch_size: int,
                                  extra_where_sql: str = "",
                                  extra_params: Tuple[Any, ...] = ()) -> List[Tuple]:
    """
    Devuelve filas aún no preprocesadas:
      - En tus vistas enriquecidas, processed_at viene de la tabla preprocesada.
      - Si no existe registro preprocesado, processed_at es NULL.
    """
    where = "processed_at IS NULL"
    if extra_where_sql:
        where = f"({where}) AND ({extra_where_sql})"

    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM {view_name}
        WHERE {where}
        ORDER BY {order_time_sql} ASC, {pk_col} ASC
        LIMIT ?;
    """
    params = (*extra_params, batch_size)
    return con.execute(sql, params).fetchall()


def _order_time_sql(source_table: str, time_col: str) -> str:
    if source_table in ("reddit_posts", "reddit_comments") and time_col == "created_utc":
        return f"COALESCE({source_table}.created_utc, 0.0)"
    if source_table == "news_articles" and time_col == "published_at":
        return (
            "COALESCE("
            "CAST(strftime('%s', news_articles.published_at) AS REAL), "
            "CAST(strftime('%s', news_articles.fetch_date)   AS REAL), "
            "0.0)"
        )
    return f"COALESCE(CAST(strftime('%s', {source_table}.{time_col}) AS REAL), 0.0)"



def fetch_unprocessed_batch(con: sqlite3.Connection,
                            source_table: str,
                            processed_table: str,
                            pk_col: str,
                            time_col: str,
                            select_cols: List[str],
                            batch_size: int,
                            checkpoint: Optional[Tuple[str, str, str, str, Optional[str], Optional[float]]] = None):
    '''
    Obtains some crude rows that arent in the preprocessed table
    Respects checkpoint: (last_time, last_pk) for sorting and continuing
    '''

    order_time = _order_time_sql(source_table, time_col)
    last_pk = None
    last_time = None
    if checkpoint:
        _, src_tbl, _, _, last_pk, last_time = checkpoint
    
    # Criteria: "not exists" + checkpoint per (time, pk) for an stable order
    # if last_time/last_pk are NULL, the condition is ignored (first time)

    where_checkpoint = f"""
    AND (
        (? IS NULL AND ? IS NULL)
        OR
        ({order_time} > ?)
        OR
        ({order_time} = ? AND {source_table}.{pk_col} > ?)
    )
    """

    sql = f"""
    SELECT {", ".join([f"{source_table}.{c}" for c in select_cols])}
    FROM {source_table}
    LEFT JOIN {processed_table} p ON p.{pk_col} = {source_table}.{pk_col}
    WHERE p.{pk_col} IS NULL
    {where_checkpoint}
    ORDER BY {order_time} ASC, {source_table}.{pk_col} ASC
    LIMIT ?;
    """
    params = (last_time, last_pk, last_time, last_time, last_pk, batch_size)
    cur = con.execute(sql, params)
    rows = cur.fetchall()
    return rows


# %% [markdown]
# # Insertion

# %%
def upsert_posts_processed(con: sqlite3.Connection, rows: List[Tuple]):
    """
    Inserta/actualiza filas en _01_reddit_posts_preprocessed.
    'rows' es una lista de tuplas con:
      (post_id, lang, lang_conf, combined_raw, combined_ml, combined_llm, content_hash, PROC_VERSION)
    """
    con.executemany("""
        INSERT INTO _01_reddit_posts_preprocessed(
            post_id, lang, lang_conf, combined_raw, combined_ml, combined_llm, content_hash, proc_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(post_id) DO UPDATE SET
            lang = excluded.lang,
            lang_conf = excluded.lang_conf,
            combined_raw = excluded.combined_raw,
            combined_ml = excluded.combined_ml,
            combined_llm = excluded.combined_llm,
            content_hash = excluded.content_hash,
            processed_at = CURRENT_TIMESTAMP,
            proc_version = excluded.proc_version;
    """, rows)

def upsert_comments_processed(con: sqlite3.Connection, rows: List[Tuple]):
    """
    Inserta/actualiza filas en _01_reddit_comments_preprocessed.
    Tuplas:
      (comment_id, post_id, lang, lang_conf, body_raw, body_ml, body_llm, content_hash, PROC_VERSION)
    """
    con.executemany("""
        INSERT INTO _01_reddit_comments_preprocessed(
            comment_id, post_id, lang, lang_conf, body_raw, body_ml, body_llm, content_hash, proc_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(comment_id) DO UPDATE SET
            post_id = excluded.post_id,
            lang = excluded.lang,
            lang_conf = excluded.lang_conf,
            body_raw = excluded.body_raw,
            body_ml = excluded.body_ml,
            body_llm = excluded.body_llm,
            content_hash = excluded.content_hash,
            processed_at = CURRENT_TIMESTAMP,
            proc_version = excluded.proc_version;
    """, rows)

def upsert_news_processed(con: sqlite3.Connection, rows: List[Tuple]):
    """
    Inserta/actualiza filas en _01_news_articles_preprocessed.
    Tuplas:
      (url, lang, lang_conf, combined_raw, combined_ml, combined_llm, content_hash, PROC_VERSION)
    """
    con.executemany("""
        INSERT INTO _01_news_articles_preprocessed(
            url, lang, lang_conf, combined_raw, combined_ml, combined_llm, content_hash, proc_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            lang = excluded.lang,
            lang_conf = excluded.lang_conf,
            combined_raw = excluded.combined_raw,
            combined_ml = excluded.combined_ml,
            combined_llm = excluded.combined_llm,
            content_hash = excluded.content_hash,
            processed_at = CURRENT_TIMESTAMP,
            proc_version = excluded.proc_version;
    """, rows)


# %%
from click import Option


def process_posts_batch(db_path: Optional[str] = None,
                        batch_size: int = 100,
                        to_lowercase_ml: bool = False) -> Tuple[int, Optional[str], Optional[float]]:
    con = get_connection(db_path)
    try:
        rows = fetch_missing_batch_from_view(
            con,
            view_name="v_reddit_posts_enriched",
            select_cols=["post_id", "title", "body", "created_utc"],
            pk_col="post_id",
            order_time_sql=_order_time_sql_posts_comments("created_utc"),
            batch_size=batch_size,
        )
        if not rows:
            return 0, None, None

        to_upsert = []
        last_pk: Optional[str] = None
        last_time: Optional[float] = None

        for post_id, title, body, created_utc in rows:
            combined_raw = combine_post_text(title, body)

            ml = text_preprocessing_ml(combined_raw, to_lowercase=to_lowercase_ml)
            llm = text_preprocessing_llm(combined_raw, to_lowercase=False)

            lang = ml.get("lang")
            try:
                lang_conf = float(ml.get("lang_conf", 0.0))
            except Exception:
                lang_conf = 0.0

            combined_ml = ml.get("text", "")
            combined_llm = llm.get("text", "")
            content_hash = compute_content_hash(combined_ml)

            to_upsert.append((
                post_id, lang, lang_conf, combined_raw, combined_ml, combined_llm, content_hash, PROC_VERSION
            ))

            last_pk = post_id
            try:
                last_time = float(created_utc) if created_utc is not None else None
            except Exception:
                last_time = None

        def _write():
            upsert_posts_processed(con, to_upsert)

        _commit_with_retry(con, _write)
        return len(to_upsert), last_pk, last_time
    finally:
        con.close()



def process_comments_batch(db_path: Optional[str] = None,
                           batch_size: int = 200,
                           to_lowercase_ml: bool = False) -> Tuple[int, Optional[str], Optional[float]]:
    con = get_connection(db_path)
    try:
        rows = fetch_missing_batch_from_view(
            con,
            view_name="v_reddit_comments_enriched",
            select_cols=["comment_id", "post_id", "body", "created_utc"],
            pk_col="comment_id",
            order_time_sql=_order_time_sql_posts_comments("created_utc"),
            batch_size=batch_size,
        )
        if not rows:
            return 0, None, None

        to_upsert = []
        last_pk: Optional[str] = None
        last_time: Optional[float] = None

        for comment_id, post_id, body, created_utc in rows:
            body_raw = _ensure_str(body)

            ml = text_preprocessing_ml(body_raw, to_lowercase=to_lowercase_ml)
            llm = text_preprocessing_llm(body_raw, to_lowercase=False)

            lang = ml.get("lang")
            try:
                lang_conf = float(ml.get("lang_conf", 0.0))
            except Exception:
                lang_conf = 0.0

            body_ml = ml.get("text", "")
            body_llm = llm.get("text", "")
            content_hash = compute_content_hash(body_ml)

            to_upsert.append((
                comment_id, post_id, lang, lang_conf, body_raw, body_ml, body_llm, content_hash, PROC_VERSION
            ))

            last_pk = comment_id
            try:
                last_time = float(created_utc) if created_utc is not None else None
            except Exception:
                last_time = None

        def _write():
            upsert_comments_processed(con, to_upsert)

        _commit_with_retry(con, _write)
        return len(to_upsert), last_pk, last_time
    finally:
        con.close()




def process_news_batch(db_path: Optional[str] = None,
                       batch_size: int = 200,
                       to_lowercase_ml: bool = False) -> Tuple[int, Optional[str], Optional[float]]:
    con = get_connection(db_path)
    try:
        rows = fetch_missing_batch_from_view(
            con,
            view_name="v_news_articles_enriched",
            select_cols=["url", "title", "description", "content", "published_at", "fetch_date"],
            pk_col="url",
            order_time_sql=_order_time_sql_news("published_at", "fetch_date"),
            batch_size=batch_size,
        )
        if not rows:
            return 0, None, None

        to_upsert = []
        last_pk: Optional[str] = None
        last_time: Optional[float] = None

        # Helper para logging (no checkpoint)
        def _epoch_from_news_row(published_at, fetch_date) -> float:
            # Si ya viene como número:
            if isinstance(published_at, (int, float)):
                return float(published_at)
            if isinstance(fetch_date, (int, float)):
                return float(fetch_date)

            # Si viene como string ISO:
            if published_at:
                v = con.execute("SELECT CAST(strftime('%s', ?) AS REAL);", (published_at,)).fetchone()[0]
                if v is not None:
                    return float(v)
            if fetch_date:
                v = con.execute("SELECT CAST(strftime('%s', ?) AS REAL);", (fetch_date,)).fetchone()[0]
                if v is not None:
                    return float(v)
            return 0.0

        for url, title, description, content, published_at, fetch_date in rows:
            combined_raw = combine_news_text(title, description, content)

            ml = text_preprocessing_ml(combined_raw, to_lowercase=to_lowercase_ml)
            llm = text_preprocessing_llm(combined_raw, to_lowercase=False)

            lang = ml.get("lang")
            try:
                lang_conf = float(ml.get("lang_conf", 0.0))
            except Exception:
                lang_conf = 0.0

            combined_ml = ml.get("text", "")
            combined_llm = llm.get("text", "")
            content_hash = compute_content_hash(combined_ml)

            to_upsert.append((
                url, lang, lang_conf, combined_raw, combined_ml, combined_llm, content_hash, PROC_VERSION
            ))

            last_pk = url
            last_time = _epoch_from_news_row(published_at, fetch_date)

        def _write():
            upsert_news_processed(con, to_upsert)

        _commit_with_retry(con, _write)
        return len(to_upsert), last_pk, last_time
    finally:
        con.close()


            
                        


# %%
def preprocess_all_tables(db_path: Optional[str] = None,
                          sample_size_per_table: int = 200,
                          batch_size: int = 50,
                          to_lowercase_ml: bool = False):
    db_path = db_path or DB_PATH_DEFAULT
    create_processed_schema(db_path)

    processed = {"posts": 0, "comments": 0, "news": 0}

    while processed["posts"] < sample_size_per_table:
        n, last_pk, _ = process_posts_batch(db_path, batch_size, to_lowercase_ml)
        if n == 0:
            break
        processed["posts"] += n
        print(f" > posts +{n} (last_pk={last_pk})")

    while processed["comments"] < sample_size_per_table:
        n, last_pk, _ = process_comments_batch(db_path, batch_size, to_lowercase_ml)
        if n == 0:
            break
        processed["comments"] += n
        print(f" > comments +{n} (last_pk={last_pk})")

    while processed["news"] < sample_size_per_table:
        n, last_pk, _ = process_news_batch(db_path, batch_size, to_lowercase_ml)
        if n == 0:
            break
        processed["news"] += n
        print(f" > news +{n} (last_pk={last_pk})")

    print("[DONE]", processed)
    return processed

        


# %%
def read_posts(con: sqlite3.Connection, limit: int = 10):
    return con.execute("""
        SELECT post_id, subreddit, title, body, created_utc
        FROM reddit_posts
        ORDER BY created_utc DESC
        LIMIT ?;
    """, (limit,)).fetchall()

def read_comments(con: sqlite3.Connection, limit: int = 10):
    return con.execute("""
        SELECT comment_id, post_id, body, created_utc
        FROM reddit_comments
        ORDER BY created_utc DESC
        LIMIT ?;
    """, (limit,)).fetchall()

def read_news(con: sqlite3.Connection, limit: int = 10):
    return con.execute("""
        SELECT url, source_name, author, title, description, published_at
        FROM news_articles
        ORDER BY published_at DESC
        LIMIT ?;
    """, (limit,)).fetchall()


