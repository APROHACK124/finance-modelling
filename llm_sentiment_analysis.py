# %% [markdown]
# # 02_openai_sentiment
# 
# **Objetivo:** clasificar sentimiento (positivo/neutral/negativo) con **OpenAI Responses API** usando **Structured Outputs (JSON Schema)** para maximizar fiabilidad.
# 
# **¿Por qué Structured Outputs y no “texto libre”?**
# - El modelo **debe** producir un JSON que **cumple** un **JSON Schema** → reduce parsing frágil y errores de formato.  
# - Menos *prompt brittleness*: definimos contrato de salida (tipos, enum, campos obligatorios).  
# - Integración directa con validadores locales y BD.
# 
# **Flujo (diagrama ASCII)**
# 
# 

# %%
from http.client import REQUEST_TIMEOUT
import os
import json
import time
import math
import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from jsonschema import Draft202012Validator, validate, ValidationError
from tabulate import tabulate

# OpenAI SDK
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError('library openai needed')

# LiteLLM
try:
    import litellm
    LITELLM_AVAILABLE = True
except Exception:
    LITELLM_AVAILABLE = False

OPENAL_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAL_API_KEY:
    print('OPENAI_API_KEY missing')

OPENAI_MODEL = 'gpt-5-nano'
TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 128
REQUEST_TIMEOUT_ = 30 # seconds
MAX_CONCURRENCY = 4
BATCH_SIZE = 20

# %%
from python_scripts.LLM_analysis.preprocess_store_database import resolve_db_path, get_connection

# %%
# JSON Schema para Structured Outputs
SENTIMENT_JSON_SCHEMA = {
    "name": "sentiment_schema",
    "schema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"]
            },
            "confidence": {
                "type": "number",
                "minimum": 1.0,
                "maximum": 10.0
            },
            "relevance" :{
                "type": "number",
                "minimum": 1.0,
                "maximum": 10.0
            },
            "explanation": {
                "type": "string",
                "minLength": 0,
                "maxLength": 512
            }
        },
        "required": ["sentiment", "confidence"]
    },
    # strict=True hace que el modelo **tenga** que cumplir el schema
    "strict": True
}

# Validador local (jsonschema)
SENTIMENT_VALIDATOR = Draft202012Validator(SENTIMENT_JSON_SCHEMA["schema"])


# %%
COST_TABLE_USD_PER_1K = {
    # Rellena con precios actuales si los conoces (input/output)
    # "gpt-4o-mini": {"input": 0.0, "output": 0.0}
    'gpt-5-nano': {'input':0.05 / 1000,	'output': 0.40/1000}
}

def get_cost_per_1k(model: str) -> Tuple[float, float]:
    """
    Devuelve (precio_input, precio_output) USD por 1k tokens.
    Si no está configurado, asume 0.0 para evitar sorpresas.
    """
    entry = COST_TABLE_USD_PER_1K.get(model, {})
    return (float(entry.get("input", 0.0)), float(entry.get("output", 0.0)))

# %%
def content_hash(text: str) -> str:
    s = (text or "").strip()
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()

def validate_or_raise(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida contra el JSON Schema. Lanza ValidationError si no cumple.
    """
    SENTIMENT_VALIDATOR.validate(payload)
    return payload

def backoff_sleep(base: float, attempt: int, max_sleep: float = 60.0) -> None:
    """
    Exponential backoff with jitter.
    """
    sleep_s = min(max_sleep, base * (2 ** attempt)) * (0.5 + 0.5 * os.urandom(1)[0] / 255)
    time.sleep(sleep_s)

# %% [markdown]
# # Actual prediction (LiteLLM)

# %%
from typing import NamedTuple

class OpenAIUsage(NamedTuple):
    input_tokens: int
    output_tokens: int
    total_tokens: int

def _extract_usage(resp) -> OpenAIUsage:
    """
    Extrae usage de la respuesta Responses API.
    """
    u = getattr(resp, "usage", None) or {}
    return OpenAIUsage(
        input_tokens=int(u.get("input_tokens", 0)),
        output_tokens=int(u.get("output_tokens", 0)),
        total_tokens=int(u.get("total_tokens", 0)),
    )

def _parse_json_from_response(resp) -> Dict[str, Any]:
    """
    En Responses API puedes usar resp.output_text para obtener el texto agregado.
    """
    text = getattr(resp, "output_text", None)
    if not text:
        # Fallback: navegar el árbol 'output'
        out = getattr(resp, "output", None) or []
        # Buscar el primer bloque de texto
        for item in out:
            # item.content es una lista de partes; buscar .text
            content = getattr(item, "content", None) or []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("output_text", "input_text", "text"):
                    maybe = part.get("text")
                    if maybe:
                        text = maybe
                        break
            if text:
                break
    if not text:
        raise ValueError("No se pudo extraer texto del objeto de respuesta.")
    return json.loads(text)

# %%
def chunked(iterable: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# %%
# === LiteLLM helpers (corrige streaming y parsing) ===
import json
from typing import Tuple, Dict, Any, List, Optional, Mapping

try:
    import litellm
    LITELLM_AVAILABLE = True
except Exception:
    LITELLM_AVAILABLE = False

from jsonschema import ValidationError

def _parse_litellm_response(resp: Any) -> str:
    """
    Extrae el contenido textual de la respuesta de LiteLLM.
    - Soporta dict-like y objetos con atributos.
    - Evita depender de resp.choices como atributo directo (mejor para linters).
    """
    # 1) Respuesta dict estilo OpenAI
    if isinstance(resp, Mapping):
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 2) Respuesta objeto estilo OpenAI / LiteLLM (sin asumir tipo)
    choices = getattr(resp, "choices", None)
    if choices is not None:
        try:
            choice0 = choices[0]
        except Exception:
            choice0 = None

        if choice0 is not None:
            # Caso: choice0.message.content
            msg = getattr(choice0, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    return content

            # Caso: choice0 dict-like
            if isinstance(choice0, Mapping):
                msg2 = choice0.get("message", {})
                if isinstance(msg2, Mapping):
                    content2 = msg2.get("content")
                    if isinstance(content2, str) and content2:
                        return content2

    # 3) Algunos backends devuelven directamente `.content` o `.output_text`
    direct = getattr(resp, "content", None)
    if isinstance(direct, str) and direct:
        return direct

    out_text = getattr(resp, "output_text", None)
    if isinstance(out_text, str) and out_text:
        return out_text

    # 4) Si accidentalmente vino en streaming, pide non-stream
    raise ValueError("Respuesta LiteLLM no parseable; asegúrate de usar stream=False.")

def _extract_litellm_usage(resp: Any) -> Dict[str, int]:
    """
    Extrae usage de una respuesta LiteLLM (si está disponible).
    Retorna dict con prompt/completion/total tokens.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # dict
    if isinstance(resp, dict):
        u = resp.get("usage") or {}
        usage["prompt_tokens"] = int(u.get("prompt_tokens", u.get("input_tokens", 0)))
        usage["completion_tokens"] = int(u.get("completion_tokens", u.get("output_tokens", 0)))
        usage["total_tokens"] = int(u.get("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"]))
        return usage
    # objeto
    u = getattr(resp, "usage", None)
    if u:
        usage["prompt_tokens"] = int(getattr(u, "prompt_tokens", getattr(u, "input_tokens", 0)))
        usage["completion_tokens"] = int(getattr(u, "completion_tokens", getattr(u, "output_tokens", 0)))
        usage["total_tokens"] = int(getattr(u, "total_tokens", usage["prompt_tokens"] + usage["completion_tokens"]))
    return usage

def _classify_litellm_one(t: str,
                          model_name: str,
                          temperature: float,
                          max_tokens: int,
                          schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Clasifica un texto usando LiteLLM (no streaming) con Structured Outputs.
    Devuelve (payload_json_validado, usage_dict).
    """
    if not LITELLM_AVAILABLE:
        raise RuntimeError("LiteLLM no está instalado. pip install litellm")

    system_instr = (
        "You are a financial sentiment classifier. "
        "Return a JSON object that matches the provided schema exactly."
    )

    # 🔴 Forzamos stream=False para evitar CustomStreamWrapper

    resp = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instr},
            {"role": "user", "content": f"Texto:\n{t[:8000]}\n\nReturn JSON : sentiment (positive/neutral/negative), confidence [1, 10] (How confident are you about the sentiment selected), relevance [1, 10] (How relevant is this information in predicting related stock prices: 10 for instant buy/sell, 5 for barely informative), explanation (short)."}
        ],
        # max_tokens=max_tokens,           # Chat Completions → max_tokens (no max_output_tokens)
        response_format={                 # Structured Outputs (si el backend lo soporta)
            "type": "json_schema",
            "json_schema": {
                "name": schema["name"],
                "schema": schema["schema"],
            },
        }
    )

    content = _parse_litellm_response(resp)
    try:
        payload = json.loads(content)
        validate_or_raise(payload)       # jsonschema local
    except (json.JSONDecodeError, ValidationError):
        payload = {"sentiment": "neutral", "confidence": 0.0, "explanation": "fallback_invalid_json"}

    usage = _extract_litellm_usage(resp)
    return payload, usage

def classify_texts_litellm(texts: List[str],
                           model_name: str = "gpt-4o-mini",
                           max_concurrency: int = 4,
                           batch_size: int = 20,
                           temperature: float = 1.0,
                           max_tokens: int = 128) -> List[Dict[str, Any]]:
    """
    Concurrencia + batches con LiteLLM. Devuelve lista de dicts con sentimiento + usage.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []
    def _chunked(items, n):
        for i in range(0, len(items), n):
            yield items[i:i+n]

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for batch in _chunked(texts, batch_size):
            futs = {
                pool.submit(_classify_litellm_one, t, model_name, temperature, max_tokens, SENTIMENT_JSON_SCHEMA): t
                for t in batch
            }
            for fut in as_completed(futs):
                t = futs[fut]
                payload, usage = fut.result()
                results.append({
                    "text_hash": content_hash(t),
                    "sentiment": payload.get("sentiment", "neutral"),
                    "confidence": float(payload.get("confidence", 0.0)),
                    "relevance": float(payload.get("relevance", 0)),
                    "explanation": payload.get("explanation", ""),
                    "usage_input_tokens": int(usage.get("prompt_tokens", 0)),
                    "usage_output_tokens": int(usage.get("completion_tokens", 0)),
                    "usage_total_tokens": int(usage.get("total_tokens", 0)),
                    "model": model_name,
                })
    return results


# %%


# %%
from difflib import get_close_matches


PRED_TABLE = "_02_sentiment_predictions"

SCHEMA_VERSION = "sentiment_v1"
SCHEMA_VERSION_SENT = "sentiment_v1"
SCHEMA_VERSION_THREAD = "thread_v1"

DDL_PREDICTIONS = f"""
CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,            -- 'reddit_post' | 'reddit_comment' | 'news' | ...
    doc_id TEXT NOT NULL,            -- PK de la tabla origen
    text_hash TEXT NOT NULL,
    sentiment TEXT NOT NULL CHECK (sentiment IN ('positive','neutral','negative')),
    confidence_10 INTEGER,
    relevance_10 INTEGER,
    explanation TEXT,
    model TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT '{SCHEMA_VERSION}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    usage_input_tokens INTEGER,
    usage_output_tokens INTEGER,
    usage_total_tokens INTEGER,
    cost_usd REAL,
    UNIQUE (source, doc_id, model, schema_version)
);
CREATE INDEX IF NOT EXISTS idx_sent_texthash ON {PRED_TABLE}(text_hash);
CREATE INDEX IF NOT EXISTS idx_sent_source_doc ON {PRED_TABLE}(source, doc_id);
"""

def init_predictions_schema(db_path: Optional[str] = None):
    
    con = get_connection(db_path)
    try:
        con.executescript(DDL_PREDICTIONS)
        # intentamos agregar columnas si vienen de versión anterior
        try: con.execute(f"ALTER TABLE {PRED_TABLE} ADD COLUMN schema_version TEXT DEFAULT '{SCHEMA_VERSION}'")
        except sqlite3.OperationalError: pass
        try: con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS uq_sent_source_doc_model_ver ON {PRED_TABLE}(source, doc_id, model, schema_version)")
        except sqlite3.OperationalError: pass
        con.commit()
    finally:
        con.close()

def init_predictions_schema_extended(db_path: str | None = None):
    con = get_connection(db_path)
    try:
        con.executescript(DDL_PREDICTIONS)
        # Migración suave (si vienes de versión anterior)
        for col_def in [
            "ADD COLUMN confidence_10 INTEGER",
            "ADD COLUMN relevance_10 INTEGER",
        ]:
            try:
                con.execute(f"ALTER TABLE {PRED_TABLE} {col_def}")
            except sqlite3.OperationalError:
                pass
        try:
            con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS uq_pred ON {PRED_TABLE}(source, doc_id, model, schema_version)")
        except sqlite3.OperationalError:
            pass
        con.commit()
    finally:
        con.close()

def write_predictions_to_sqlite(preds: List[Dict[str, Any]],
                                source: str,
                                id_list: List[str],
                                db_path: Optional[str] = None,
                                schema_version: str = SCHEMA_VERSION):
    if len(id_list) != len(preds):
        raise ValueError("id_list debe tener misma longitud que preds")

    price_in, price_out = get_cost_per_1k(OPENAI_MODEL)
    rows = []
    for i, p in enumerate(preds):
        cost = (p["usage_input_tokens"]/1000.0)*price_in + (p["usage_output_tokens"]/1000.0)*price_out
        rows.append((
            source, id_list[i], p["text_hash"], p["sentiment"], int(p.get("confidence", 0)), int(p.get("relevance", 0)),
            p.get("explanation",""), p.get("model", OPENAI_MODEL), schema_version,
            int(p.get("usage_input_tokens",0)), int(p.get("usage_output_tokens",0)), int(p.get("usage_total_tokens",0)),
            float(cost)
        ))

    con = get_connection(db_path)
    try:
        con.executemany(f"""
        INSERT INTO {PRED_TABLE} (
            source, doc_id, text_hash, sentiment, confidence_10, relevance_10, explanation,
            model, schema_version,
            usage_input_tokens, usage_output_tokens, usage_total_tokens, cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, doc_id, model, schema_version) DO UPDATE SET
            text_hash=excluded.text_hash,
            sentiment=excluded.sentiment,
            confidence_10=excluded.confidence_10,
            relevance_10=excluded.relevance_10,
            explanation=excluded.explanation,
            usage_input_tokens=excluded.usage_input_tokens,
            usage_output_tokens=excluded.usage_output_tokens,
            usage_total_tokens=excluded.usage_total_tokens,
            cost_usd=excluded.cost_usd,
            created_at=CURRENT_TIMESTAMP;
        """, rows)
        con.commit()
    finally:
        con.close()

    

# %%
# ChatGPT generated this

'''

sample_texts = [
    # Titulares
    "NVDA se dispara 10% tras resultados; analistas suben precio objetivo.",
    "El mercado cae por temores de recesión, bancos lideran las pérdidas.",
    "Resultados de TSLA decepcionan: margen en mínimos y guía recortada.",
    "La Fed mantendrá tipos; señales mixtas sobre inflación subyacente.",
    "BTC supera los 80k USD; rotación hacia cripto en riesgo-on.",
    # Comentarios / tweets
    "jajaja esto va to the moon 🚀🚀 compré más, diamond hands!",
    "vendí en el mínimo... qué desastre, nunca más confío en ese CEO.",
    "meh, nada cambia con este reporte, neutral AF.",
    "qué robo esas comisiones, pésimo broker 😤",
    "buen reporte, pero ya estaba descontado, poco recorrido."
]

init_predictions_schema(str(resolve_db_path))

preds_llm = classify_texts_litellm(sample_texts, model_name=os.getenv("LITELLM_MODEL", "gpt-5-nano"),
                                       max_concurrency=MAX_CONCURRENCY, batch_size=5)
df2 = pd.DataFrame(preds_llm)
# Opción 1 (bonita)
try:
    print(df2.to_markdown(index=False))
except Exception:
    # Opción 2 si no tienes tabulate/markdown
    print(df2.to_string(index=False))

'''


# %% [markdown]
# ## Thread builder

# %% [markdown]
# # Daily batches
# 
# 

# %%
def order_time_expr(source_table: str, time_col: str, alias: str | None = None) -> str:
    tbl = alias or source_table
    if source_table in ("reddit_posts", "reddit_comments") and time_col == "created_utc":
        return f"COALESCE({tbl}.{time_col}, 0.0)"
    if source_table == "news_articles" and time_col == "published_at":
        # published_at puede ser NULL; fallback a fetch_date
        return (
            f"COALESCE("
            f"CAST(strftime('%s', {tbl}.published_at) AS REAL), "
            f"CAST(strftime('%s', {tbl}.fetch_date)   AS REAL), "
            f"0.0)"
        )

    return f"COALESCE(CAST(strftime('%s', {tbl}.{time_col}) AS REAL), 0.0)"


# %%
def fetch_unscored_posts(con: sqlite3.Connection, limit: int, model: str, schema_version: str):
    texpr = order_time_expr("reddit_posts", "created_utc", alias="p")
    sql = f"""
    SELECT p.post_id, pp.combined_llm AS text_llm, {texpr} AS t
    FROM reddit_posts p
    JOIN _01_reddit_posts_preprocessed pp ON pp.post_id = p.post_id
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source='reddit_post' AND sp.doc_id=p.post_id
     AND sp.model=? AND sp.schema_version=?
    WHERE sp.doc_id IS NULL
    ORDER BY t ASC, p.post_id ASC
    LIMIT ?;
    """
    return con.execute(sql, (model, schema_version, limit)).fetchall()

def fetch_unscored_comments(con: sqlite3.Connection, limit: int, model: str, schema_version: str):
    texpr = order_time_expr("reddit_comments", "created_utc", alias="c")
    sql = f"""
    SELECT c.comment_id, cp.body_llm AS text_llm, {texpr} AS t
    FROM reddit_comments c
    JOIN _01_reddit_comments_preprocessed cp ON cp.comment_id = c.comment_id
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source='reddit_comment' AND sp.doc_id=c.comment_id
     AND sp.model=? AND sp.schema_version=?
    WHERE sp.doc_id IS NULL
    ORDER BY t ASC, c.comment_id ASC
    LIMIT ?;
    """
    return con.execute(sql, (model, schema_version, limit)).fetchall()

def fetch_unscored_news(con: sqlite3.Connection, limit: int, model: str, schema_version: str):
    texpr = order_time_expr("news_articles", "published_at", alias = 'n')
    sql = f"""
    SELECT n.url, np.combined_llm AS text_llm, {texpr} AS t
    FROM news_articles n
    JOIN _01_news_articles_preprocessed np ON np.url = n.url
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source='news' AND sp.doc_id=n.url
     AND sp.model=? AND sp.schema_version=?
    WHERE sp.doc_id IS NULL
    ORDER BY t ASC, n.url ASC
    LIMIT ?;
    """
    return con.execute(sql, (model, schema_version, limit)).fetchall()

def fetch_unscored_threads(con: sqlite3.Connection, limit: int, model: str, schema_version: str):
    """
    Devuelve posts que aún no tienen predicción de 'reddit_thread' para (model, schema_version).
    """
    texpr = order_time_expr("reddit_posts", "created_utc", alias="p")
    sql = f"""
    SELECT p.post_id, {texpr} AS t
    FROM reddit_posts p
    LEFT JOIN {PRED_TABLE} sp
      ON sp.source='reddit_thread' AND sp.doc_id=p.post_id
     AND sp.model=? AND sp.schema_version=?
    WHERE sp.doc_id IS NULL
    ORDER BY t ASC, p.post_id ASC
    LIMIT ?;
    """
    return con.execute(sql, (model, schema_version, limit)).fetchall()



# %%
def process_sentiment_posts_batch_litellm(db_path: Optional[str] = None,
                                          batch_size: int = 100,
                                          model_name: str = 'gpt-4-nano',
                                          max_concurrency: int = 4,
                                          schema_version: str = SCHEMA_VERSION) -> int:
    con = get_connection()
    try:
        rows = fetch_unscored_posts(con, batch_size, model_name, schema_version)

    finally:
        con.close()
    
    if not rows:
        return 0
    
    ids = [r[0] for r in rows]
    texts = [(r[1] or "").strip() for r in rows]
    preds = classify_texts_litellm(texts, model_name=model_name, 
                                   max_concurrency=max_concurrency, batch_size=batch_size)
    write_predictions_to_sqlite(preds, source='reddit_post', id_list=ids, db_path=db_path, schema_version=schema_version)
    return len(preds)

    
def process_sentiment_comments_batch_litellm(db_path: Optional[str] = None,
                                             batch_size: int = 200,
                                             model_name: str = "gpt-5-nano",
                                             max_concurrency: int = 4,
                                             schema_version: str = SCHEMA_VERSION) -> int:
    con = get_connection(db_path)
    try:
        rows = fetch_unscored_comments(con, batch_size, model_name, schema_version)
    finally:
        con.close()
    if not rows:
        return 0

    ids = [r[0] for r in rows]
    texts = [(r[1] or "").strip() for r in rows]
    preds = classify_texts_litellm(texts, model_name=model_name,
                                   max_concurrency=max_concurrency, batch_size=batch_size)
    write_predictions_to_sqlite(preds, source="reddit_comment", id_list=ids, db_path=db_path, schema_version=schema_version)
    return len(preds)

def process_sentiment_news_batch_litellm(db_path: Optional[str] = None,
                                         batch_size: int = 200,
                                         model_name: str = "gpt-5-nano",
                                         max_concurrency: int = 4,
                                         schema_version: str = SCHEMA_VERSION) -> int:
    con = get_connection(db_path)
    try:
        rows = fetch_unscored_news(con, batch_size, model_name, schema_version)
    finally:
        con.close()
    if not rows:
        return 0

    ids = [r[0] for r in rows]
    texts = [(r[1] or "").strip() for r in rows]
    preds = classify_texts_litellm(texts, model_name=model_name,
                                   max_concurrency=max_concurrency, batch_size=batch_size)
    write_predictions_to_sqlite(preds, source="news", id_list=ids, db_path=db_path, schema_version=schema_version)
    return len(preds)


# %% [markdown]
# ## Posts + comments
# 

# %%
from xml.etree.ElementInclude import include


def build_thread_text(con: sqlite3.Connection,
                      post_id: str,
                      top_k_comments: int = 5,
                      max_comment_chars: int = 600) -> tuple[str, list[str], dict]:
    '''
    Returns:
        - thread text (for the prompt)
        - included_comment_ids (list of ids)
        - meta (score_post, num_comments, created_utc)

    '''
    row = con.execute("""
                      SELECT p.post_id, p.subreddit, p.title, p.body, p.score, p.num_comments, p.created_utc,
                      pp.combined_llm
                      FROM reddit_posts p
                      JOIN _01_reddit_posts_preprocessed pp ON pp.post_id = p.post_id
                      WHERE p.post_id = ?;
                      """, (post_id,)).fetchone()
    
    if not row:
        raise ValueError(f'post_id not found: {post_id}')
    
    _, subreddit, title, body, score_post, num_comments, created_utc, post_llm = row
    post_llm = (post_llm or "").strip()

    # Top-K comments by score DESC and then time ASC
    comments = con.execute("""
                            SELECT c.comment_id, c.score, c.created_utc, cp.body_llm 
                            FROM reddit_comments c
                            JOIN _01_reddit_comments_preprocessed cp ON cp.comment_id = c.comment_id
                            WHERE c.post_id = ?
                            ORDER BY c.score DESC, c.created_utc ASC
                            LIMIT ?;
                           """, (post_id, top_k_comments)).fetchall()
    
    included_ids = []
    comment_lines = []
    for i, (cid, cscore, ctime, body_llm) in enumerate(comments, start=1):
        included_ids.append(cid)
        text = (body_llm or "").strip()
        if max_comment_chars and len(text) > max_comment_chars:
            text = text[:max_comment_chars] + "..."
        comment_lines.append(f"{i}. [score={cscore}] {text}")

    header = f"SUBREDDIT: {subreddit}\n"
    post_block = f"POST:\n{post_llm}\n"
    if comment_lines:
        comments_block = "TOP_COMMENTS:\n" + "\n".join(comment_lines) + "\n"
    else:
        comments_block = "TOP_COMMENTS:\n(none)\n"

    thread_text = header + post_block + comments_block

    meta = {"score_post": score_post, "num_comments": num_comments, "created_utc": float(created_utc or 0.0)}
    return thread_text, included_ids, meta

# %%
THREAD_JSON_SCHEMA = {
    "name": "thread_sentiment_schema",
    "schema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentiment":       {"type": "string", "enum": ["positive", "neutral", "negative"]},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
            "relevance":  {"type": "integer", "minimum": 1, "maximum": 10},
            "explanation": {"type": "string", "minLength": 0, "maxLength": 384}
        },
        "required": ["sentiment", "confidence", "relevance"]
    },
}


# %%
def _classify_litellm_one_schema(t: str,
                                 model_name: str,
                                 temperature: float,
                                 max_tokens: int,
                                 schema: dict,
                                 system_instr: str) -> tuple[dict, dict]:
    # More generic version

    resp = litellm.completion(
        model = model_name,
        messages=[
            {"role": "system", "content": system_instr},
            {"role": "user", "content": t[:16000]},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema["name"], "schema": schema["schema"]},
        }
    )
    content = _parse_litellm_response(resp)
    try:
        payload = json.loads(content)
        validate_or_raise(payload)
    except Exception:
        payload = {"sentiment": "neutral", "confidence": 1, "relevance": 1, "explanation": "fallback_invalid_json"}
    usage = _extract_litellm_usage(resp)
    return payload, usage

def classify_threads_litellm(thread_texts: list[str],
                             model_name: str = 'gpt-5-nano',
                             max_concurrency: int = 4,
                             batch_size: int = 20,
                             temperature: float = 1.0,
                             max_tokens: int = 128) -> list[dict]:
    # Classifies list of thread texts
    # Returns dicts with label/confidence/relevance + usage

    system_instr = (
        "You are a financial sentiment rater. "
        "Given a Reddit post and its top comments, return STRICT JSON matching the schema: "
        "sentiment in {positive, neutral, negative}; "
        "confidence integer 1..10 (how confident about the sentiment); "
        "relevance integer 1..10 (10 = highly actionable for trading decisions, 5 = mildly informative). "
        "Keep explanation short."
    )

    results: list[dict] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _chunked(items, n):
        for i in range(0, len(items), n):
            yield items[i:i + n]

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for batch in _chunked(thread_texts, batch_size):
            futs = {
                pool.submit(_classify_litellm_one_schema, t, model_name, temperature, max_tokens, THREAD_JSON_SCHEMA, system_instr): t for t in batch


            }
            for fut in as_completed(futs):
                t = futs[fut]
                payload, usage = fut.result()
                results.append({
                    "text_hash": content_hash(t),
                    "sentiment": payload.get("sentiment", "neutral"),
                    "confidence": int(payload.get("confidence", 1)),
                    "relevance": int(payload.get("relevance", 1)),
                    "explanation": payload.get("explanation", ""),
                    "usage_input_tokens": int(usage.get("prompt_tokens", 0)),
                    "usage_output_tokens": int(usage.get("completion_tokens", 0)),
                    "model": model_name,
                })                
    return results

def process_thread_sentiment_batch_litellm(db_path: str | None = None,
                                           batch_size: int = 100,
                                           top_k_comments: int = 5,
                                           model_name: str = "gpt-4o-mini",
                                           max_concurrency: int = 6,
                                           schema_version: str = SCHEMA_VERSION_THREAD) -> int:
    """
    Arma hilos (post + top-K comentarios) y los clasifica con LiteLLM.
    Reanuda automáticamente gracias al NOT EXISTS en fetch_unscored_threads.
    """
    con = get_connection(db_path)
    try:
        rows = fetch_unscored_threads(con, batch_size, model_name, schema_version)
    finally:
        con.close()
    if not rows:
        return 0

    post_ids = [r[0] for r in rows]

    # Construye los textos de hilo
    thread_texts: list[str] = []
    con = get_connection(db_path)
    try:
        for pid in post_ids:
            ttext, comment_ids, meta = build_thread_text(con, pid, top_k_comments=top_k_comments)
            # Puedes incluir el score del post en el encabezado (ya lo hace build_thread_text)
            thread_texts.append(ttext)
    finally:
        con.close()

    # Clasifica
    preds = classify_threads_litellm(thread_texts, model_name=model_name,
                                     max_concurrency=max_concurrency, batch_size=min(batch_size, 20))

    # Escribe (source='reddit_thread')
    write_predictions_to_sqlite(preds, source="reddit_thread", id_list=post_ids,
                                db_path=db_path, schema_version=schema_version)
    return len(preds)



# %%
def run_daily_sentiment_litellm(db_path: Optional[str] = None,
                                per_source_cap: int = 2000,
                                batch_size_posts: int = 200,
                                batch_size_comments: int = 400,
                                batch_size_news: int = 400,
                                batch_size_threads: int = 50,
                                top_k_comments: int = 5,
                                model_name: str = "gpt-4o-mini",
                                max_concurrency: int = 8,
                                schema_version: str = SCHEMA_VERSION, threads_only: bool = True):
    """
    Procesa en bucles de lotes hasta 'per_source_cap' por fuente.
    Reanuda automáticamente porque usa NOT EXISTS sobre la tabla de predicciones.
    threads_only: True si no se quiere una fila para cada comentario, sino mas bien un analisis del thread completo nada mas
    """
    init_predictions_schema(db_path)

    processed = {"posts": 0, "comments": 0, "news": 0, "threads": 0}

    while processed["posts"] < per_source_cap and not threads_only:
        n = process_sentiment_posts_batch_litellm(db_path, batch_size_posts, model_name, max_concurrency, schema_version)
        print(f'Processed {n} posts')
        if n == 0: break
        processed["posts"] += n
        print(f"[posts] +{n}  tot={processed['posts']}")

    while processed["comments"] < per_source_cap and not threads_only:
        n = process_sentiment_comments_batch_litellm(db_path, batch_size_comments, model_name, max_concurrency, schema_version)
        print(f'Processed {n} comments')
        if n == 0: break
        processed["comments"] += n
        print(f"[comments] +{n}  tot={processed['comments']}")

    while processed["news"] < per_source_cap:
        n = process_sentiment_news_batch_litellm(db_path, batch_size_news, model_name, max_concurrency, schema_version)
        print(f'Processed {n} news')
        if n == 0: break
        processed["news"] += n
        print(f"[news] +{n}  tot={processed['news']}")

    while processed["threads"] < per_source_cap:
        n = process_thread_sentiment_batch_litellm(db_path, batch_size_threads, top_k_comments,
                                                   model_name, max_concurrency, SCHEMA_VERSION_THREAD)
        if n == 0: break
        processed["threads"] += n
        print(f"[threads] +{n}  tot={processed['threads']}")

    print("[DONE sentiment]", processed)
    return processed

# %% [markdown]
# # Vistas

# %%


# %% [markdown]
# # Cost estimation (unfinished)

# %%
def estimate_daily_cost_from_sample(preds: List[Dict[str, Any]], expected_items_per_day: int, model: str = OPENAI_MODEL) -> Dict[str, Any]:
    if not preds:
        return {"avg_input_tokens":0, "avg_output_tokens":0, "avg_total_tokens":0, "daily_cost_usd":0.0}
    avg_in = sum(p["usage_input_tokens"] for p in preds) / len(preds)
    avg_out = sum(p["usage_output_tokens"] for p in preds) / len(preds)
    price_in, price_out = get_cost_per_1k(model)
    daily_cost = (avg_in * expected_items_per_day / 1000.0) * price_in + (avg_out * expected_items_per_day / 1000.0) * price_out
    return {
        "avg_input_tokens": round(avg_in, 2),
        "avg_output_tokens": round(avg_out, 2),
        "avg_total_tokens": round(avg_in + avg_out, 2),
        "daily_cost_usd": round(daily_cost, 4),
        "assumptions": {
            "items_per_day": expected_items_per_day,
            "model": model,
            "price_in_per_1k": price_in,
            "price_out_per_1k": price_out
        }
    }


if __name__ == '__main__':
    run_daily_sentiment_litellm(per_source_cap= 40, batch_size_posts=10, batch_size_comments=10, batch_size_news=20, batch_size_threads=4, model_name="gpt-5-nano", max_concurrency=6)