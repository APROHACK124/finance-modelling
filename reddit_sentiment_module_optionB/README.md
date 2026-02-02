# Reddit Sentiment Feature Store (tickers 5/20/60)

## What it does
- Reads **already-collected** Reddit threads + LLM sentiment outputs from a sqlite DB.
- Produces a **daily feature store** (1 row per `date,ticker`) with sentiment features for horizons **5/20/60** (configurable).
- Produces a **daily market/macro index** derived from Reddit scopes (macro/rates/policy/sector) (configurable).
- (Optional) Produces **sector indices** if you provide a `ticker -> sector` mapping table.

## Avoiding leakage
- Each thread is assigned to a `feature_date` using a **timezone + cutoff time** (`asof_timezone`, `asof_cutoff_time`).
- A thread posted **after the cutoff** is assigned to **next** `feature_date`.
- All time decay is computed relative to **UTC cutoffs** (DST-safe).

## Inputs (sqlite tables)
Required:
- `_02_sentiment_predictions` (LLM outputs), must include: `doc_id`, `tickers_json`, `sentiment`, `confidence_10`, `relevance_10`, `scope`
- `reddit_posts` (metadata), must include: `post_id`, `created_utc`, `score`, `subreddit`
- `_ref_ticker_universe` with column `ticker`

Optional:
- `reddit_comments` with columns that can link to a post (`post_id`/`link_id`/`submission_id`) + `score`
- `_ref_ticker_aliases` (alias -> ticker) to reduce false positives
- sector mapping table (set `table_ticker_sector` in config)

## Outputs (sqlite tables)
- `_03_sentiment_feature_store`: daily features per ticker
- `_03_sentiment_market_index`: daily market index per horizon
- `_03_sentiment_sector_sent` (optional): daily sector indices

## Run
```bash
python -m sentiment.cli --db ../data/stock_data.db
```

Optional:
```bash
python -m sentiment.cli --db db.sqlite --config config.json --start-date 2025-01-01 --end-date 2026-01-31
```

## Integration with your agent
- Use `sent_net_20`, `avg_rel_20`, `sent_volume_20`, `disagree_20` as gating/position sizing features.
- Use `sent_rank_h`/`sent_z_h` for cross-sectional comparability.
- Use `top_positive_threads`/`top_negative_threads` for explainability (thread ids to fetch permalink/url from `reddit_posts`).
