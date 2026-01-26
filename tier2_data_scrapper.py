import os
import requests
import sqlite3
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Any, Iterable

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = "https://financialmodelingprep.com"

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DB_FOLDER = os.path.join(SCRIPT_DIR, "data")
DB_FILE = os.path.join(DB_FOLDER, "stock_data.db")

TARGET_STOCKS = [
    # ETFs (20)
    # "SPY","QQQ","IWM","DIA","VTI",
    # "XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY",
    # "TLT","IEF","LQD","GLD",

    # Stocks (60)
    "AAPL","MSFT","AMZN","GOOGL","META","NFLX","ADBE","CSCO",
    "NVDA","AMD","TSM",
    "JPM","BAC","WFC","GS","MS","V","MA","BLK",
    "UNH","JNJ","MRK","ABBV","TMO","AMGN",
    "TSLA","HD","LOW","MCD","NKE","DIS","SBUX",
    "PG","KO","PEP","WMT","COST",
    "CAT","DE","HON","GE","RTX","LMT",
    "XOM","CVX","COP",
    "LIN","FCX",
    "NEE","AMT",

    # CRIPTO
    "BTC",
]

# Para fundamentals FMP, sacamos BTC (ahorra calls + evita errores)
FMP_SYMBOLS = [s for s in TARGET_STOCKS if s != "BTC"]

PERIODS_TO_FETCH = 5

# Reducido a endpoints “worth it”
ENDPOINTS_TO_FETCH = [
    {"path": "/stable/ratios", "is_historical": True},
    {"path": "/stable/key-metrics", "is_historical": True},
    {"path": "/stable/key-metrics-ttm", "is_historical": False},
]

# “Whitelist” de keys útiles (si alguna no existe en tu JSON real, simplemente se ignora)
FEATURE_KEYS: dict[str, set[str]] = {
    "profile": {
        "price", "beta", "volAvg", "mktCap", "lastDiv",
        # a veces existen flags útiles (numéricos/booleanos)
        "isEtf", "isFund", "isActivelyTrading",
    },
    "ratios": {
        # Liquidez / leverage
        "currentRatio", "quickRatio", "cashRatio",
        "debtRatio", "debtEquityRatio", "interestCoverage",

        # Márgenes / returns
        "grossProfitMargin", "operatingProfitMargin", "netProfitMargin",
        "returnOnAssets", "returnOnEquity",

        # Valuación / yield
        "priceEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
        "priceToFreeCashFlowsRatio", "dividendYield",
        "enterpriseValueMultiple", "priceFairValue",
    },
    "key_metrics": {
        # Valoración / tamaño
        "marketCap", "enterpriseValue",
        "peRatio", "pbRatio", "pfcfRatio", "pocfratio",
        "evToSales", "enterpriseValueOverEBITDA",
        "evToOperatingCashFlow", "evToFreeCashFlow",
        "earningsYield", "freeCashFlowYield",

        # Calidad / riesgo
        "netDebtToEBITDA", "debtToEquity", "currentRatio", "interestCoverage",

        # Per-share (muy útil para horizons largos)
        "revenuePerShare", "netIncomePerShare",
        "operatingCashFlowPerShare", "freeCashFlowPerShare",
        "bookValuePerShare", "tangibleBookValuePerShare",
        "capexPerShare",
        "dividendYield", "payoutRatio",
    },
    "key_metrics_ttm": {
        # típicamente viene con sufijos TTM, pero depende del endpoint
        # dejamos ambos estilos (con y sin TTM) para “cazar” lo que venga
        "peRatioTTM", "pbRatioTTM", "pfcfRatioTTM", "pocfratioTTM",
        "enterpriseValueOverEBITDATTM", "freeCashFlowYieldTTM", "earningsYieldTTM",
        "revenuePerShareTTM", "netIncomePerShareTTM",
        "operatingCashFlowPerShareTTM", "freeCashFlowPerShareTTM",
        "dividendYieldTTM", "payoutRatioTTM",

        # fallback por si el endpoint no usa sufijo
        "peRatio", "pbRatio", "pfcfRatio", "pocfratio",
        "enterpriseValueOverEBITDA", "freeCashFlowYield", "earningsYield",
        "revenuePerShare", "netIncomePerShare",
        "operatingCashFlowPerShare", "freeCashFlowPerShare",
        "dividendYield", "payoutRatio",
    },
}

def endpoint_prefix(path: str) -> str:
    # '/stable/key-metrics-ttm' -> 'key_metrics_ttm'
    return path.strip("/").split("/")[-1].replace("-", "_")

def normalize_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []

def to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"null", "none", "nan"}:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None

def extract_features(symbol: str, path: str, data: Any, fetch_iso: str) -> list[tuple[str, str, str, float, str]]:
    pref = endpoint_prefix(path)
    wanted = FEATURE_KEYS.get(pref)
    if not wanted:
        return []

    rows: list[tuple[str, str, str, float, str]] = []
    items = normalize_payload(data)

    # Para endpoints históricos, vienen varios periodos con 'date'
    # Para profile, suele venir lista con 1 dict
    for item in items:
        asof_date = (
            item.get("date")
            or item.get("reportedDate")
            or item.get("calendarYear")
            or fetch_iso[:10]
        )

        for k in wanted:
            v = to_float(item.get(k))
            if v is None:
                continue

            feature_name = f"{pref}.{k}"
            rows.append((symbol, str(asof_date), feature_name, v, fetch_iso))

    return rows

def setup_database():
    os.makedirs(DB_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Raw snapshots (ya lo tenías, lo dejo igual)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fmp_json_data_snapshots (
            symbol TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            fetch_date TEXT NOT NULL,
            data_json TEXT NOT NULL,
            PRIMARY KEY (symbol, endpoint, fetch_date)
        )
    """)

    # Features listas (nuevo)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fmp_features (
            symbol TEXT NOT NULL,
            asof_date TEXT NOT NULL,
            feature TEXT NOT NULL,
            value REAL,
            fetch_date TEXT NOT NULL,
            PRIMARY KEY (symbol, asof_date, feature)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_fmp_features_symbol_asof
        ON fmp_features(symbol, asof_date)
    """)

    conn.commit()
    conn.close()

def fetch_and_store():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    api_calls_made = 0
    total_snapshots = 0
    total_features_upserted = 0

    print("\n--- Starting FMP fetch (reducido + features) ---")
    for symbol in FMP_SYMBOLS:
        print(f"\nFetching {symbol}...")
        for ep in ENDPOINTS_TO_FETCH:
            path = ep["path"]
            is_historical = ep["is_historical"]

            try:
                params: dict[str, Any] = {"apikey": FMP_API_KEY}

                if path == "/api/v3/profile":
                    url = f"{FMP_BASE_URL}{path}/{symbol}"
                else:
                    url = f"{FMP_BASE_URL}{path}"
                    params["symbol"] = symbol
                    if is_historical:
                        params["limit"] = PERIODS_TO_FETCH

                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                api_calls_made += 1

                data = r.json()
                if not data:
                    print(f"  - No data for {path}")
                    continue

                fetch_iso = datetime.now().isoformat()

                # 1) Guardar snapshot crudo
                cur.execute(
                    """
                    INSERT OR REPLACE INTO fmp_json_data_snapshots (symbol, endpoint, fetch_date, data_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (symbol, path, fetch_iso, json.dumps(data)),
                )
                total_snapshots += cur.rowcount

                # 2) Extraer features y guardarlas listas
                feat_rows = extract_features(symbol, path, data, fetch_iso)
                if feat_rows:
                    cur.executemany(
                        """
                        INSERT OR REPLACE INTO fmp_features (symbol, asof_date, feature, value, fetch_date)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        feat_rows,
                    )
                    total_features_upserted += len(feat_rows)

                print(f"  - OK: {path} | features: {len(feat_rows)}")

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                print(f"  - FAILED {path}: HTTP {status}")
            except Exception as e:
                print(f"  - ERROR {path}: {e}")

    conn.commit()
    conn.close()

    print("\n--- Finished ---")
    print(f"API calls: {api_calls_made}")
    print(f"Snapshots upserted: {total_snapshots}")
    print(f"Feature rows upserted: {total_features_upserted}")

if __name__ == "__main__":
    setup_database()
    fetch_and_store()
