# database_tier1.py

from time import sleep
from python_scripts.LLM_analysis.preprocess_store_database import get_connection
TARGET_STOCKS = [
        # ETFs (20)
        "SPY","QQQ","IWM","DIA","VTI",
        "XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY",
        "TLT","IEF","LQD","GLD",

        # Stocks (60)
        "AAPL","MSFT","AMZN","GOOGL","META","NFLX","ORCL","CRM","ADBE","INTU","CSCO",
        "NVDA","AMD","AVGO","QCOM","TXN","MU","AMAT","LRCX","KLAC","TSM",
        "JPM","BAC","WFC","GS","MS","V","MA","BLK",
        "UNH","JNJ","MRK","ABBV","TMO","AMGN",
        "TSLA","HD","LOW","MCD","NKE","DIS","SBUX",
        "PG","KO","PEP","WMT","COST",
        "CAT","DE","HON","GE","RTX","LMT",
        "XOM","CVX","COP",
        "LIN","FCX",
        "NEE","AMT", "ASML",

        # CRIPTO
        "BTC"
    ]

def main():

    import os
    import requests
    import json
    import sqlite3
    from dotenv import load_dotenv
    # CHANGED: Imported timezone to address the deprecation warning
    from datetime import datetime, timedelta, timezone

    # --- Configuration ---
    load_dotenv() # Load environment variables from .env file

    # --- ROBUST DATABASE PATH SETUP ---
    # Get the absolute directory path of the script file
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Define the path to the 'data' subfolder
    DB_FOLDER = os.path.join(SCRIPT_DIR, 'data')
    # Define the full path to the database file
    DB_FILE = os.path.join(DB_FOLDER, 'stock_data.db')

    # Your list of stocks
    # TARGET_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'] OLD
    
    DAYS_OF_HISTORY_TO_FETCH = 4 # How many days of daily/hourly data to get
    MINUTES_OF_HISTORY_TO_FETCH = 30

    # --- NEW CONFIGURATION FOR BACKFILLING ---
    # How many days of older history to fetch in each run.
    # This helps gradually build your dataset without making a huge single request.
    DAYS_TO_BACKFILL_PER_RUN = 35
    # Note: The Alpaca free plan uses SIP data, which typically only goes back a few years.


    # Get API keys from environment
    API_KEY = os.getenv('APCA_API_KEY_ID')
    SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

    # Alpaca API headers
    HEADERS = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    def setup_database():
        """Creates/updates the SQLite database and the unified stock_bars table."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_bars (
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                timeframe TEXT NOT NULL, 
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                trade_count INTEGER NOT NULL,
                vwap REAL NOT NULL,
                PRIMARY KEY (symbol, timestamp, timeframe)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_stock_bars_sym_tf_ts
            ON stock_bars(symbol, timeframe, timestamp);
        ''')


        
        
        conn.commit()
        conn.close()
        print("Database setup complete with unified 'stock_bars' table.")

    def store_bars_to_db(cursor, symbol, bars, timeframe):
        """
        A helper function to store a list of bars in the database.
        """
        sql = '''
            INSERT INTO stock_bars (
            symbol, timestamp, timeframe, open, high, low, close, volume, trade_count, vwap
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timestamp, timeframe) DO UPDATE SET
            open        = excluded.open,
            high        = excluded.high,
            low         = excluded.low,
            close       = excluded.close,
            volume      = excluded.volume,
            trade_count = excluded.trade_count,
            vwap        = excluded.vwap
        WHERE excluded.trade_count >= stock_bars.trade_count
          AND excluded.volume      >= stock_bars.volume
        '''
        
        new_entries_count = 0 # ADDED: Counter for this specific batch
        for bar in bars:
            data_tuple = (
                symbol, bar['t'], timeframe, bar['o'], bar['h'], 
                bar['l'], bar['c'], bar['v'], bar['n'], bar['vw']
            )
            cursor.execute(sql, data_tuple)
            new_entries_count += cursor.rowcount # ADDED: Add 1 if a row was inserted, 0 if ignored

        return new_entries_count # ADDED: Return the final count

    def fetch_and_store_data():
        """Fetches and stores data for all stocks and all required timeframes."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # --- 1. Fetch LATEST 1-MINUTE BARS (Unchanged) ---

        '''
        print("\nFetching latest 1-Minute bars...")
        symbols_str = ",".join(TARGET_STOCKS)
        url_latest = f"https://data.alpaca.markets/v2/stocks/bars/latest?symbols={symbols_str}&feed=sip"
        try:
            response = requests.get(url_latest, headers=HEADERS)
            response.raise_for_status()
            latest_bars_data = response.json().get('bars', {})
            added = 0
            for symbol, bar in latest_bars_data.items():
                count = store_bars_to_db(cursor, symbol, [bar], '1Min') # Note: store_bars_to_db now needs to handle dicts
                added += count
            print(f"Successfully processed latest 1-Min bars for {len(latest_bars_data)} symbols. ({added} rows added)")
        except Exception as e:
            print(f"Could not fetch latest 1-Min bars. Error: {e}")

        '''

        # --- Setup for Historical Data ---
        end_date = (datetime.now(timezone.utc) - timedelta(minutes=40)).isoformat()
        start_date = (datetime.now(timezone.utc) - timedelta(days=DAYS_OF_HISTORY_TO_FETCH)).isoformat()
        added = 0
        # --- 2. Fetch HISTORICAL 15-MINUTE BARS (SIMPLIFIED) ---

        '''
        Not necessary now
        print("\nFetching historical 15-Minute bars...")
        for symbol in TARGET_STOCKS:
            url_hist = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
            params = {
                'timeframe': '15Min', # <-- CHANGED: Request 15Min directly
                'start': start_date,
                'end': end_date,
                'limit': 10000,
                'adjustment': 'raw',
                'feed': 'sip'
            }
            try:
                response = requests.get(url_hist, headers=HEADERS, params=params)
                response.raise_for_status()
                hist_bars_data = response.json().get('bars', [])
                
                if hist_bars_data:
                    # No resampling needed, just store the data directly
                    count = store_bars_to_db(cursor, symbol, hist_bars_data, '15Min')
                    added += count

                    print(f"  - Stored {len(hist_bars_data)} 15-Minute bars for {symbol}. ({added} rows added).")
                else:
                    print(f"  - No new 15-Minute bars found for {symbol}.")
            except Exception as e:
                print(f"  - Could not fetch 15-Min bars for {symbol}. Error: {e}")
        '''
                
        # --- 3. Fetch HISTORICAL 1-DAY BARS (Unchanged) ---
        print("\nFetching historical 1Day bars...")
        for symbol in TARGET_STOCKS:
            url_hist = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
            params = {
                'timeframe': '1Day',
                'start': start_date,
                'end': end_date,
                'limit': 10000,
                'adjustment': 'raw',
                'feed': 'sip'
            }
            try:
                response = requests.get(url_hist, headers=HEADERS, params=params)
                response.raise_for_status()
                hist_bars_data = response.json().get('bars', [])
                if hist_bars_data:
                    added = store_bars_to_db(cursor, symbol, hist_bars_data, '1Day')
                    print(f"  - Stored {len(hist_bars_data)} 1Day bars for {symbol}. ({added} rows added).")
                else:
                    print(f"  - No new 1Day bars found for {symbol}.")
            except Exception as e:
                print(f"  - Could not fetch 1Day bars for {symbol}. Error: {e}")

        conn.commit()
        conn.close()

    # --- NEW FUNCTION TO BACKFILL OLDER DATA ---
    def backfill_historical_data():
        """
        Checks the database for the oldest data point and fetches even older data,
        allowing for the gradual build-up of a large historical dataset.
        """
        print("\n--- Starting Historical Data Backfill Process ---")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # We typically backfill longer timeframes like 15Min and 1Day
        timeframes_to_backfill = ['1Day']

        for symbol in TARGET_STOCKS:
            for timeframe in timeframes_to_backfill:
                print(f"\nChecking backfill for {symbol} with timeframe {timeframe}...")
                
                # Find the earliest timestamp we already have for this symbol and timeframe
                cursor.execute("""
                    SELECT MIN(timestamp) FROM stock_bars
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                result = cursor.fetchone()
                oldest_timestamp_str = result[0] if result and result[0] else None

                if not oldest_timestamp_str:
                    print(f"  - No existing data found for {symbol} ({timeframe}). Backfill will not run.")
                    print("  - Run the recent fetch first to get a starting point.")
                    continue

                # The new 'end' date for our request is the oldest date we have.
                # Alpaca's API 'end' parameter is exclusive, so we won't re-fetch this exact bar.
                end_date_dt = datetime.fromisoformat(oldest_timestamp_str.replace('Z', '+00:00'))
                start_date_dt = end_date_dt - timedelta(days=DAYS_TO_BACKFILL_PER_RUN)

                print(f"  - Oldest record is {end_date_dt.date()}. Fetching data from {start_date_dt.date()} to {end_date_dt.date()}.")

                url_hist = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
                params = {
                    'timeframe': timeframe,
                    'start': start_date_dt.isoformat(),
                    'end': end_date_dt.isoformat(),
                    'limit': 10000,
                    'adjustment': 'raw',
                    'feed': 'sip'
                }
                
                try:
                    response = requests.get(url_hist, headers=HEADERS, params=params)
                    response.raise_for_status()
                    hist_bars_data = response.json().get('bars', [])
                    
                    if hist_bars_data:
                        count = store_bars_to_db(cursor, symbol, hist_bars_data, timeframe)
                        print(f"  - SUCCESS: Backfilled and stored {count} new historical bars for {symbol}.")
                    else:
                        print(f"  - No older data found for {symbol} in this period. History might be complete for this stock.")
                
                except requests.exceptions.HTTPError as e:
                    # Alpaca often returns a 422 error if the start date is too far in the past for the SIP feed
                    if e.response.status_code == 422:
                        print(f"  - Could not fetch older data for {symbol}. The start date may be too old for the SIP feed.")
                    else:
                        print(f"  - An HTTP error occurred for {symbol}. Error: {e}")
                except Exception as e:
                    print(f"  - An unexpected error occurred for {symbol}. Error: {e}")

        conn.commit()
        conn.close()
        print("\n--- Historical Data Backfill Finished ---")

    
    # ChatGPT wrote this
    # This is to fill gaps

    import time

    def alpaca_throttle(requests_made, max_per_minute=150):
        """
        Freno simple para no pasarnos de rate limit.
        """
        if requests_made >= max_per_minute:
            print("⏸️  Rate limit alcanzado, durmiendo 60s...")
            time.sleep(60)
            return 0
        return requests_made

    def get_alpaca_calendar(headers, base_url, start_date: str, end_date: str):
        """
        Pide a Alpaca el calendario de mercado entre dos fechas (YYYY-MM-DD).
        Devuelve una lista de dicts con: date, open, close.
        """
        import requests

        url = f"{base_url}/v2/calendar"
        params = {"start": start_date, "end": end_date}

        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    
    def build_sessions_utc(calendar_rows):
        import pandas as pd

        tz = "America/New_York"
        sessions = []
        for row in calendar_rows:
            date = row["date"]
            o = row["open"]
            c = row["close"]
            open_local = pd.Timestamp(f"{date} {o}", tz=tz)
            close_local = pd.Timestamp(f"{date} {c}", tz=tz)
            sessions.append((open_local.tz_convert("UTC"), close_local.tz_convert("UTC")))

        sessions.sort(key=lambda x: x[0])
        return sessions
    
    from bisect import bisect_right

    def count_expected_starts_in_gap(gap_start, gap_end, sessions, step):
        """
        gap_start/gap_end: pandas Timestamp UTC
        sessions: lista de (open_utc, close_utc)
        step: pandas Timedelta (ej 15min)

        Cuenta cuántos posibles "inicios de barra" caen dentro del gap (gap_start, gap_end)
        durante sesiones. Si da 0, probablemente es overnight/weekend normal.
        """
        import pandas as pd

        step_ns = step.value
        gs = gap_start.value
        ge = gap_end.value

        # Precomputamos arrays de cierre para ubicar rápido por bisección
        closes = [c.value for (_, c) in sessions]
        opens  = [o.value for (o, _) in sessions]

        i = bisect_right(closes, gs)  # primer día cuya sesión cierra después de gap_start
        missing = 0

        while i < len(sessions) and opens[i] < ge:
            o_ns = opens[i]
            c_ns = closes[i]

            last_start_ns = c_ns - step_ns
            if last_start_ns < o_ns:
                i += 1
                continue

            max_k = (last_start_ns - o_ns) // step_ns

            # k_low = floor((gs - o)/step) + 1, clamp >=0
            k_low = (gs - o_ns) // step_ns + 1
            if k_low < 0:
                k_low = 0

            # k_high = ceil((ge - o)/step) - 1, clamp <= max_k
            ceil_div = -((-(ge - o_ns)) // step_ns)   # ceil(a/b)
            k_high = ceil_div - 1
            if k_high > max_k:
                k_high = max_k

            if k_high >= k_low:
                missing += (k_high - k_low + 1)

            i += 1

        return missing

    def find_gaps_fast(conn, headers, trading_base_url, symbol: str, timeframe: str):
        import pandas as pd

        tf_to_step = {"15Min": pd.Timedelta(minutes=15), "1Day": pd.Timedelta(days=1)}
        step = tf_to_step[timeframe]

        df = pd.read_sql_query(
            """
            SELECT timestamp
            FROM stock_bars
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp
            """,
            conn,
            params=(symbol, timeframe),
        )
        if df.empty:
            return pd.DataFrame(columns=["hole_start", "hole_end", "delta", "expected_missing_bars"])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        df["prev"] = df["timestamp"].shift(1)
        df["delta"] = df["timestamp"] - df["prev"]

        # candidatos: saltos > step
        cand = df[df["delta"] > step].dropna(subset=["prev"]).copy()
        if cand.empty:
            return pd.DataFrame(columns=["hole_start", "hole_end", "delta", "expected_missing_bars"])

        # calendario una sola vez (rango mínimo)
        min_date = cand["prev"].min().tz_convert("America/New_York").date().isoformat()
        max_date = cand["timestamp"].max().tz_convert("America/New_York").date().isoformat()

        cal = get_alpaca_calendar(headers, trading_base_url, min_date, max_date)
        sessions = build_sessions_utc(cal)

        rows = []
        for _, r in cand.iterrows():
            hs = r["prev"]
            he = r["timestamp"]

            missing = count_expected_starts_in_gap(hs, he, sessions, step)

            # Si missing==0 -> típico overnight/weekend o borde de cierre.
            if missing > 0:
                rows.append({
                    "hole_start": hs,
                    "hole_end": he,
                    "delta": r["delta"],
                    "expected_missing_bars": missing
                })
        if len(rows) == 0:
            return pd.DataFrame(columns=["hole_start", "hole_end", "delta", "expected_missing_bars"])
        return pd.DataFrame(rows).sort_values("hole_start").reset_index(drop=True)


    
    def expected_bar_starts_for_session(date_str: str, open_str: str, close_str: str, freq: str):
            """
            Genera los timestamps (timezone NY) donde deberían empezar las barras.
            date_str: 'YYYY-MM-DD'
            open_str/close_str: 'HH:MM'
            freq: '15min', '1min', etc.
            """
            import pandas as pd

            tz = "America/New_York"

            session_open = pd.Timestamp(f"{date_str} {open_str}").tz_localize(tz)
            session_close = pd.Timestamp(f"{date_str} {close_str}").tz_localize(tz)

            # Las barras representan intervalos [start, start+freq)
            # así que el último start válido es close - freq
            last_start = session_close - pd.Timedelta(freq)

            if last_start < session_open:
                return pd.DatetimeIndex([], tz=tz)

            return pd.date_range(session_open, last_start, freq=freq)
        
    def count_expected_bars_in_gap(gap_start_utc, gap_end_utc, calendar_rows, freq: str):
        """
        gap_start_utc y gap_end_utc: pandas Timestamp tz-aware (UTC)
        calendar_rows: lista de dicts del calendario de Alpaca
        freq: '15min', etc.

        Devuelve cuántas barras *deberían* existir entre esos dos timestamps
        durante horas de mercado.
        """
        import pandas as pd

        # Pasamos el gap a horario de NY para compararlo con el calendario
        gap_start_local = gap_start_utc.tz_convert("America/New_York")
        gap_end_local = gap_end_utc.tz_convert("America/New_York")

        expected_count = 0

        for row in calendar_rows:
            date_str = row["date"]
            open_str = row["open"]
            close_str = row["close"]

            expected = expected_bar_starts_for_session(date_str, open_str, close_str, freq)

            # Solo cuentan las barras estrictamente dentro del intervalo
            inside = expected[(expected > gap_start_local) & (expected < gap_end_local)]
            expected_count += len(inside)

        return expected_count
    

    def find_real_gaps_from_db(conn, headers, trading_base_url, symbol: str, timeframe: str):
        import pandas as pd

        # 1) Cargamos timestamps de un símbolo y timeframe
        df = pd.read_sql_query(
            """
            SELECT timestamp
            FROM stock_bars
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp
            """,
            conn,
            params=(symbol, timeframe),
        )

        if df.empty:
            return pd.DataFrame(columns=["hole_start", "hole_end", "delta", "expected_missing_bars"])

        # 2) Parse a datetime UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        raw = df["timestamp"].copy()
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        bad = parsed.isna().sum()
        # print("timestamps inválidos (NaT):", bad)
        # if bad:
            # print("ejemplos inválidos:", raw[parsed.isna()].head(10).tolist())
        
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # 3) Frecuencia esperada según timeframe
        tf_to_freq = {
            "1Min": "1min",
            "5Min": "5min",
            "15Min": "15min",
            "1Hour": "1h",
        }
        if timeframe not in tf_to_freq:
            raise ValueError(f"No tengo mapeo de frecuencia para timeframe={timeframe}")

        freq = tf_to_freq[timeframe]
        expected_step = pd.Timedelta(freq)

        # 4) Candidatos: saltos mayores que 1 barra
        df["prev_ts"] = df["timestamp"].shift(1)
        df["delta"] = df["timestamp"] - df["prev_ts"]
        candidates = df[df["delta"] > expected_step].copy()
        candidates = candidates.dropna(subset=["prev_ts"])

        if candidates.empty:
            return pd.DataFrame(columns=["hole_start", "hole_end", "delta", "expected_missing_bars"])

        # 5) Pedimos el calendario para el rango que cubren los gaps (en fechas NY)
        min_date = candidates["prev_ts"].min().tz_convert("America/New_York").date().isoformat()
        max_date = candidates["timestamp"].max().tz_convert("America/New_York").date().isoformat()

        calendar_rows = get_alpaca_calendar(headers, trading_base_url, min_date, max_date)

        # 6) Para cada gap candidato, calculamos cuántas barras “deberían” existir dentro
        rows = []
        for _, r in candidates.iterrows():
            hole_start = r["prev_ts"]
            hole_end = r["timestamp"]

            expected_missing = count_expected_bars_in_gap(hole_start, hole_end, calendar_rows, freq)

            # Si expected_missing == 0, el salto probablemente es overnight/weekend/holiday (normal)
            if expected_missing > 0:
                rows.append({
                    "hole_start": hole_start,
                    "hole_end": hole_end,
                    "delta": r["delta"],
                    "expected_missing_bars": expected_missing,
                })

        return pd.DataFrame(rows).sort_values(["hole_start"]).reset_index(drop=True)
    
    def merge_gaps_into_windows(gaps_df, max_merge_minutes=60):
        """
        Une gaps cercanos en ventanas grandes.
        max_merge_minutes: si dos gaps están a <= este tiempo, se unen.
        """
        import pandas as pd

        if gaps_df.empty:
            return []

        gaps_df = gaps_df.sort_values("hole_start")

        windows = []
        current_start = gaps_df.iloc[0]["hole_start"]
        current_end   = gaps_df.iloc[0]["hole_end"]

        for _, row in gaps_df.iloc[1:].iterrows():
            gap = row["hole_start"] - current_end

            if gap <= pd.Timedelta(minutes=max_merge_minutes):
                # unir
                current_end = max(current_end, row["hole_end"])
            else:
                windows.append((current_start, current_end))
                current_start = row["hole_start"]
                current_end   = row["hole_end"]

        windows.append((current_start, current_end))
        return windows


    def fetch_bars_paginated(headers, symbol: str, timeframe: str, start_iso: str, end_iso: str, feed="sip"):
        import requests

        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": timeframe,
            "start": start_iso,
            "end": end_iso,
            "limit": 10000,
            "adjustment": "raw",
            "feed": feed,
        }

        out = []
        page_token = None

        while True:
            if page_token:
                params["page_token"] = page_token

            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            # debugging
            # print("status:", r.status_code, "len:", len(data.get("bars", [])))
            # if data.get("bars"):
                # print("first t:", data["bars"][0]["t"], "last t:", data["bars"][-1]["t"])
            # else:
                # print("response keys:", list(data.keys()))
                


            out.extend(data.get("bars", []))

            page_token = data.get("next_page_token")
            if not page_token:
                break

        return out

    def fill_real_gaps(conn, cursor, headers, gaps_df, symbol: str, timeframe: str):
        """
        gaps_df: DataFrame con hole_start/hole_end
        """
        import pandas as pd

        total_added = 0

        for _, row in gaps_df.iterrows():
            start_ts = row["hole_start"]
            end_ts = row["hole_end"]

            # Alpaca usa ISO 8601
            start_iso = pd.Timestamp(start_ts).isoformat()
            end_iso = pd.Timestamp(end_ts).isoformat()

            # print(f"\nRellenando gap {symbol} {timeframe}: {start_iso} -> {end_iso} "
                # f"(esperadas ~{row['expected_missing_bars']} barras)")

            bars = fetch_bars_paginated(headers, symbol, timeframe, start_iso, end_iso, feed="sip") #NOT UPDATED

            if not bars:
                # print("  - Alpaca devolvió 0 barras en este rango.")
                continue

            added_now = store_bars_to_db(cursor, symbol, bars, timeframe)
            conn.commit()

            total_added += added_now
            print(f"  - Insertadas: {added_now} (recibidas: {len(bars)})")

        return total_added
    
    def fill_gaps_by_windows(conn, cursor, headers, symbol, timeframe, windows, feed='sip'):
        import pandas as pd
        import time

        total_added = 0
        requests_made = 0

        for start_ts, end_ts in windows:
            start_iso = pd.Timestamp(start_ts).isoformat()
            end_iso   = pd.Timestamp(end_ts).isoformat()

            # print(f"\n Rellenando ventana {symbol} {timeframe}")
            # print(f"    {start_iso} → {end_iso}")

            bars = fetch_bars_paginated(
                headers=headers,
                symbol=symbol,
                timeframe=timeframe,
                start_iso=start_iso,
                end_iso=end_iso,
                feed=feed
            )

            requests_made += 1

            if bars:
                added = store_bars_to_db(cursor, symbol, bars, timeframe)
                conn.commit()
                total_added += added
                # print(f"    ✔ {added} barras insertadas ({len(bars)} recibidas)")
            else:
                print("    ⚠ Alpaca no devolvió barras")

            # ⏸ dormir SIEMPRE un poco
            time.sleep(0.5)

        return total_added

    def fix_gaps_for_symbol(conn, cursor, headers, trading_base_url, symbol: str, timeframe: str):
        import pandas as pd

        # 1) Encuentra gaps reales
        gaps = find_gaps_fast(conn, headers, trading_base_url, symbol, timeframe)

        if gaps.empty:
            print(f"✅ {symbol} {timeframe}: no hay gaps reales.")
            return 0

        print(f"⚠️ {symbol} {timeframe}: gaps reales encontrados = {len(gaps)}")
        # print(gaps.head(5))

        # 2) Agrupar gaps en ventanas grandes (reduce llamadas)
        windows = merge_gaps_into_windows(gaps, max_merge_minutes=180)
        # print(f"📦 {symbol} {timeframe}: ventanas generadas = {len(windows)}")

        # 3) Limitar trabajo por ejecución (CRUCIAL)
        MAX_WINDOWS_PER_RUN = 3
        windows = windows[:MAX_WINDOWS_PER_RUN]

        # 4) Rellenar por ventanas
        added = fill_gaps_by_windows(conn, cursor, headers, symbol, timeframe, windows, feed="sip")

        # 5) (Opcional) Re-chequear rápido
        #gaps_after = find_real_gaps_from_db(conn, headers, trading_base_url, symbol, timeframe)
        #print(f"🔎 {symbol} {timeframe}: gaps después = {len(gaps_after)}")

        return added








    print("--- Starting Full Data Logging Script ---")
    
    # 1. Ensure the database is ready
    setup_database()

    # 2. Fetch the most recent data (today, this hour, etc.)
    fetch_and_store_data()

    # 3. Backfill older data that is missing from your history

    backfill_historical_data()


    # 4. Filling gaps
    
    print("\n--- Filling REAL gaps (mid-history holes) ---")

    import sqlite3
    import os

    ALPACA_TRADING_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    #TIMEFRAMES_TO_FIX = ["15Min", "1Day"]
    TIMEFRAMES_TO_FIX = ["1Day"]

    total_added = 0
    for tf in TIMEFRAMES_TO_FIX:
        for symbol in TARGET_STOCKS:
            total_added += fix_gaps_for_symbol(conn, cursor, HEADERS, ALPACA_TRADING_BASE_URL, symbol, tf)

    conn.commit()
    conn.close()

    print(f"\n✅ Gap filling finished. Total rows added = {total_added}")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main()
