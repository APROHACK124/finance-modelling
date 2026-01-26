# calculate_indicators.py

import os
import sqlite3
import pandas as pd
import pandas_ta as ta

# --- ROBUST DATABASE PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DB_FOLDER = os.path.join(SCRIPT_DIR, 'data')
DB_FILE = os.path.join(DB_FOLDER, 'stock_data.db')

def calculate_and_store_indicators():
    """
    Reads raw stock bar data, validates it, calculates technical indicators
    only if there is enough data, and stores them in a new table.
    """
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file not found at {DB_FILE}")
        return

    print(f"Connecting to database at {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)

    try:
        print("Loading raw stock bar data...")
        df = pd.read_sql_query("SELECT * FROM stock_bars", conn)
        print(f"Loaded {len(df)} rows.")

        if df.empty:
            print("The 'stock_bars' table is empty.")
            return
        
        # Unconditionally drop any rows with missing data to be safe
        df.dropna(inplace=True)
        
        # Convert timestamp to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        all_indicators_df = []

        print("\nCalculating indicators for each symbol and timeframe...")
        for group_keys, group_df in df.groupby(['symbol', 'timeframe']):
            symbol, timeframe = group_keys
            
            # Sort data chronologically, which is essential for indicators
            group_df = group_df.sort_values(by='timestamp').copy()
            n = len(group_df)
            print(f"  - Processing {symbol} ({timeframe}) which has {n} data points...")

            # Define the minimum number of periods needed for each indicator
            sma_len = 20
            ema_len = 50
            rsi_len = 14
            macd_fast = 12
            macd_slow = 26
            macd_signal = 9
            bbands_len = 20
            atr_len = 14
            
            # Only calculate indicators if the number of rows (n) is sufficient
            if n > sma_len:
                group_df.ta.sma(length=sma_len, append=True)
            
            if n > ema_len:
                group_df.ta.ema(length=ema_len, append=True)
            
            if n > rsi_len:
                group_df.ta.rsi(length=rsi_len, append=True)
            
            # --- ROBUST FIX for MACD ---
            # We require enough data for the slow period + signal period to avoid the library bug.
            if n > macd_slow + macd_signal:
                group_df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
            
            if n > bbands_len:
                group_df.ta.bbands(length=bbands_len, std=2, append=True)

            if n > atr_len:
                group_df.ta.atr(length=atr_len, append=True)
            
            all_indicators_df.append(group_df)

        # Combine all processed groups back into a single DataFrame
        final_df = pd.concat(all_indicators_df)

        # Define all possible indicator columns we might have created
        indicator_columns = [
            'symbol', 'timestamp', 'timeframe',
            f'SMA_{sma_len}', f'EMA_{ema_len}', f'RSI_{rsi_len}',
            f'MACD_{macd_fast}_{macd_slow}_{macd_signal}', f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}', f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}',
            f'BBL_{bbands_len}_2.0', f'BBM_{bbands_len}_2.0', f'BBU_{bbands_len}_2.0', f'BBB_{bbands_len}_2.0', f'BBP_{bbands_len}_2.0',
            f'ATRr_{atr_len}'
        ]
        
        # Select only the primary key and indicator columns that actually exist in the final DataFrame
        final_indicator_columns = [col for col in indicator_columns if col in final_df.columns]
        final_df_to_save = final_df[final_indicator_columns]
        
        # Drop rows where indicators are naturally NaN (e.g., the first 20 rows for a 20-period SMA)
        final_df_to_save = final_df_to_save.dropna()

        print(f"\nSaving {len(final_df_to_save)} rows with calculated indicators to the 'stock_indicators' table...")
        final_df_to_save.to_sql('stock_indicators', conn, if_exists='replace', index=False)
        print("Successfully created/updated the 'stock_indicators' table.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()
        print("Database connection closed.")


if __name__ == "__main__":
    print("--- Starting Indicator Calculation Script ---")
    calculate_and_store_indicators()
    print("\n--- Script Finished ---")