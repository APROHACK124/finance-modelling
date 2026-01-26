# fetch_economic_data.py

def main():

    import os
    import requests
    import sqlite3
    from dotenv import load_dotenv
    from datetime import datetime

    # --- Configuration ---
    load_dotenv()

    # --- Database Path Setup ---
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    DB_FOLDER = os.path.join(SCRIPT_DIR, 'data')
    DB_FILE = os.path.join(DB_FOLDER, 'stock_data.db')

    # --- Alpha Vantage Configuration ---
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

    def setup_database_economic():
        """Sets up the database table for storing economic indicators."""
        print("Setting up 'economic_indicators' table...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_indicators (
                indicator_name TEXT NOT NULL,
                date DATE NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (indicator_name, date)
            )
        ''')
        conn.commit()
        conn.close()
        print("Database setup for economic data complete.")

    def fetch_and_store_economic_data():
        """Fetches key economic indicators from Alpha Vantage."""
        if not ALPHA_VANTAGE_API_KEY:
            print("Error: ALPHA_VANTAGE_API_KEY not found in .env file. Skipping economic data fetch.")
            return

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Define the economic indicators we want from Alpha Vantage
        indicators_to_fetch = {
            'REAL_GDP': {'interval': 'quarterly'},
            'CPI': {'interval': 'monthly'},
            'UNEMPLOYMENT': {'interval': 'monthly'},
            'FEDERAL_FUNDS_RATE': {'interval': 'monthly'}
        }

        total_new_points = 0
        api_calls_made = 0
        
        print("\n--- Fetching Economic Data from Alpha Vantage ---")
        for name, config in indicators_to_fetch.items():
            print(f"  - Fetching {name}...")
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': name,
                    'interval': config['interval'],
                    'datatype': 'json',
                    'apikey': ALPHA_VANTAGE_API_KEY
                }
                if name == 'REAL_GDP':
                    del params['interval'] # GDP endpoint doesn't use 'interval'

                response = requests.get(url, params=params)
                response.raise_for_status()
                api_calls_made += 1
                
                data = response.json().get('data', [])
                if not data:
                    print(f"    - No data found for {name}.")
                    continue

                sql = "INSERT OR IGNORE INTO economic_indicators (indicator_name, date, value) VALUES (?, ?, ?)"
                new_rows = 0
                for point in data:
                    # Some values might be '.', so we check if it's a valid number
                    try:
                        value = float(point['value'])
                        cursor.execute(sql, (name, point['date'], value))
                        new_rows += cursor.rowcount
                    except (ValueError, TypeError):
                        continue # Skip points where value is not a number
                
                total_new_points += new_rows
                print(f"    -> Stored {new_rows} new data points for {name}.")

            except Exception as e:
                print(f"    - Error fetching {name}: {e}")
                # The API might return a note about call frequency, which we can print
                if 'note' in response.text.lower():
                    print(f"      API Note: {response.json().get('Note')}")
        
        conn.commit()
        conn.close()

        print("\n--- Economic Data Fetch Finished ---")
        print(f"Total API calls made: {api_calls_made} (Limit: 25/day)")
        print(f"Total new data points added to database: {total_new_points}")

    setup_database_economic()
    fetch_and_store_economic_data()

if __name__ == "__main__":
    main()