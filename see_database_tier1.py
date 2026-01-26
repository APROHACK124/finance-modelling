import sqlite3
import pandas as pd
import os

# --- Configuration ---

# --- ROBUST DATABASE PATH SETUP ---
# Get the absolute directory path of the script file
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# Define the path to the 'data' subfolder
DB_FOLDER = os.path.join(SCRIPT_DIR, 'data')
# Define the full path to the database file
DB_FILE = os.path.join(DB_FOLDER, 'stock_data.db')

def view_database_contents():
    """Connects to the SQLite database and prints its contents using Pandas."""
    
    # First, check if the database file actually exists
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file '{DB_FILE}' not found.")
        print("Please run the data_logger.py script first to create and populate it.")
        return

    conn = None # Initialize conn to None
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(DB_FILE)
        
        print(f"--- Successfully connected to {DB_FILE} ---")
        
        # --- Example 1: View the LATEST 10 entries in the entire table ---
        print("\n=======================================================")
        print("           Most Recent 10 Entries Overall")
        print("=======================================================")
        
        query_all = "SELECT * FROM stock_bars ORDER BY timestamp DESC LIMIT 10"
        df_all = pd.read_sql_query(query_all, conn)
        
        if df_all.empty:
            print("The database is empty. No data to display.")
        else:
            print(df_all.to_string()) # .to_string() ensures all columns are shown


        # --- Example 2: View all data for a SINGLE stock (e.g., AAPL) ---
        print("\n\n=======================================================")
        print("              All Stored Data for AAPL")
        print("=======================================================")
        
        symbol_to_check = 'AAPL'
        query_symbol = f"SELECT * FROM stock_bars WHERE symbol = '{symbol_to_check}' ORDER BY timestamp DESC"
        df_symbol = pd.read_sql_query(query_symbol, conn)
        
        if df_symbol.empty:
            print(f"No data found for symbol: {symbol_to_check}")
        else:
            print(df_symbol.to_string())
            
            
        # --- Example 3: View a count of entries per stock ---
        print("\n\n=======================================================")
        print("            Total Entries Logged Per Stock")
        print("=======================================================")
        
        query_count = "SELECT symbol, COUNT(*) as entry_count FROM stock_bars GROUP BY symbol"
        df_count = pd.read_sql_query(query_count, conn)
        
        if df_count.empty:
             print("No data to count.")
        else:
            print(df_count.to_string())


    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure the database connection is closed, even if an error occurred
        if conn:
            conn.close()
            print("\n--- Database connection closed ---")

if __name__ == "__main__":
    view_database_contents()