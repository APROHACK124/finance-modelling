# %%
from typing import Dict, List, Tuple

from python_scripts.LLM_analysis.preprocess_store_database import get_connection


# %%
import pandas as pd

def visualize_stock_bars():
    con = get_connection()

    SQL = '''
        SELECT * FROM stock_bars
        '''

    df = pd.read_sql_query(SQL, con)
    print('stock_bars: ')
    print(df.describe())

    print(df.groupby(['symbol']).describe())

visualize_stock_bars()



# %%

def look_for_holes_stock_bars(symbol: str):
    import pandas as pd
    from python_scripts.LLM_analysis.preprocess_store_database import get_connection
    
    con = get_connection()
    df = pd.read_sql_query(f'''SELECT * FROM stock_bars WHERE symbol = '{symbol}' ''', con)
    lst = []
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    timestamps = df[['timestamp']]

    sorted = timestamps.sort_values(by='timestamp')

    df['delta'] = sorted.diff()
    
    # print(difference.describe())
    
    threshold = pd.Timedelta(minutes=15)
    holes = df[df['delta'] > threshold]


    print(holes.describe())
    
    

look_for_holes_stock_bars('AAPL')



# %%
