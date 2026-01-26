# fetch_news_data.py
def main():
    import os
    import requests
    import sqlite3
    from dotenv import load_dotenv
    from datetime import datetime, timedelta

    # --- Configuration ---
    load_dotenv()

    # --- Database Path Setup ---
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    DB_FOLDER = os.path.join(SCRIPT_DIR, 'data')
    DB_FILE = os.path.join(DB_FOLDER, 'stock_data.db')

    # --- NewsAPI Configuration ---
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    TARGET_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BTC', 'Bitcoin']

    def setup_database_news():
        """Sets up the database table for storing news articles."""
        print("Setting up 'news_articles' table...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                url TEXT PRIMARY KEY,
                query_term TEXT NOT NULL,
                source_name TEXT,
                author TEXT,
                title TEXT,
                description TEXT,
                published_at DATETIME NOT NULL,
                content TEXT,
                fetch_date DATETIME NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("Database setup for news complete.")

    def store_articles_to_db(cursor, articles, query_term):
        """Stores a list of article dictionaries in the database."""
        sql = '''
            INSERT OR IGNORE INTO news_articles 
            (url, query_term, source_name, author, title, description, published_at, content, fetch_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        new_rows = 0
        for article in articles:
            # Skip articles with no URL, as it's our primary key
            if not article.get('url'):
                continue
                
            # The 'content' provided by the free tier is often truncated.
            data_tuple = (
                article.get('url'),
                query_term,
                article.get('source', {}).get('name'),
                article.get('author'),
                article.get('title'),
                article.get('description'),
                article.get('publishedAt'),
                article.get('content'),
                datetime.now().isoformat()
            )
            cursor.execute(sql, data_tuple)
            new_rows += cursor.rowcount
        return new_rows

    def fetch_and_store_news():
        """Fetches news from NewsAPI and stores it in the database."""
        if not NEWS_API_KEY:
            print("Error: NEWS_API_KEY not found in .env file. Skipping news fetch.")
            return

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        headers = {'X-Api-Key': NEWS_API_KEY}
        total_new_articles = 0
        api_calls_made = 0
        
        # --- 1. Fetch General Business News ---
        print("\nFetching general business news...")
        try:
            # Use the 'top-headlines' endpoint for general news
            url = "https://newsapi.org/v2/top-headlines"
            params = {'category': 'business', 'language': 'en', 'country': 'us'}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            api_calls_made += 1
            
            articles = response.json().get('articles', [])
            new_rows = store_articles_to_db(cursor, articles, 'general_business')
            total_new_articles += new_rows
            print(f"-> Stored {new_rows} new general business articles.")
            
        except Exception as e:
            print(f"  - Error fetching general business news: {e}")

        # --- 2. Fetch Stock-Specific News ---
        print("\nFetching stock-specific news...")
        # Look for articles from the last 3 days to keep it relevant
        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        for symbol in TARGET_STOCKS:
            print(f"  - Searching for '{symbol}'...")
            try:
                # Use the 'everything' endpoint for specific keyword searches
                url = "https://newsapi.org/v2/everything"
                params = {'q': symbol, 'language': 'en', 'from': from_date, 'sortBy': 'relevancy'}
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                api_calls_made += 1
                
                articles = response.json().get('articles', [])
                new_rows = store_articles_to_db(cursor, articles, symbol)
                total_new_articles += new_rows
                print(f"  -> Stored {new_rows} new articles for {symbol}.")

            except Exception as e:
                print(f"    - Error fetching news for {symbol}: {e}")
                
        conn.commit()
        conn.close()
        
        print("\n--- News Fetch Finished ---")
        print(f"Total API calls made: {api_calls_made} (Limit: 100/day)")
        print(f"Total new articles added to database: {total_new_articles}")


    setup_database_news()
    fetch_and_store_news()

if __name__ == "__main__":
    main()