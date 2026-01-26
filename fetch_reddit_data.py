# fetch_reddit_data.py

def main():
    import os
    import praw
    import sqlite3
    from datetime import datetime
    from dotenv import load_dotenv

    # --- Configuration ---
    # Load credentials from a .env file for security
    load_dotenv()

    # IMPORTANT: Fill these in with your credentials from Part 1
    # You can either put them directly here or in your .env file
    # For security, the .env file method is highly recommended.
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    USERNAME = os.getenv("REDDIT_USERNAME")
    PASSWORD = os.getenv("REDDIT_PASSWORD")
    # Example: "python:ai_trader_scraper:v1.0 (by /u/your_username)"
    USER_AGENT = os.getenv("REDDIT_USER_AGENT")

    # --- Database Path Setup ---
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    DB_FOLDER = os.path.join(SCRIPT_DIR, 'data')
    DB_FILE = os.path.join(DB_FOLDER, 'stock_data.db')

    # How many of the top posts from the last week to fetch per subreddit
    POST_LIMIT = 15
    # How many of the top comments to fetch from each post
    COMMENT_LIMIT = 7


    SUBREDDITS_TO_SCAN = [
        ['NvidiaStock', 5],
        ['nvidia', 5],
        ['microsoft', 5],
        ['amazon', 2],
        ['Bitcoin', 6],
        ['inflation', 6],
        ['news', 7],
        ['AskEconomics', 2],
        ['Economics', 3],
        ['CryptoCurrency', 7],
        ['apple', 5],
        ['technology', 3],
        ['google', 5],
        ['stocks', 8],
        ['wallstreetbets', 5],
        ['investing_discussion', 7],
        ['investing', 8],
        ['STOCKMARKETNEWS', 8],
        ['DividendCult', 2],
        ['economy', 3],
        ['StockMarket', 8],
        ['options', 8],
        ['SecurityAnalysis', 6],
        ['ValueInvesting', 8],
        ['macroeconomics', 6],
        ['bonds', 4],
        ['ETF',15],
        ['Bogleheads', 4],
        ['BitcoinMarkets', 5],
        ['CryptoMarkets', 2],


        


    ]


    def setup_database_reddit():
        """Sets up the database tables for storing Reddit data."""
        print("Setting up 'reddit_posts' and 'reddit_comments' tables...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Table for Reddit posts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reddit_posts (
                post_id TEXT PRIMARY KEY,
                subreddit TEXT NOT NULL,
                title TEXT,
                score INTEGER,
                num_comments INTEGER,
                url TEXT,
                body TEXT,
                created_utc REAL,
                fetch_date DATETIME NOT NULL
            )
        ''')

        # Table for Reddit comments
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reddit_comments (
                comment_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                score INTEGER,
                body TEXT,
                created_utc REAL,
                fetch_date DATETIME NOT NULL,
                FOREIGN KEY (post_id) REFERENCES reddit_posts (post_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database setup for Reddit complete.")

    def fetch_and_store_reddit_data():
        """
        Connects to the Reddit API, fetches top posts and their top comments
        from a list of subreddits, and stores them in the database.
        """
        print("--- Starting Reddit Data Fetch ---")

        if not all([CLIENT_ID, CLIENT_SECRET, USERNAME, PASSWORD, USER_AGENT]):
            print("Error: Missing Reddit API credentials. Please set them in your .env file.")
            return

        try:
            reddit = praw.Reddit(
                client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                username=USERNAME, password=PASSWORD, user_agent=USER_AGENT,
            )
            print(f"Successfully authenticated as user: {reddit.user.me()}")
        except Exception as e:
            print(f"Failed to authenticate with Reddit. Error: {e}")
            return

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        total_new_posts = 0
        total_new_comments = 0

        for sub_name, post_amount in SUBREDDITS_TO_SCAN:
            print(f"\nScanning subreddit: r/{sub_name}")
            try:
                subreddit = reddit.subreddit(sub_name)
                top_posts = subreddit.top(time_filter='week', limit=post_amount)

                for post in top_posts:
                    # Store post data
                    post_sql = "INSERT OR IGNORE INTO reddit_posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    post_data = (
                        post.id, sub_name, post.title, post.score, post.num_comments,
                        f"https://reddit.com{post.permalink}", post.selftext,
                        post.created_utc, datetime.now().isoformat()
                    )
                    cursor.execute(post_sql, post_data)
                    total_new_posts += cursor.rowcount

                    # Fetch and store comments for this post
                    post.comment_sort = 'top'
                    post.comments.replace_more(limit=0)
                    
                    comment_count = 0
                    for comment in post.comments:
                        if comment_count >= COMMENT_LIMIT or comment.stickied:
                            continue
                        
                        comment_sql = "INSERT OR IGNORE INTO reddit_comments VALUES (?, ?, ?, ?, ?, ?)"
                        comment_data = (
                            comment.id, post.id, comment.score, comment.body,
                            comment.created_utc, datetime.now().isoformat()
                        )
                        cursor.execute(comment_sql, comment_data)
                        total_new_comments += cursor.rowcount
                        comment_count += 1
                
                # Commit changes to the database after each subreddit is processed
                conn.commit()
                print(f"  -> Scanned {post_amount} posts.")

            except Exception as e:
                print(f"Could not process subreddit r/{sub_name}. Error: {e}")

        conn.close()
        print("\n--- Reddit Data Fetch Finished ---")
        print(f"Total new posts added to database: {total_new_posts}")
        print(f"Total new comments added to database: {total_new_comments}")


    setup_database_reddit()
    fetch_and_store_reddit_data()

if __name__ == '__main__':
    main()