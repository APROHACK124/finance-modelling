# visualize_status.py

def main():
    from python_scripts.LLM_analysis.preprocess_store_database import get_connection

    # preprocessing status:
    con = get_connection()
    print("missing posts:", con.execute("SELECT COUNT(*) FROM v_reddit_posts_enriched WHERE processed_at IS NULL;").fetchone()[0])
    print("missing comments:", con.execute("SELECT COUNT(*) FROM v_reddit_comments_enriched WHERE processed_at IS NULL;").fetchone()[0])
    print("missing news:", con.execute("SELECT COUNT(*) FROM v_news_articles_enriched WHERE processed_at IS NULL;").fetchone()[0])
    con.close()
    


if __name__ == "__main__":
    main()