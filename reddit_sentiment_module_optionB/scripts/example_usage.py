from sentiment.cli import run

if __name__ == "__main__":
    run(
        db_path="data.db",
        out_db_path=None,
        cfg_path=None,
        start_date=None,
        end_date=None,
        if_exists="replace",
        no_top_threads=False,
    )
