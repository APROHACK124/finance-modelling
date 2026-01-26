# preprocess_data.py

def main():

    from python_scripts.LLM_analysis.preprocess_store_database import preprocess_all_tables

    preprocess_all_tables(sample_size_per_table=2000, batch_size=500)


if __name__ == "__main__":
    main()