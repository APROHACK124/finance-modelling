#!/bin/bash

# This script runs the stock data scraper
# It's important to change to the script's directory first
cd /home/aprohack/Desktop/all_folders/Investings_project/app || exit

# Execute the python script using the virtual environment's python


#/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python database_tier1.py >> /home/aprohack/Desktop/all_folders/Investings_project/app/cron.log 2>&1

#/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python calculate_indicators.py >> /home/aprohack/Desktop/all_folders/Investings_project/app/cron.log 2>&1

#/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python tier2_data_scrapper.py >> /home/aprohack/Desktop/all_folders/Investings_project/app/cron.log 2>&1

#/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python fetch_news_data.py >> /home/aprohack/Desktop/all_folders/Investings_project/app/cron.log 2>&1

#/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python fetch_economic_data.py >> /home/aprohack/Desktop/all_folders/Investings_project/app/cron.log 2>&1

#/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python fetch_reddit_data.py >> /home/aprohack/Desktop/all_folders/Investings_project/app/cron.log 2>&1

/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python preprocess_data.py

/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python database_tier1.py 

/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python calculate_indicators.py 

/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python tier2_data_scrapper.py 

/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python fetch_news_data.py 

/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python fetch_economic_data.py 

/home/aprohack/Desktop/all_folders/Investings_project/app/investenv/bin/python fetch_reddit_data.py




