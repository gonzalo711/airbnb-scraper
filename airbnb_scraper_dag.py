from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Assuming your script is named airbnb_scraper.py and is in the same directory as your DAG file
# Adjust the path accordingly if your file is in a different location
sys.path.append(os.path.dirname(__file__))

from airbnb_scraper import main as scrape_airbnb

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),  # Adjust start date as needed
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'airbnb_scraping',
    default_args=default_args,
    description='A simple DAG to scrape Airbnb data',
    schedule_interval=timedelta(days=1),  # Adjust the interval as needed
)

scrape_task = PythonOperator(
    task_id='scrape_airbnb',
    python_callable=scrape_airbnb,
    dag=dag,
)

