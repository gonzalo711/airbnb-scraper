import toml
from st_files_connection import FilesConnection
from urllib.parse import urlparse, parse_qs
import streamlit as st
import pandas as pd
import gcsfs
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import numpy as np
from streamlit_option_menu import option_menu

# Ensure these libraries are in your requirements.txt

# Load GCP credentials directly from Streamlit's secrets
gcp_credentials = {
    "type": st.secrets["gcp_service_account"]["type"],
    "project_id": st.secrets["gcp_service_account"]["project_id"],
    "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
    "private_key": st.secrets["gcp_service_account"]["private_key"],
    "client_email": st.secrets["gcp_service_account"]["client_email"],
    "client_id": st.secrets["gcp_service_account"]["client_id"],
    "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
    "token_uri": st.secrets["gcp_service_account"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
}

# Use gcsfs to interact with GCS
fs = gcsfs.GCSFileSystem(token=gcp_credentials)

# Define your bucket and files (ensure the file paths are correct)
bucket_name = 'us-central1-airbnbcomposer-b06b3309-bucket'
file_paths = ['data/airbnb_final_listings_2024_4.csv',
              'data/airbnb_final_listings_2024_5.csv',
              'data/airbnb_final_listings_2024_6.csv',
              'data/airbnb_final_listings_2024_7.csv']

# Function to load data
def load_and_merge_csv(bucket_name, file_paths):
    all_data = pd.DataFrame()
    for file_path in file_paths:
        with fs.open(f'{bucket_name}/{file_path}') as f:
            df = pd.read_csv(f)
            all_data = pd.concat([all_data, df], ignore_index=True)
    all_data.to_csv('final.csv', index=False)
    return all_data

#Function to clean the data
def clean_transform_data(df):
    df['Price'] = df['Price'].str.extract(r'€ (\d+,\d+|\d+)')[0].str.replace(',', '').astype(float)
    
    # Extract Check_in and Check_out dates from the URL column
    df['Check_in'] = df['URL'].apply(lambda x: pd.to_datetime(parse_qs(urlparse(x).query).get('check_in', [None])[0]))
    df['Check_out'] = df['URL'].apply(lambda x: pd.to_datetime(parse_qs(urlparse(x).query).get('check_out', [None])[0]))
    
    # Calculate the number of nights
    df['number_nights'] = (df['Check_out'] - df['Check_in']).dt.days
    
    # Calculate Price per night; handle NaN nights
    df['Price_per_night'] = df.apply(lambda row: row['Price'] / row['number_nights'] if row['number_nights'] > 0 else None, axis=1)
    
    # Extract day of week for Check-in and Check-out
    df['Check_in_day'] = df['Check_in'].dt.dayofweek
    df['Check_out_day'] = df['Check_out'].dt.dayofweek
    
    # Determine if the period is a Weekend or Weekday stay
    df['Period'] = df.apply(lambda row: 'Weekend' if 5 <= row['Check_in_day'] <= 6 or 5 <= row['Check_out_day'] <= 6 else 'Weekday', axis=1)
    df['Interval'] = df.apply(lambda row: f"{row['Check_in'].strftime('%Y-%m-%d')} to {row['Check_out'].strftime('%Y-%m-%d')}", axis=1)
    return df

# Load, clean, and transform data
bucket_name = 'us-central1-airbnbcomposer-b06b3309-bucket/data'
file_paths = ['airbnb_final_listings_2024_4.csv', 'airbnb_final_listings_2024_5.csv', 'airbnb_final_listings_2024_6.csv','airbnb_final_listings_2024_7.csv']
data = load_and_merge_csv(bucket_name, file_paths)
data = clean_transform_data(data)


def calculate_percentage_difference(livin_paris_data, competitors_data):
    # Ensure both dataframes have the same structure
    combined = livin_paris_data.merge(competitors_data, on=['Bedrooms', 'Interval'], suffixes=('_livin', '_comp'))
    combined['Percentage Difference'] = ((combined['Price_per_night_livin'] - combined['Price_per_night_comp']) / combined['Price_per_night_comp']) * 100
    percentage_difference_pivot = combined.pivot("Bedrooms", "Interval", "Percentage Difference")
    return percentage_difference_pivot


# Streamlit UI for interactive visualization
st.title('🏡 Airbnb competitor pricing Analysis')


with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Settings'],
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    selected
    
month_selection = st.selectbox('Select Month', data['Check_in'].dt.month_name().unique())
bedroom_selection = st.selectbox('Select Number of Bedrooms', sorted(data['Bedrooms'].unique()))

filtered_data = data[(data['Check_in'].dt.month_name() == month_selection) & (data['Bedrooms'] == bedroom_selection)]

livin_paris_count = filtered_data[filtered_data['Livinparis'] == 'Yes'].shape[0]
competitors_count = filtered_data[filtered_data['Competitor'] == 'Yes'].shape[0]

total_count = filtered_data.shape[0]

percentage_of_total_livin_paris = (livin_paris_count / total_count * 100) if total_count else 0
percentage_of_total_competitors = (competitors_count / total_count * 100) if total_count else 0

delta_livin_paris = "{:.2f}%".format(percentage_of_total_livin_paris)
delta_competitors = "{:.2f}%".format(percentage_of_total_competitors)

st.metric(label="LivinParis Apartments", value=livin_paris_count, delta=delta_livin_paris, delta_color="normal",delta_color="off")
st.metric(label="Competitor Apartments", value=competitors_count, delta=delta_competitors, delta_color="normal",delta_color="off")


st.download_button(
    label="Download data as CSV",
    data=data.to_csv().encode('utf-8'),
    file_name='consolidated_data.csv',
    mime='text/csv',
)

# Filter data

# Pivot table and heatmap visualization
pivot_table = filtered_data.pivot_table(values='Price_per_night', index='Bedrooms', columns='Interval', aggfunc='mean').fillna(0)
plt.figure(figsize=(15, 8))
ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Average Price'})

# Decorate the plot
ax.set_title('Average Price Per Night Calendar View')
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Date')

# Improve the x-axis labels to show weekdays/weekends
# You can customize DateFormatter based on how you want to show the dates
date_format = DateFormatter("%b %d\n%A")
ax.xaxis.set_major_formatter(date_format)
plt.xticks(rotation=45)
