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
import plotly.figure_factory as ff

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
file_paths = ['data/airbnb_final_listings_2024_4_final.csv',
              'data/airbnb_final_listings_2024_5_final.csv',
              'data/airbnb_final_listings_2024_6_final.csv']

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
    df['Price'] = df['Price'].astype(str)
    df['Price'] = df['Price'].str.extract(r'€ (\d+,\d+|\d+)')[0].str.replace(',', '').astype(float)
    
    # Drop rows with NaN values in 'Check_in' and 'Check_out' if present
    df.dropna(subset=['Check_in', 'Check_out'], inplace=True)
    
    # Convert 'Check_in' and 'Check_out' from strings to datetime objects
    df['Check_in'] = pd.to_datetime(df['Check_in'])
    df['Check_out'] = pd.to_datetime(df['Check_out'])
    
    # Correctly extract Check_in and Check_out dates from the URL column
    df['Check_in_url'] = df['URL'].apply(lambda x: pd.to_datetime(parse_qs(urlparse(x).query).get('check_in', [None])[0]))
    df['Check_out_url'] = df['URL'].apply(lambda x: pd.to_datetime(parse_qs(urlparse(x).query).get('check_out', [None])[0]))
    
    # Calculate the day of the week for Check_in and Check_out
    df['Check_in_day'] = df['Check_in_url'].dt.dayofweek
    df['Check_out_day'] = df['Check_out_url'].dt.dayofweek
    
    # Calculate the number of nights
    df['number_nights'] = (df['Check_out_url'] - df['Check_in_url']).dt.days
    
    # Calculate Price per night; handle NaN nights
    df['Price_per_night'] = df.apply(lambda row: row['Price'] / row['number_nights'] if row['number_nights'] > 0 else None, axis=1)
    
    # Determine if the period is a Weekend or Weekday stay
    df['Period'] = df.apply(lambda row: 'Weekend' if 5 <= row['Check_in_day'] <= 6 or 5 <= row['Check_out_day'] <= 6 else 'Weekday', axis=1)
    df['Interva_url'] = df.apply(lambda row: f"{row['Check_in_url'].strftime('%Y-%m-%d')} to {row['Check_out_url'].strftime('%Y-%m-%d')}", axis=1)
    df['Interval'] = df.apply(lambda row: f"{row['Check_in'].strftime('%Y-%m-%d')} to {row['Check_out'].strftime('%Y-%m-%d')}", axis=1)

    # Add the desired_interval column
    df['desired_interval'] = np.where(df['Interva_url'] == df['Interval'], 'Yes', 'No')
    
    # Return the transformed DataFrame
    return df

# Load, clean, and transform data
bucket_name = 'us-central1-airbnbcomposer-b06b3309-bucket/data'
file_paths = ['airbnb_final_listings_2024_4.csv', 'airbnb_final_listings_2024_5.csv', 'airbnb_final_listings_2024_6.csv','airbnb_final_listings_2024_7.csv']
data = load_and_merge_csv(bucket_name, file_paths)
data = clean_transform_data(data)


def calculate_percentage_difference(livin_paris_data, competitors_data):
    # Calculating average price per night for LivinParis and competitors
    avg_price_livin = livin_paris_data.groupby('Interval')['Price_per_night'].mean().reset_index()
    avg_price_comp = competitors_data.groupby('Interval')['Price_per_night'].mean().reset_index()
    
    # Merging the average prices on Interval
    combined = pd.merge(avg_price_livin, avg_price_comp, on='Interval', suffixes=('_livin', '_comp'))
    
    # Calculating the percentage difference
    combined['Percentage Difference'] = (combined['Price_per_night_livin'] - combined['Price_per_night_comp']) / combined['Price_per_night_comp'] * 100
    
    # Pivot table for visualization
    percentage_difference_pivot = combined.pivot_table(index='Interval', values='Percentage Difference', aggfunc='mean').fillna(0)
    
    return percentage_difference_pivot



# Streamlit UI for interactive visualization

st.set_page_config(layout="wide")

col1, col2= st.columns([0.8, 0.2])
with col1:
    st.header('🏡 Airbnb competitor pricing Analysis',divider="rainbow")
with col2:
    st.image("pictures/linvinparis.png")

st.subheader("Please select the month and number of bedrooms for the benchmark")
col1, col2= st.columns([0.5, 0.5])
with col1:
    month_selection = st.selectbox('Select Month 🗓️', data['Check_in'].dt.month_name().unique())
with col2:
    bedroom_selection = st.selectbox('Select Number of Bedrooms 🛏️', sorted(data['Bedrooms'].unique()))

st.divider()
filtered_data = data[(data['Check_in'].dt.month_name() == month_selection) & (data['Bedrooms'] == bedroom_selection) & (data['desired_interval'] == 'Yes')]

livin_paris_count = filtered_data[filtered_data['Livinparis'] == 'Yes'].shape[0]
competitors_count = filtered_data[filtered_data['Competitor'] == 'Yes'].shape[0]


filtered_livin_paris = filtered_data[filtered_data['Livinparis'] == 'Yes']
filtered_competitors = filtered_data[filtered_data['Competitor'] == 'Yes']

total_count = filtered_data.shape[0]

percentage_of_total_livin_paris = (livin_paris_count / total_count * 100) if total_count else 0
percentage_of_total_competitors = (competitors_count / total_count * 100) if total_count else 0

delta_livin_paris = "{:.2f}%".format(percentage_of_total_livin_paris)
delta_competitors = "{:.2f}%".format(percentage_of_total_competitors)

st.subheader("What is my data sample?")
col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
with col1:
    st.metric(label="Total apartments scraped", value=total_count)
with col2:
    st.metric(label="Amount of LivinParis apartments", value=livin_paris_count, delta=delta_livin_paris,delta_color="off")
with col3:
    st.metric(label="Amount of Competitor Apartments", value=competitors_count, delta=delta_competitors, delta_color="off")

st.divider()

st.download_button(
    label="Download data as CSV",
    data=data.to_csv().encode('utf-8'),
    file_name='consolidated_data.csv',
    mime='text/csv',type="primary"
)

# Plotly Heatmap for Average Price Per Night
pivot_avg_price = filtered_data.pivot_table(index='Bedrooms', columns='Interval', values='Price_per_night', aggfunc='mean').fillna(0)
annotation_text = np.vectorize(lambda x: "€{:.0f}".format(x))(pivot_avg_price.values)


st.divider()

fig_avg_price = ff.create_annotated_heatmap(
    z=pivot_avg_price.values,
    x=pivot_avg_price.columns.tolist(),
    y=pivot_avg_price.index.tolist(),
    annotation_text=annotation_text,
    colorscale='amp',
    showscale=True
)
fig_avg_price.update_layout(title_text='Average Price Per Night', xaxis_title="Bedrooms", yaxis_title="Interval")
st.plotly_chart(fig_avg_price, use_container_width=True)

# Plotly Heatmap for Percentage Difference between LivinParis and Competitors
# Ensure calculate_percentage_difference is correctly implemented
pivot_percentage_diff = calculate_percentage_difference(filtered_livin_paris, filtered_competitors)
fig_percentage_diff = ff.create_annotated_heatmap(
    z=pivot_percentage_diff.values,
    x=pivot_percentage_diff.columns.tolist(),
    y=pivot_percentage_diff.index.tolist(),
    annotation_text=np.around(pivot_percentage_diff.values, decimals=2).astype(str),
    colorscale='RdYlGn',
    showscale=True
)

fig_percentage_diff.update_layout(title_text='Percentage Difference between LivinParis and Competitors', xaxis_title="Interval", yaxis_title="Bedrooms")
st.plotly_chart(fig_percentage_diff, use_container_width=True)
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
