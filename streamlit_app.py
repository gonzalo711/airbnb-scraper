"""import streamlit as st
import pandas as pd
import gcsfs
import seaborn as sns
import matplotlib.pyplot as plt
import toml
from st_files_connection import FilesConnection"""

import streamlit as st
import pandas as pd
import gcsfs
import seaborn as sns
import matplotlib.pyplot as plt
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
              'data/airbnb_final_listings_2024_6.csv']

# Function to load data
def load_data():
    all_data = []
    for file_path in file_paths:
        full_path = f"{bucket_name}/{file_path}"
        with fs.open(full_path) as f:
            df = pd.read_csv(f)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

data = load_data()
# Continue with your data processing and visualization...





"""conn = st.connection('gcs', type=FilesConnection)
df = conn.read("us-central1-airbnbcomposer-b06b3309-bucket/myfile.csv", input_format="csv", ttl=600)

# Print results.
for row in df.itertuples():
    st.write(f"{row.Owner} has a :{row.Pet}:")"""


"""
# Function to load and merge data from multiple CSV files
def load_data(bucket_name, file_paths):
    #gcp_credentials = toml.load('.streamlit/secrets.toml')['gcp_service_account']
    gcp_credentials = st.secrets["gcp_service_account"]
    fs = gcsfs.GCSFileSystem(token=gcp_credentials)
    #fs = gcsfs.GCSFileSystem(project='airbnbscraper-417722', token=gcp_credentials)
    all_data = pd.DataFrame()
    
    # Ensure the file paths list is not empty and contains valid file paths
    if file_paths and all(isinstance(path, str) for path in file_paths):
        for file_path in file_paths:
            full_path = f"{bucket_name}/{file_path}"
            with fs.open(full_path) as f:
                df = pd.read_csv(f)
                all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data

# Function to clean and transform data
def clean_transform_data(df):
    df['Price'] = df['Price'].str.extract(r'€ (\d+,\d+)')[0].str.replace(',', '').astype(float)
    df['Check_in'] = pd.to_datetime(df['Check_in'])
    df['Check_out'] = pd.to_datetime(df['Check_out'])
    df['number_nights'] = (df['Check_out'] - df['Check_in']).dt.days
    df['Interval'] = df.apply(lambda row: f"{row['Check_in'].strftime('%Y-%m-%d')} to {row['Check_out'].strftime('%Y-%m-%d')}", axis=1)
    df['Check_in_day'] = df['Check_in'].dt.dayofweek
    df['Check_out_day'] = df['Check_out'].dt.dayofweek
    df['Price_per_night'] = df['Price'] / df['number_nights']
    df['Period'] = df.apply(lambda row: 'Weekend' if row['Check_in_day'] >= 5 or row['Check_out_day'] >= 5 else 'Weekday', axis=1)
    return df


st.write(st.secrets)


# Load data from GCS
bucket_name = 'us-central1-airbnbcomposer-b06b3309-bucket/data'
file_paths = ['airbnb_final_listings_2024_4.csv', 'airbnb_final_listings_2024_5.csv', 'airbnb_final_listings_2024_6.csv']
data = load_data(bucket_name, file_paths)
data = clean_transform_data(data)

# Streamlit UI
st.title('Airbnb Listings Dashboard')
month_selection = st.selectbox('Select Month', data['Check_in'].dt.month_name().unique())
bedroom_selection = st.selectbox('Select Number of Bedrooms', data['Bedrooms'].unique())

# Filter data based on selections
filtered_data = data[(data['Check_in'].dt.month_name() == month_selection) & (data['Bedrooms'] == bedroom_selection)]

# Pivot table for Livin Paris
livin_paris_data = filtered_data[filtered_data['Livinparis'] == 'Yes']
pivot_table_livin_paris = livin_paris_data.pivot_table(
    values='Price_per_night',
    index='Bedrooms',
    columns='Interval',
    aggfunc='mean'
).fillna(0)

# Pivot table for Competitors
competitors_data = filtered_data[filtered_data['Competitor'] == 'Yes']
pivot_table_competitors = competitors_data.pivot_table(
    values='Price_per_night',
    index='Bedrooms',
    columns='Interval',
    aggfunc='mean'
).fillna(0)

# Display heatmaps
st.header('Livin Paris Average Price Per Night')
fig, ax = plt.subplots()
sns.heatmap(pivot_table_livin_paris, annot=True, fmt=".2f", cmap='Blues', ax=ax)
st.pyplot(fig)

st.header('Competitors Average Price Per Night')
fig, ax = plt.subplots()
sns.heatmap(pivot_table_competitors, annot=True, fmt=".2f", cmap='Blues', ax=ax)
st.pyplot(fig)
"""
