import toml
from st_files_connection import FilesConnection
from urllib.parse import urlparse, parse_qs
import streamlit as st
import pandas as pd
import gcsfs
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import numpy as np
import plotly.figure_factory as ff
import calplot
import matplotlib.dates as mdates
from pandas.tseries.offsets import MonthBegin, MonthEnd
import july
from july.utils import date_range
from calendar import month_abbr
from datetime import datetime
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


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
       
    # Convert 'Check_in' and 'Check_out' from strings to datetime objects
    df['Check_in'] = pd.to_datetime(df['Check_in'])
    df['Check_out'] = pd.to_datetime(df['Check_out'])
    
    df.dropna(subset=['Check_in', 'Check_out'], inplace=True)
    
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
    
    #Check reviews and rating
    
    df[['Rating', 'Number_of_reviews']] = df['Review_rating'].str.extract(r'(\d+\.\d+|New)\s*(\(\d+\)|New)?')
    
    # Clean up the 'Number_of_reviews' by removing parentheses
    df['Number_of_reviews'] = df['Number_of_reviews'].str.replace(r'[()]', '', regex=True)
    
    # Return the transformed DataFrame
    return df

# Load, clean, and transform data
bucket_name = 'us-central1-airbnbcomposer-b06b3309-bucket/data'
file_paths = ['airbnb_final_listings_2024_4_final.csv', 'airbnb_final_listings_2024_5_final.csv', 'airbnb_final_listings_2024_6_final.csv']
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
    
def make_clickable(url):
    # This function returns an HTML anchor tag with the URL as the hyperlink target
    return f'<a target="_blank" href="{url}">{url}</a>'
    
def create_calendar_heatmap(df, year, month):
    # Filter for the year and month
    df_month = df[df['Check_in'].dt.year == year]
    df_month = df_month[df_month['Check_in'].dt.month == month]
    
    # Group by day and calculate mean price
    df_daily = df_month.set_index('Check_in').resample('D')['Price_per_night'].mean()
    
    # Use calplot to plot the data
    calplot.calplot(df_daily, how=None, cmap='YlGn', fillcolor='whitesmoke', linewidth=1, linecolor=None,
                    daylabels='MTWTFSS', dayticks=True, dropzero=True, yearlabels=False,
                    edgecolor='gray', figsize=(16, 3), suptitle=f'Average Price Per Night for {year}-{month:02d}')
    plt.show()
def aggrid_interactive_table(df):
            # Convert the 'URL' column to clickable links using HTML anchor tags
    df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

            # Create an interactive grid with AgGrid
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()  # Enable pagination
    gb.configure_side_bar()  # Enable a side bar
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    grid_options = gb.build()

            # Display the grid
    AgGrid(df, gridOptions=grid_options, enable_enterprise_modules=True, allow_unsafe_jscode=True)

# Streamlit UI for interactive visualization

st.set_page_config(layout="wide")

col1, col2= st.columns([0.8, 0.2])
with col1:
    st.title('🏡 Airbnb competitor pricing Analysis')
    
    st.markdown("<hr style='border-top: 2px solid red; margin-top: 20px; margin-bottom: 20px'/>", unsafe_allow_html=True)
with col2:
    st.image("pictures/linvinparis.png")

st.write("#")
tabs = st.tabs(['Pricing benchmark 🔍','Review scraped results for an interval 🗓️', 'Explore the dataset 📚'])

with tabs[0]:
    st.write("#")
    st.subheader("Please select the month and number of bedrooms")
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
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33])
    with col1:
        st.metric(label=f"Total apartments scraped for {month_selection}", value=total_count)
    with col2:
        st.metric(label=f"LivinParis apartments scraped in {month_selection}", value=livin_paris_count, delta=delta_livin_paris,delta_color="off")
    with col3:
        st.metric(label=f"Competitor Apartments scraped in {month_selection}", value=competitors_count, delta=delta_competitors, delta_color="off")

    st.divider()

        
    st.subheader("Pricing deep dive")
    st.write("#")

    # Plotly Heatmap for Average Price Per Night
    filtered_data_competitor= filtered_data[filtered_data['Livinparis'] == 'Yes']
    filtered_data_livinparis= filtered_data[filtered_data['Competitor'] == 'Yes']
    
    pivot_avg_price_competitor = filtered_data_competitor.pivot_table(index='Bedrooms', columns='Interval', values='Price_per_night', aggfunc='mean').fillna(0)
    pivot_avg_price_livinparis = filtered_data_livinparis.pivot_table(index='Bedrooms', columns='Interval', values='Price_per_night', aggfunc='mean').fillna(0)
    annotation_text_competitor = np.vectorize(lambda x: "€{:.0f}".format(x))(pivot_avg_price_competitor.values)
    annotation_text_livinparis = np.vectorize(lambda x: "€{:.0f}".format(x))(pivot_avg_price_livinparis.values)



    fig_avg_price_competitor = ff.create_annotated_heatmap(
        z=pivot_avg_price_competitor.values,
        x=pivot_avg_price_competitor.columns.tolist(),
        y=pivot_avg_price_competitor.index.tolist(),
        annotation_text=annotation_text_competitor,
        colorscale='blues',
        showscale=True
    )
    
    fig_avg_price_livinparis = ff.create_annotated_heatmap(
    z=pivot_avg_price_livinparis.values,
    x=pivot_avg_price_livinparis.columns.tolist(),
    y=pivot_avg_price_livinparis.index.tolist(),
    annotation_text=annotation_text_livinparis,
    colorscale='blues',
    showscale=True
    )
    
    col1, col2= st.columns([0.1, 0.9])
    with col1:
        st.image("pictures/airbnb.png")
    with col2:
        st.divider()
    fig_avg_price_competitor.update_layout(title_text='Competitor Average Price Per Night 💵', xaxis_title="Interval", yaxis_title="Bedrooms")
    st.plotly_chart(fig_avg_price_competitor, use_container_width=True)
    
    
    col1, col2= st.columns([0.1, 0.9])
    with col1:
        st.image("pictures/linvinparis.png")
    with col2:
        st.divider()

    fig_avg_price_livinparis.update_layout(title_text='Livinparis Average Price Per Night 💵', xaxis_title="Interval", yaxis_title="Bedrooms")
    
    
    st.plotly_chart(fig_avg_price_livinparis, use_container_width=True)
    
    st.caption("Note: Negative percentages 🟥 indicate intervals where LivinParis' prices are higher on average compared to competitors.")
    
        
    
    pivot_percentage_diff = calculate_percentage_difference(filtered_livin_paris, filtered_competitors)

    transposed_pivot = pivot_percentage_diff.T
    
    annotation_text = np.vectorize(lambda x: f"{int(round(x))}%")(transposed_pivot.values)

    fig_percentage_diff = ff.create_annotated_heatmap(
    z=transposed_pivot.values,  # Note that we are using the transposed pivot now
    x=transposed_pivot.columns.tolist(),  # These are now the intervals
    y=transposed_pivot.index.tolist(),  # This is now 'Bedrooms'
    annotation_text=np.around(transposed_pivot.values).astype(str),
    colorscale='RdYlGn',
    showscale=True
    )
    
    fig_percentage_diff.update_traces(zmin=-60, zmax=60)

    fig_percentage_diff.update_layout(
        title_text='LivinParis vs Competitors 🥊',
        xaxis_title="Interval",
        yaxis_title=""
    )
    
    fig_percentage_diff.update_yaxes(title_text="Percentage Difference", title_standoff=25, autorange=True)

    st.plotly_chart(fig_percentage_diff, use_container_width=True)
    
    
    
    st.divider()
    
    col1, col2, col3 = st.columns([0.2, 0.8, 0.2])
    with col1:
        st.download_button(
            label="Download data as CSV",
            data=data.to_csv().encode('utf-8'),
            file_name='consolidated_data.csv',
            mime='text/csv',type="primary"
        )
    with col2:
        st.write("#")
    
    with col3:
        st.link_button("Go to Airbnb", "https://www.airbnb.com/")
    
    
with tabs[1]:
    
    filtered_data_month = filtered_data[filtered_data['Check_in'].dt.month_name() == month_selection]
    intervals_in_month = filtered_data_month['Interval'].unique()
    
    st.subheader("Pick an interval to check out the competitors 👀")
    interval_selection = st.selectbox('Select Interval', intervals_in_month)
    
    
    filtered_data_interval = filtered_data_month[filtered_data_month['Interval'] == interval_selection]
    columns_to_display = ['Title', 'Price_per_night','Rating', 'Number_of_reviews', 'URL']
    df_display = filtered_data_interval[columns_to_display].copy()
    
    
    fig = px.histogram(
        filtered_data_interval,
        x='Price_per_night',
        color='Competitor',
        color_discrete_map={'Yes': 'red', 'No': 'black'},
        barmode='overlay',
        nbins=6,  # Adjust the number of bins as needed
        range_x=[500, 3000],  # Adjust the range as needed
        title='Distribution of Average Price per Night'
    )

    fig.update_layout(
        xaxis_title='Average Price per Night',
        yaxis_title='Count',
        legend_title='Source',
        legend=dict(traceorder='normal', font=dict(size=12)),
    )

    # Show the figure in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        # Assuming df_display is your dataframe with the 'URL' column
        aggrid_interactive_table(df_display)
        
    with col2:
        competitors_interval_count = filtered_data_interval[filtered_data['Competitor'] == 'Yes'].shape[0]
        livinparis_interval_count = filtered_data_interval[filtered_data['Livinparis'] == 'Yes'].shape[0]
        st.metric(label="Number of competitors scraped", value=competitors_interval_count)
        st.metric(label="Number of LivinParis appartments", value=livinparis_interval_count)
        


with tabs[2]:
    # Selecting specific columns
    columns_to_display = ['Title', 'Price_per_night', 'Check_in', 'Check_out', 'URL']
    filtered_subset = filtered_data[columns_to_display]
    
    AgGrid(filtered_subset)
    # Displaying the subset DataFrame
    #st.dataframe(filtered_subset)
