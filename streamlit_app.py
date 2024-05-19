import toml
import streamlit as st
import pandas as pd
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

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Ensure these libraries are in your requirements.txt

# Define your local file paths
file_paths = [
    'airbnb_final_listings_2024_5_final_19-05-2024.csv',
    'airbnb_final_listings_2024_6_final_19-05-2024.csv',
    'airbnb_final_listings_2024_7_final_19-05-2024.csv',
    'airbnb_final_listings_2024_8_final_19-05-2024.csv',
    'airbnb_final_listings_2024_9_final_19-05-2024.csv',
    'airbnb_final_listings_2024_10_final_19-05-2024.csv',
]

# Function to load data
def load_and_merge_csv(file_paths):
    all_data = pd.DataFrame()
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        all_data = pd.concat([all_data, df], ignore_index=True)
    all_data.to_csv('final.csv', index=False)
    return all_data

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Convert strings to datetime if possible
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    # Remove timezone information for comparison
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    with st.container():
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                # Categorical filter
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                # Numeric range filter
                min_val = float(df[column].min())
                max_val = float(df[column].max())
                step = (max_val - min_val) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                # Date range filter
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    start_date, end_date = user_date_input
                    df = df[df[column].between(start_date, end_date)]
            else:
                # String match filter
                user_text_input = right.text_input(f"Substring in {column}")
                df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


# Function to clean the data
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
    
    # Check reviews and rating
    
    df[['Rating', 'Number_of_reviews']] = df['Review_rating'].str.extract(r'(\d+\.\d+|New)\s*(\(\d+\)|New)?')
    
    # Clean up the 'Number_of_reviews' by removing parentheses
    df['Number_of_reviews'] = df['Number_of_reviews'].str.replace(r'[()]', '', regex=True)
    
    # Return the transformed DataFrame
    return df

# Load, clean, and transform data
data = load_and_merge_csv(file_paths)
data = clean_transform_data(data)


def calculate_percentage_difference(livin_paris_data, competitors_data):
    # Calculating average price per night for LivinParis and competitors
    avg_price_livin = livin_paris_data.groupby('Interval')['Price_per_night'].mean().reset_index()
    avg_price_comp = competitors_data.groupby('Interval')['Price_per_night'].mean().reset_index()
    
    # Merging the average prices on Interval
    combined = pd.merge(avg_price_livin, avg_price_comp, on='Interval', suffixes=('_livin', '_comp'))
    
    # Calculating the percentage difference
    combined['Percentage Difference'] = (combined['Price_per_night_comp']-combined['Price_per_night_livin']) / combined['Price_per_night_comp'] * 100
    
    # Pivot table for visualization
    percentage_difference_pivot = combined.pivot_table(index='Interval', values='Percentage Difference', aggfunc='mean').fillna(0)
    
    return percentage_difference_pivot
    
def make_clickable(url):
    # This function returns an HTML anchor tag with the URL as the hyperlink target
    return f'<a target="_blank" href="{url}">{url}</a>'
    
def create_calendar_heatmap(df, year, month):
    # Filter for the year and month
    df_month = df[df['Check_in'].dt.year == year]
    df_month = df_month[df['Check_in'].dt.month == month]
    
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
    st.title('Airbnb pricing assistant 🤖🏡 ')
    
    st.markdown("<hr style='border-top: 2px solid red; margin-top: 20px; margin-bottom: 20px'/>", unsafe_allow_html=True)
with col2:
    st.image("pictures/linvinparis.png")

st.write("#")
tabs = st.tabs(['Pricing benchmark 🔍','Inspect an interval 🗓️', 'Explore dataset 📚'])

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

    st.subheader("Dataset overview")
    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
    with col1:
        st.metric(label=f"Total apartments scraped for {month_selection}", value=total_count)
    with col2:
        st.metric(label=f"LivinParis apartments scraped in {month_selection}", value=livin_paris_count, delta=delta_livin_paris,delta_color="off")
    with col3:
        st.metric(label=f"Competitor Apartments scraped in {month_selection}", value=competitors_count, delta=delta_competitors, delta

