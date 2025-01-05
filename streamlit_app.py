import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


# File paths 
DATASET_PATH = "dataset/filtered_date_traffic_activity_data.parquet"
FUTURE_FORECAST_PATH = "dataset/future_traffic_forecast.parquet"
MODEL_PATH = "model/vehicle_traffic_prediction_model.pkl"
SCALER_PATH = "model/vehicle_traffic_scaler_total.pkl" 

# Load dataset and preprocess
@st.cache_data
def load_dataset():
    df = pd.read_parquet(DATASET_PATH)

    # Ensure Timestamp is datetime and create additional columns
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    df['Hour'] = df['Timestamp'].dt.hour
    df['TimeOfDay_Morning'] = ((df['Hour'] >= 6) & (df['Hour'] < 12)).astype(int)
    df['TimeOfDay_Afternoon'] = ((df['Hour'] >= 12) & (df['Hour'] < 18)).astype(int)
    df['TimeOfDay_Evening'] = ((df['Hour'] >= 18) & (df['Hour'] < 24)).astype(int)
    df['TimeOfDay_Night'] = (df['Hour'] < 6).astype(int)

    return df


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_future_forecast():
    future_traffic = pd.read_parquet(FUTURE_FORECAST_PATH)
    return future_traffic


# Fetch historical data for the same date and time of day
def get_latest_historical_data(df, site, date, time_of_day):
    time_of_day_features = {
        'Morning': 'TimeOfDay_Morning',
        'Afternoon': 'TimeOfDay_Afternoon',
        'Evening': 'TimeOfDay_Evening',
        'Night': 'TimeOfDay_Night'
    }
    df['MonthDay'] = df['Timestamp'].dt.strftime('%m-%d')  # Extract month and day

    # Filter for the same site, date (ignoring year), and time of day
    historical_data = df[
        (df['Site'] == site) &
        (df['MonthDay'] == date.strftime('%m-%d')) &
        (df[time_of_day_features[time_of_day]] == 1)
    ]

    latest_date = historical_data['Date'].max()  # Find the latest date (year)
    if latest_date is None or historical_data.empty:
        return None

    filtered_data = historical_data[historical_data['Date'] == latest_date]
    return {
        "Date": str(latest_date),
        "Northbound": filtered_data["Northbound"].sum(),
        "Southbound": filtered_data["Southbound"].sum(),
        "Eastbound": filtered_data["Eastbound"].sum(),
        "Westbound": filtered_data["Westbound"].sum(),
        "Total": filtered_data["Total"].sum()
    }


# Prediction function
def predict_traffic(model, scaler, future_traffic, site, date, time_of_day):
    time_of_day_features = {
        'Morning': [1, 0, 0, 0],
        'Afternoon': [0, 1, 0, 0],
        'Evening': [0, 0, 1, 0],
        'Night': [0, 0, 0, 1]
    }

    future_traffic['ds'] = pd.to_datetime(future_traffic['ds']).dt.date
    site = site.strip().title()

    future_row = future_traffic[(future_traffic['Site'] == site) & (future_traffic['ds'] == date)]

    if future_row.empty:
        return None

    northbound = future_row['Northbound'].values[0]
    southbound = future_row['Southbound'].values[0]
    eastbound = future_row['Eastbound'].values[0]
    westbound = future_row['Westbound'].values[0]

    time_of_day_values = time_of_day_features[time_of_day]

    future_input = pd.DataFrame({
        'Northbound': [northbound],
        'Southbound': [southbound],
        'Eastbound': [eastbound],
        'Westbound': [westbound],
        'TimeOfDay_Morning': [time_of_day_values[0]],
        'TimeOfDay_Afternoon': [time_of_day_values[1]],
        'TimeOfDay_Evening': [time_of_day_values[2]],
        'TimeOfDay_Night': [time_of_day_values[3]],
        **{f'Site_{s}': [int(s == site)] for s in future_traffic['Site'].unique()}
    })

    # Ensure input columns match training columns
    for col in model.feature_names_in_:
        if col not in future_input.columns:
            future_input[col] = 0
    future_input = future_input[model.feature_names_in_]

    predicted_scaled = model.predict(future_input)[0]
    predicted_total = scaler.inverse_transform([[predicted_scaled]])[0][0]

    return {
        'total': predicted_total,
        'northbound': northbound,
        'southbound': southbound,
        'eastbound': eastbound,
        'westbound': westbound
    }


# Streamlit App
st.title("Vehicle Traffic Prediction")

# Load data and models
df = load_dataset()
future_traffic = load_future_forecast()
model, scaler = load_model_and_scaler()

# Sidebar inputs
st.sidebar.header("Prediction Results")
site = st.sidebar.selectbox("Select Site", df["Site"].unique())
date = st.sidebar.date_input("Select Date")
time_of_day = st.sidebar.selectbox("Select Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

if st.sidebar.button("Predict"):
    # Get historical data
    historical_data = get_latest_historical_data(df, site, date, time_of_day)

    # Make prediction
    prediction = predict_traffic(model, scaler, future_traffic, site, date, time_of_day)

    # Use columns for side-by-side display
    col1, col2 = st.columns(2)

    if prediction:
        with col1:
            #st.subheader("Prediction Results")
            prediction_data = pd.DataFrame({
                "Metric": ["Total Traffic", "Northbound", "Southbound", "Eastbound", "Westbound"],
                "Value": [
                    int(round(prediction['total'])),
                    int(round(prediction['northbound'])),
                    int(round(prediction['southbound'])),
                    int(round(prediction['eastbound'])),
                    int(round(prediction['westbound']))
                ]
            })
            st.table(prediction_data)

    if historical_data:
        with col2:
            st.subheader("Latest Historical Data")
            historical_table = pd.DataFrame({
                "Metric": ["Date", "Northbound", "Southbound", "Eastbound", "Westbound", "Total"],
                "Value": [
                    historical_data["Date"],
                    int(round(historical_data['Northbound'])),
                    int(round(historical_data['Southbound'])),
                    int(round(historical_data['Eastbound'])),
                    int(round(historical_data['Westbound'])),
                    int(round(historical_data['Total']))
                ]
            })
            st.table(historical_table)

    if not prediction:
        st.error("No forecasted data available for the selected input.")
    if not historical_data:
        st.warning("No historical data available for the selected input.")
