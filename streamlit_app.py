import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Paths to necessary files
DATASET_PATH = "dataset/filtered_date_traffic_activity_data.parquet"
FUTURE_FORECAST_PATH = "dataset/future_traffic_forecast.parquet"
MODEL_PATH = "model/vehicle_traffic_prediction_model.pkl"
SCALER_PATH = "model/vehicle_traffic_scaler_total.pkl"

# Load the dataset
@st.cache_data
def load_dataset():
    df = pd.read_parquet(DATASET_PATH)
    return df

@st.cache_data
def load_future_forecast():
    future_traffic = pd.read_parquet(FUTURE_FORECAST_PATH)
    return future_traffic

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# Prepare historical data for display
def get_historical_data(df, site, date):
    # Ensure 'Date' column exists
    if 'Date' not in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date  # Extract the date part

    # Filter for the given site and date
    site_data = df[(df["Site"] == site) & (df["Date"] == pd.to_datetime(date).date())]

    if site_data.empty:
        return None

    return {
        "Schedule Period": str(date),
        "Northbound": site_data["Northbound"].sum(),
        "Southbound": site_data["Southbound"].sum(),
        "Eastbound": site_data["Eastbound"].sum(),
        "Westbound": site_data["Westbound"].sum(),
        "Total Historical Traffic": site_data["Total"].sum(),
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
st.sidebar.header("Input Parameters")
site = st.sidebar.selectbox("Select Site", df["Site"].unique())
date = st.sidebar.date_input("Select Date")
time_of_day = st.sidebar.selectbox("Select Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

if st.sidebar.button("Predict"):
    # Get historical data
    historical_data = get_historical_data(df, site, date)

    # Make prediction
    prediction = predict_traffic(model, scaler, future_traffic, site, date, time_of_day)

    if prediction:
        # Display prediction results
        st.subheader("Prediction Results")
        prediction_data = pd.DataFrame({
            "Metric": ["Total Traffic", "Northbound", "Southbound", "Eastbound", "Westbound"],
            "Value": [
                f"{prediction['total']:.2f}",
                f"{prediction['northbound']:.2f}",
                f"{prediction['southbound']:.2f}",
                f"{prediction['eastbound']:.2f}",
                f"{prediction['westbound']:.2f}"
            ]
        })
        st.table(prediction_data)

    if historical_data:
        # Display historical data
        st.subheader("Latest Historical Data")
        historical_table = pd.DataFrame({
            "Metric": ["Northbound", "Southbound", "Eastbound", "Westbound", "Total Historical Traffic"],
            "Value": [
                f"{historical_data['Northbound']:.2f}",
                f"{historical_data['Southbound']:.2f}",
                f"{historical_data['Eastbound']:.2f}",
                f"{historical_data['Westbound']:.2f}",
                f"{historical_data['Total Historical Traffic']:.2f}"
            ]
        })
        st.table(historical_table)
    else:
        st.warning("No historical data available for the selected input.")

    if not prediction:
        st.error("No forecasted data available for the selected input.")
