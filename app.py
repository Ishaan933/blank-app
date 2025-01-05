import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Paths to necessary files
MODEL_PATH = "model/vehicle_traffic_prediction_model.pkl"
SCALER_PATH = "model/vehicle_traffic_scaler_total.pkl"
DATASET_PATH = "data/filtered_date_traffic_activity_data.parquet"
FUTURE_FORECAST_PATH = "data/future_traffic_forecast.parquet"

# Load model, scaler, and datasets
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_data():
    df = pd.read_parquet(DATASET_PATH)
    future_traffic = pd.read_parquet(FUTURE_FORECAST_PATH)
    return df, future_traffic

model = load_model()
scaler = load_scaler()
df, future_traffic = load_data()

# Prediction function
def predict_traffic(site, date, time_of_day):
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
    })

    # Add site-specific columns
    for s in future_traffic['Site'].unique():
        future_input[f"Site_{s}"] = int(s == site)

    # Ensure input columns match model training
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

# Streamlit app
st.title("Vehicle Traffic Prediction")
st.sidebar.header("Input Parameters")

# Sidebar inputs
site = st.sidebar.selectbox("Select Site", df['Site'].unique())
date = st.sidebar.date_input("Select Date")
time_of_day = st.sidebar.selectbox("Select Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

if st.sidebar.button("Predict"):
    result = predict_traffic(site, date, time_of_day)
    if result:
        st.subheader("Prediction Results")
        st.write(f"**Total Traffic**: {result['total']:.2f}")
        st.write(f"**Northbound**: {result['northbound']:.2f}")
        st.write(f"**Southbound**: {result['southbound']:.2f}")
        st.write(f"**Eastbound**: {result['eastbound']:.2f}")
        st.write(f"**Westbound**: {result['westbound']:.2f}")
    else:
        st.error("No forecasted data available for the selected input.")

