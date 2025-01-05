
# Vehicle Traffic Prediction App

This is a **Streamlit-based application** for predicting vehicle traffic and viewing historical traffic data based on user-selected parameters such as site, date, and time of day. The app uses machine learning models and preprocessed datasets for predictions and displays results in a user-friendly interface.

## Features

- **Vehicle Traffic Prediction**: Predict total traffic and directional traffic (Northbound, Southbound, Eastbound, Westbound) for a given site, date, and time of day.
- **Latest Historical Data**: Retrieve the most recent historical traffic data matching the selected parameters.
- **Time of Day Filtering**: Filter both predictions and historical data based on the selected time of day (Morning, Afternoon, Evening, or Night).
- **Side-by-Side Display**: Display prediction results and historical data side by side for easy comparison.
- **Rounded Integer Values**: All numeric results are displayed as integers for better readability.

---

## Project Structure

```
.
├── streamlit_app.py               # Main Streamlit application
├── filtered_date_traffic_activity_data.parquet # Dataset file with historical traffic data
├── future_traffic_forecast.parquet # Dataset file with future traffic forecasts
├── vehicle_traffic_prediction_model.pkl # Trained Random Forest Regressor model
├── vehicle_traffic_scaler_total.pkl # Scaler used for the model
├── README.md                      # Project documentation
```

---

## Requirements

- Python 3.7 or higher
- Required Python packages:
  - `streamlit`
  - `pandas`
  - `scikit-learn`
  - `joblib`

---

## Installation

1. Clone the repository or download the project files.
2. Install the required Python packages:
   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```
3. Ensure the following files are in the same directory as `streamlit_app.py`:
   - `filtered_date_traffic_activity_data.parquet`
   - `future_traffic_forecast.parquet`
   - `vehicle_traffic_prediction_model.pkl`
   - `vehicle_traffic_scaler_total.pkl`

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Use the sidebar to:
   - Select a **Site** from the dropdown.
   - Choose a **Date** for prediction.
   - Select the **Time of Day** (Morning, Afternoon, Evening, or Night).
3. Click the **Predict** button to view:
   - **Prediction Results**: Displays predicted total and directional traffic for the selected parameters.
   - **Latest Historical Data**: Displays the most recent historical data matching the selected parameters.

---

## How It Works

1. **Prediction Results**:
   - Uses a trained Random Forest Regressor model (`vehicle_traffic_prediction_model.pkl`) to predict traffic.
   - Input features are dynamically adjusted based on the selected site and time of day.

2. **Historical Data**:
   - Retrieves the most recent historical traffic data (excluding the year) for the selected site, date, and time of day from the dataset (`filtered_date_traffic_activity_data.parquet`).

3. **Time of Day Filtering**:
   - Filters both prediction and historical data based on the `Time of Day` (Morning, Afternoon, Evening, or Night).

---

## Example Screenshots

### Input Parameters
![Input Parameters](path/to/screenshot_input_parameters.png)

### Prediction and Historical Data
![Prediction and Historical Data](path/to/screenshot_prediction_and_historical_data.png)

---

## Notes

- Ensure that the dataset files (`.parquet`) are up-to-date and aligned with the trained model.
- The `Time of Day` filtering is critical for accurate predictions and historical data retrieval.
- For dynamic calculations like the total number of days, you can modify the code to fetch these values based on the date range.

---

## Future Improvements

- Add additional parameters for filtering (e.g., weather conditions, day of the week).
- Enhance the visualization with charts or graphs.
- Integrate real-time data feeds for predictions.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
 
