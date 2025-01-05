 
# Vehicle Traffic Prediction App

This is a **Streamlit-based application** for predicting vehicle traffic and viewing historical traffic data based on user-selected parameters such as site, date, and time of day. The app uses machine learning models and preprocessed datasets for predictions and displays results in a user-friendly interface.

---

## Features

- **Vehicle Traffic Prediction**: Predict total traffic and directional traffic (Northbound, Southbound, Eastbound, Westbound) for a given site, date, and time of day.
- **Latest Historical Data**: Retrieve the most recent historical traffic data matching the selected parameters.
- **Time of Day Filtering**: Filter both predictions and historical data based on the selected time of day (Morning, Afternoon, Evening, or Night).
- **Side-by-Side Display**: Display prediction results and historical data side by side for easy comparison.
- **Rounded Integer Values**: All numeric results are displayed as integers for better readability.

---

## Folder Structure

```
.
├── .devcontainer/                   # Optional: Configuration for VS Code remote container
├── .github/                         # Optional: GitHub-specific configurations
├── dataset/                         # Folder for dataset files
│   ├── filtered_date_traffic_activity_data.parquet
│   ├── future_traffic_forecast.parquet
├── model/                           # Folder for model and scaler files
│   ├── vehicle_traffic_prediction_model.pkl
│   ├── vehicle_traffic_scaler_total.pkl
├── .gitignore                       # Ignored files for Git
├── LICENSE                          # License for the project
├── README.md                        # Project documentation
├── requirements.txt                 # Python package dependencies
├── streamlit_app.py                 # Main Streamlit application
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

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/predict-vehicle-traffic.git
   cd predict-vehicle-traffic
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are in the correct directories:
   - **Datasets** (`.parquet` files) in the `dataset/` folder.
   - **Model files** (`.pkl` files) in the `model/` folder.

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

- Ensure the dataset files (`.parquet`) are up-to-date and aligned with the trained model.
- The `Time of Day` filtering is critical for accurate predictions and historical data retrieval.

---

## Future Improvements

- Add additional parameters for filtering (e.g., weather conditions, day of the week).
- Enhance the visualization with charts or graphs.
- Integrate real-time data feeds for predictions.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

### Updated File Paths in `streamlit_app.py`

Update the file paths in `streamlit_app.py` to reflect the new folder structure:

```python
# Updated file paths
DATASET_PATH = "dataset/filtered_date_traffic_activity_data.parquet"
FUTURE_FORECAST_PATH = "dataset/future_traffic_forecast.parquet"
MODEL_PATH = "model/vehicle_traffic_prediction_model.pkl"
SCALER_PATH = "model/vehicle_traffic_scaler_total.pkl"
``` 
