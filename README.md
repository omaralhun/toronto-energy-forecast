# Toronto Energy Demand Forecasting

**Predicting Ontario electricity demand and price demand from weather patterns using machine learning.**

---

## Project Overview

This project builds a machine learning system that predicts hourly electricity demand for the province of Ontario, Canada based on weather data. The system analyzes the relationship between climate conditions and grid demand to forecast future energy needs and price demand.

**Why this matters:**
- Utilities use demand forecasts to plan generation capacity
- Accurate predictions help reduce waste and costs
- Understanding weather impact on energy helps with grid stability

---

## Dataset

### Data Sources

| Dataset | Source | Time Span | Variables |
|---------|--------|-----------|-----------|
| **Weather** | WeatherStats Toronto | 1937-2025 (daily) | Temperature, precipitation, HDD/CDD |
| **Electricity Demand** | Ontario IESO | 2002-2026 (hourly) | Market Demand, Ontario Demand (MW) |

### Data Details

- **Weather:** 32,036 daily records (32 MB)
- **Demand:** 208,872 hourly records (5 MB)
- **Final merged dataset:** 43,680 hourly records (2020-2024)

### Key Variables
- Temperature (max, min, average)
- Heating Degree Days (HDD)
- Cooling Degree Days (CDD)
- Precipitation
- Hour, day of week, month, season

---

## Methods

### Models Used

| Model | Purpose | Framework |
|-------|---------|-----------|
| **Prophet** | Climate trend analysis & seasonality | Facebook/Meta |
| **LightGBM** | Hourly demand forecasting | Gradient Boosting |

### Feature Engineering

- **Lag features:** Previous 1, 2, 3, 6, 12, 24, 48, 168 hours
- **Rolling statistics:** 6h, 24h, 168h mean and std
- **Cyclical encoding:** Sin/cos transforms for hour, day, month
- **Interaction features:** Temperature × hour

### Validation

- Time-based train/validation split (80/20)
- Metrics: RMSE, MAE, R², MAPE

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| **RMSE** | 164.66 MW |
| **MAE** | 123.60 MW |
| **R²** | 0.9948 |
| **MAPE** | 0.77% |

The model explains **99.5%** of variance in electricity demand.

### Key Findings

1. **Peak demand occurs in winter** (January-February) due to heating needs
2. **Hour of day** is the strongest predictor — peak at 6 PM, lowest at 4 AM
3. Temperature has a slight negative correlation with demand (colder = more heating)
4. Weekdays show higher demand than weekends (commercial/industrial activity)

### Feature Importance (Top 5)

1. Lag_1 (previous hour's demand)
2. Lag_24 (same hour yesterday)
3. Hour_cos (time of day)
4. Lag_2
5. Rolling mean (6 hours)

---

## Repository Structure

```
toronto-energy-forecast/
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── data/
│   └── processed/          # Cleaned & merged datasets
│       ├── merged_data.parquet
│       └── features.parquet
├── models/
│   ├── grid_demand_model.txt    # Trained LightGBM model
│   ├── climate_forecast.parquet # Prophet forecasts
│   └── climate_anomalies.parquet
├── src/
│   ├── ingestion/          # Data loading scripts
│   │   ├── load_weather.py
│   │   ├── load_ieso.py
│   │   └── prepare_data.py
│   ├── features/
│   │   └── engineer_features.py
│   └── models/
│       ├── climate_model.py
│       └── grid_model.py
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md             # This file
```

---

## How to Run

### Prerequisites

- Python 3.9+
- macOS, Linux, or Windows (via WSL)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/toronto-energy-forecast.git
cd toronto-energy-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser to `http://localhost:8501`

---

## Dashboard Features

### Overview Page
- Key demand metrics (average, peak, minimum)
- Historical demand trends
- Temperature vs. demand scatter plot
- Key findings from analysis

### Climate Page
- Temperature trend analysis (Prophet)
- Yearly seasonality patterns
- Anomaly detection

### Forecast Page
- Interactive demand predictions
- Input: hour, month, temperature
- Output: predicted demand (MW) + estimated cost ($)
- TOU rate calculation (Ontario 2026 rates)

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Data Processing | Polars, Pandas |
| Machine Learning | LightGBM, Prophet |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Environment | Python venv |

---

## Future Improvements

- Add hourly weather data for more accurate predictions
- Include electricity price data
- Add carbon emissions estimation

---

## License

MIT License

---

## Author

Insert name here

---

*Last updated: February 2026*
