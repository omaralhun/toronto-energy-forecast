"""
Streamlit dashboard for Climate → Grid Demand Intelligence System.
"""
import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import lightgbm as lgb

# Configuration
st.set_page_config(
    page_title="Toronto Energy Intelligence",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"


@st.cache_data
def load_features_data():
    df = pl.read_parquet(DATA_DIR / "features.parquet")
    return df.to_pandas()


@st.cache_data
def load_climate_forecast():
    return pd.read_parquet(MODEL_DIR / "climate_forecast.parquet")


@st.cache_data
def load_anomalies():
    return pd.read_parquet(MODEL_DIR / "climate_anomalies.parquet")


def load_demand_model():
    model = lgb.Booster(model_file=str(MODEL_DIR / "grid_demand_model.txt"))
    return model


# ============== PAGE: OVERVIEW ==============
def page_overview(df):
    st.title("Toronto Energy Intelligence")
    st.markdown("**Climate → Grid Demand Forecasting System**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Hours", f"{len(df):,}")
    with col2:
        st.metric("Avg Demand", f"{df['ontario_demand'].mean():,.0f} MW")
    with col3:
        st.metric("Peak Demand", f"{df['ontario_demand'].max():,.0f} MW")
    with col4:
        st.metric("Min Demand", f"{df['ontario_demand'].min():,.0f} MW")
    
    st.divider()
    
    st.subheader("Ontario Electricity Demand (2020-2024)")
    
    daily = df.groupby("date").agg({
        "ontario_demand": "mean",
        "avg_temperature": "mean"
    }).reset_index()
    
    fig = px.line(daily, x="date", y="ontario_demand", 
                  title="Daily Average Demand (MW)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Demand (MW)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Temperature vs Demand")
    fig2 = px.scatter(df.sample(min(5000, len(df))), 
                      x="avg_temperature", y="ontario_demand",
                      color="hour",
                      title="Temperature vs Demand (colored by hour)",
                      opacity=0.5)
    fig2.update_layout(xaxis_title="Avg Temperature (C)", yaxis_title="Demand (MW)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Key Findings")
    
    avg_demand = df["ontario_demand"].mean()
    peak_demand = df["ontario_demand"].max()
    min_demand = df["ontario_demand"].min()
    
    hourly_avg = df.groupby("hour")["ontario_demand"].mean()
    peak_hour = hourly_avg.idxmax()
    peak_hour_demand = hourly_avg.max()
    
    low_hour = hourly_avg.idxmin()
    low_hour_demand = hourly_avg.min()
    
    monthly_avg = df.groupby("month")["ontario_demand"].mean()
    peak_month = monthly_avg.idxmax()
    low_month = monthly_avg.idxmin()
    
    temp_corr = df["avg_temperature"].corr(df["ontario_demand"])
    
    weekend_avg = df[df["is_weekend"] == True]["ontario_demand"].mean()
    weekday_avg = df[df["is_weekend"] == False]["ontario_demand"].mean()
    
    findings = """
    ### Demand Statistics
    - Average demand: %s MW
    - Peak demand: %s MW | Minimum demand: %s MW
    
    ### Time Patterns
    - Peak hour: %s:00 (%s MW average)
    - Lowest hour: %s:00 (%s MW average)
    - Peak month: Month %s (%s MW)
    - Lowest month: Month %s (%s MW)
    
    ### Weather Impact
    - Temperature correlation: %s (slightly negative = warmer days = slightly lower demand)
    - Weekend vs Weekday: Weekend avg = %s MW | Weekday avg = %s MW
    
    ### Key Insights
    1. Electricity demand is highest in winter (January-February) due to heating needs
    2. Hour of day is the strongest predictor - demand peaks at 6 PM, lowest at 4 AM
    3. Temperature has a slight negative correlation with demand (more heating in cold weather)
    4. Weekdays show higher demand than weekends (commercial/industrial activity)
    """ % (
        f"{avg_demand:,.0f}",
        f"{peak_demand:,.0f}",
        f"{min_demand:,.0f}",
        peak_hour,
        f"{peak_hour_demand:,.0f}",
        low_hour,
        f"{low_hour_demand:,.0f}",
        peak_month,
        f"{monthly_avg[peak_month]:,.0f}",
        low_month,
        f"{monthly_avg[low_month]:,.0f}",
        f"{temp_corr:.3f}",
        f"{weekend_avg:,.0f}",
        f"{weekday_avg:,.0f}"
    )
    
    st.markdown(findings)


# ============== PAGE: CLIMATE ==============
def page_climate():
    st.title("Climate Analysis")
    
    forecast = load_climate_forecast()
    anomalies = load_anomalies()
    
    st.subheader("Temperature Trend")
    fig = px.line(forecast, x="ds", y=["yhat", "trend"], 
                  title="Temperature Forecast & Trend")
    fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (C)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Yearly Seasonality")
    forecast["month"] = pd.to_datetime(forecast["ds"]).dt.month
    monthly = forecast.groupby("month")["yearly"].mean().reset_index()
    
    fig2 = px.bar(monthly, x="month", y="yearly", 
                  title="Average Seasonal Effect by Month",
                  labels={"yearly": "Temperature Effect (C)"})
    fig2.update_layout(xaxis_title="Month", yaxis_title="Temperature Effect")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Temperature Anomalies")
    st.write(f"Found **{len(anomalies)}** anomalous days (beyond 2 std dev)")
    
    if len(anomalies) > 0:
        anomalies_plot = anomalies.head(50).copy()
        anomalies_plot["date_str"] = anomalies_plot["ds"].dt.strftime("%Y-%m-%d")
        
        fig3 = px.scatter(anomalies_plot, x="date_str", y="avg_temperature",
                         size=anomalies_plot["z_score"].abs(), color="z_score",
                         title="Anomalous Days (size = deviation)")
        fig3.update_layout(xaxis_title="Date", yaxis_title="Temperature (C)")
        st.plotly_chart(fig3, use_container_width=True)


# ============== PAGE: FORECAST ==============
def page_forecast(df):
    st.title("Demand Forecasting")
    
    model = load_demand_model()
    
    st.sidebar.header("Make a Prediction")
    
    hours = list(range(1, 25))
    hour_labels = [f"{h-1}:00 {'AM' if h-1 < 12 else 'PM'}" if h-1 != 0 else "12:00 AM" if h == 1 else "12:00 PM" for h in hours]
    
    months = list(range(1, 13))
    month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    pred_hour = st.sidebar.selectbox("Hour of Day", hours, index=12, format_func=lambda x: hour_labels[x-1])
    pred_month = st.sidebar.selectbox("Month", months, index=6, format_func=lambda x: month_names[x-1])
    pred_temp = st.sidebar.slider("Avg Temperature (C)", -20, 35, 20)
    
    
    if st.sidebar.button("Predict Demand"):
        features = {
            "hour": pred_hour,
            "day_of_week": 3,
            "month": pred_month,
            "day_of_year": pred_month * 30,
            "year": 2024,
            "is_weekend": 0,
            "max_temperature": pred_temp + 5,
            "min_temperature": pred_temp - 5,
            "avg_temperature": pred_temp,
            "precipitation": 0,
            "heatdegdays": 0,
            "cooldegdays": 0,
            "lag_1": df["ontario_demand"].mean(),
            "lag_24": df["ontario_demand"].mean(),
            "rolling_mean_24": df["ontario_demand"].mean(),
        }
        
        for col in df.columns:
            if col not in features and col not in ["date", "ontario_demand", "market_demand", "season"]:
                features[col] = 0
        
        feature_cols = [c for c in df.columns if c not in ["date", "ontario_demand", "market_demand", "season"]]
        X_pred = pd.DataFrame([{c: features.get(c, 0) for c in feature_cols}])
        
        pred = model.predict(X_pred)[0]
        
        hour = pred_hour
        if 7 <= hour < 11:
            rate = 0.17
            rate_type = "Peak"
        elif 11 <= hour < 17:
            rate = 0.12
            rate_type = "Mid-Peak"
        elif 17 <= hour < 19:
            rate = 0.17
            rate_type = "Peak"
        else:
            rate = 0.08
            rate_type = "Off-Peak"
        
        mwh = pred
        kwh = mwh * 1000
        cost_per_hour = kwh * rate
        
        st.sidebar.success(f"Predicted Demand: **{pred:,.0f} MW**")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Estimated Cost")
        
        cost_col1, cost_col2 = st.sidebar.columns(2)
        with cost_col1:
            st.sidebar.metric("Rate Type", rate_type)
        with cost_col2:
            st.sidebar.metric("Rate", f"${rate:.2f}/kWh")
        
        st.sidebar.metric("Energy", f"{kwh:,.0f} kWh")
        st.sidebar.metric("Cost per Hour", f"${cost_per_hour:,.2f}")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Daily Estimate (24 hours)**")
        
        daily_cost_estimate = cost_per_hour * 24
        st.sidebar.metric("Estimated Daily Cost", f"${daily_cost_estimate:,.2f}")
    
    st.subheader("Demand Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hourly = df.groupby("hour")["ontario_demand"].mean().reset_index()
        fig = px.bar(hourly, x="hour", y="ontario_demand",
                    title="Average Demand by Hour")
        fig.update_layout(xaxis_title="Hour", yaxis_title="Demand (MW)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        monthly = df.groupby("month")["ontario_demand"].mean().reset_index()
        fig2 = px.bar(monthly, x="month", y="ontario_demand",
                     title="Average Demand by Month")
        fig2.update_layout(xaxis_title="Month", yaxis_title="Demand (MW)")
        st.plotly_chart(fig2, use_container_width=True)


# ============== MAIN ==============
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Climate", "Forecast"])
    
    if page in ["Overview", "Forecast"]:
        df = load_features_data()
    
    if page == "Overview":
        page_overview(df)
    elif page == "Climate":
        page_climate()
    elif page == "Forecast":
        page_forecast(df)


if __name__ == "__main__":
    main()
