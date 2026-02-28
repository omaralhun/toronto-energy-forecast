"""
Climate model using Prophet for weather trend analysis.
"""
import polars as pl
import pandas as pd
from prophet import Prophet
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def prepare_daily_weather(data_dir: str = None) -> pd.DataFrame:
    """Prepare daily weather data for Prophet."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    else:
        data_dir = Path(data_dir)
    
    csv_path = data_dir / "weatherstats_toronto_daily.csv"
    df = pl.read_csv(csv_path)
    
    # Select columns
    cols = ["date", "max_temperature", "min_temperature", "avg_temperature", 
            "precipitation", "heatdegdays", "cooldegdays"]
    df = df.select([c for c in cols if c in df.columns])
    
    # Parse date and filter
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    df = df.filter(pl.col("date").dt.year() >= 2020)
    df = df.sort("date")
    df = df.fill_null(strategy="forward")
    df = df.drop_nulls()
    
    # Convert to pandas for Prophet
    pdf = df.to_pandas()
    pdf = pdf.rename(columns={"date": "ds"})
    
    return pdf


def train_temperature_model(weather_df: pd.DataFrame) -> tuple:
    """Train Prophet model for average temperature."""
    # Prepare data
    df = weather_df[["ds", "avg_temperature"]].copy()
    df = df.rename(columns={"avg_temperature": "y"})
    
    # Train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    return model, forecast


def analyze_trends(model, forecast: pd.DataFrame) -> dict:
    """Analyze climate trends."""
    # Get last year's forecast vs actual
    recent = forecast[forecast["ds"] >= "2024-01-01"]
    
    # Trend analysis
    trend_slope = (forecast["trend"].iloc[-1] - forecast["trend"].iloc[0]) / len(forecast)
    
    # Yearly seasonality peaks
    yearly = forecast.groupby(forecast["ds"].dt.month)["yearly"].mean()
    
    return {
        "trend_slope": trend_slope,
        "yearly_peak_month": yearly.idxmax(),
        "yearly_low_month": yearly.idxmin(),
        "forecast_2025": forecast[forecast["ds"].dt.year == 2025][["ds", "yhat", "trend"]].head(30)
    }


def detect_anomalies(weather_df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """Detect temperature anomalies."""
    # Calculate rolling mean and std
    weather_df = weather_df.copy()
    weather_df["temp_rolling_mean"] = weather_df["avg_temperature"].rolling(30).mean()
    weather_df["temp_rolling_std"] = weather_df["avg_temperature"].rolling(30).std()
    
    # Flag anomalies
    weather_df["z_score"] = (weather_df["avg_temperature"] - weather_df["temp_rolling_mean"]) / weather_df["temp_rolling_std"]
    weather_df["is_anomaly"] = weather_df["z_score"].abs() > threshold
    
    anomalies = weather_df[weather_df["is_anomaly"]]
    
    return anomalies


def main():
    print("Loading daily weather data...")
    weather_df = prepare_daily_weather()
    print(f"Training data: {len(weather_df)} days")
    print(f"Date range: {weather_df['ds'].min()} to {weather_df['ds'].max()}")
    
    print("\nTraining Prophet model for temperature...")
    model, forecast = train_temperature_model(weather_df)
    
    print("\nAnalyzing trends...")
    trends = analyze_trends(model, forecast)
    
    print(f"\nTrend analysis:")
    print(f"  Trend slope: {trends['trend_slope']:.4f}°C per day")
    print(f"  Peak month: {trends['yearly_peak_month']} (warmest)")
    print(f"  Low month: {trends['yearly_low_month']} (coldest)")
    
    print("\nDetecting anomalies...")
    anomalies = detect_anomalies(weather_df)
    print(f"  Found {len(anomalies)} anomalous days")
    
    # Save model components
    output_dir = Path(__file__).parent.parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    
    # Save forecast
    forecast.to_parquet(output_dir / "climate_forecast.parquet", index=False)
    print(f"\nSaved forecast to: {output_dir / 'climate_forecast.parquet'}")
    
    # Save anomalies
    if len(anomalies) > 0:
        anomalies.to_parquet(output_dir / "climate_anomalies.parquet", index=False)
        print(f"Saved anomalies to: {output_dir / 'climate_anomalies.parquet'}")
    
    print("\n" + "=" * 50)
    print("SAMPLE FORECAST (next 10 days)")
    print("=" * 50)
    print(forecast[["ds", "yhat", "yearly", "trend"]].tail(10))


if __name__ == "__main__":
    main()
