"""
Load Toronto daily weather data from WeatherStats CSV.
"""
import polars as pl
from pathlib import Path


def load_weather_data(data_dir: str = None) -> pl.DataFrame:
    """Load and clean Toronto weather data.
    
    Args:
        data_dir: Path to data/raw folder. Defaults to project data/raw.
    
    Returns:
        Polars DataFrame with weather data
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    else:
        data_dir = Path(data_dir)
    
    csv_path = data_dir / "weatherstats_toronto_daily.csv"
    
    # Read CSV
    df = pl.read_csv(csv_path)
    
    # Select relevant columns
    cols_to_keep = [
        "date",
        "max_temperature",
        "min_temperature", 
        "avg_temperature",
        "precipitation",
        "avg_wind_speed",
        "heatdegdays",
        "cooldegdays",
    ]
    
    # Filter to only columns that exist
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df = df.select(cols_available)
    
    # Parse date
    df = df.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )
    
    # Sort by date
    df = df.sort("date")
    
    return df


def get_weather_summary(df: pl.DataFrame) -> dict:
    """Get summary statistics for weather data."""
    return {
        "date_range": (df["date"].min(), df["date"].max()),
        "total_days": df.height,
        "missing_values": df.null_count().to_dict(),
        "temp_stats": {
            "max_temp": df["max_temperature"].describe(),
            "min_temp": df["min_temperature"].describe(),
        }
    }


if __name__ == "__main__":
    df = load_weather_data()
    print(f"Loaded {df.height} days of weather data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(df.head())
