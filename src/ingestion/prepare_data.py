"""
Clean and merge weather + IESO demand data.
"""
import polars as pl
from pathlib import Path


def load_and_clean_weather(data_dir: str = None) -> pl.DataFrame:
    """Load and clean weather data."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    else:
        data_dir = Path(data_dir)
    
    csv_path = data_dir / "weatherstats_toronto_daily.csv"
    
    # Load
    df = pl.read_csv(csv_path)
    
    # Select relevant columns
    cols_to_keep = [
        "date",
        "max_temperature",
        "min_temperature", 
        "avg_temperature",
        "precipitation",
        "heatdegdays",
        "cooldegdays",
    ]
    df = df.select([c for c in cols_to_keep if c in df.columns])
    
    # Parse date
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    
    # Filter to 2020-2024
    df = df.filter(
        (pl.col("date").dt.year() >= 2020) & 
        (pl.col("date").dt.year() <= 2024)
    )
    
    # Sort by date
    df = df.sort("date")
    
    # Forward fill nulls
    df = df.fill_null(strategy="forward")
    
    # Drop any remaining nulls
    df = df.drop_nulls()
    
    print(f"Weather: {df.height} days ({df['date'].min()} to {df['date'].max()})")
    
    return df


def load_and_clean_ieso(data_dir: str = None) -> pl.DataFrame:
    """Load and clean IESO demand data."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "hourlydemanddata"
    else:
        data_dir = Path(data_dir)
    
    csv_files = sorted(data_dir.glob("PUB_Demand_*.csv"))
    
    dfs = []
    for csv_file in csv_files:
        year = int(csv_file.stem.split("_")[-1])
        if year < 2020 or year > 2024:
            continue
        
        df = pl.read_csv(csv_file, skip_rows=3, try_parse_dates=True)
        dfs.append(df)
    
    combined = pl.concat(dfs)
    
    # Clean column names
    combined.columns = [c.strip().lower().replace(" ", "_") for c in combined.columns]
    
    # Keep only needed columns
    cols = ["date", "hour", "market_demand", "ontario_demand"]
    combined = combined.select([c for c in cols if c in combined.columns])
    
    # Filter to 2020-2024
    combined = combined.filter(
        (pl.col("date").dt.year() >= 2020) & 
        (pl.col("date").dt.year() <= 2024)
    )
    
    combined = combined.sort(["date", "hour"])
    
    print(f"IESO: {combined.height} hours ({combined['date'].min()} to {combined['date'].max()})")
    
    return combined


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add time-based features."""
    return df.with_columns(
        # Time features from date
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
        pl.col("date").dt.year().alias("year"),
    ).with_columns(
        # Derived features
        (pl.col("day_of_week") >= 5).alias("is_weekend"),
        # Season (meteorological)
        pl.when(pl.col("month").is_in([12, 1, 2]))
        .then(pl.lit("winter"))
        .when(pl.col("month").is_in([3, 4, 5]))
        .then(pl.lit("spring"))
        .when(pl.col("month").is_in([6, 7, 8]))
        .then(pl.lit("summer"))
        .otherwise(pl.lit("fall"))
        .alias("season")
    )


def merge_weather_demand(weather: pl.DataFrame, demand: pl.DataFrame) -> pl.DataFrame:
    """Merge daily weather with hourly demand."""
    # For each hour, attach that day's weather
    merged = demand.join(weather, on="date", how="inner")
    
    # Add time features
    merged = add_time_features(merged)
    
    # Reorder columns
    col_order = [
        "date", "hour", "day_of_week", "month", "day_of_year", "year",
        "is_weekend", "season",
        "max_temperature", "min_temperature", "avg_temperature",
        "precipitation", "heatdegdays", "cooldegdays",
        "market_demand", "ontario_demand"
    ]
    merged = merged.select([c for c in col_order if c in merged.columns])
    
    return merged


def main():
    print("Loading data...")
    weather = load_and_clean_weather()
    demand = load_and_clean_ieso()
    
    print("\nMerging weather + demand...")
    merged = merge_weather_demand(weather, demand)
    
    print(f"\nFinal dataset: {merged.height:,} rows")
    print(f"Columns: {merged.columns}")
    print(f"\nDate range: {merged['date'].min()} to {merged['date'].max()}")
    
    # Save
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "merged_data.parquet"
    merged.write_parquet(output_path)
    print(f"\nSaved to: {output_path}")
    
    # Summary stats
    print("\n" + "=" * 50)
    print("SAMPLE DATA")
    print("=" * 50)
    print(merged.head(10))
    
    print("\n" + "=" * 50)
    print("STATS")
    print("=" * 50)
    print(merged.describe())


if __name__ == "__main__":
    main()
