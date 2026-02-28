"""
Feature engineering for grid demand prediction.
"""
import polars as pl
import numpy as np
from pathlib import Path


def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lag features for demand."""
    # Sort by date + hour first
    df = df.sort(["date", "hour"])
    
    # Lags (previous hours)
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, 3h, 6h, 12h, 24h, 48h, 1 week
    
    for lag in lag_hours:
        df = df.with_columns(
            pl.col("ontario_demand").shift(lag).alias(f"lag_{lag}")
        )
    
    return df


def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling window features."""
    # Rolling mean and std for demand
    for window in [6, 24, 168]:  # 6h, 24h, 1 week
        df = df.with_columns(
            pl.col("ontario_demand").rolling_mean(window).alias(f"rolling_mean_{window}"),
            pl.col("ontario_demand").rolling_std(window).alias(f"rolling_std_{window}"),
        )
    
    # Same for temperature
    for col in ["avg_temperature", "heatdegdays", "cooldegdays"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).rolling_mean(24).alias(f"{col}_rolling_24")
            )
    
    return df


def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cyclical encoding for time features."""
    # Hour (0-23)
    df = df.with_columns(
        (2 * np.pi * pl.col("hour") / 24).alias("hour_rad")
    ).with_columns(
        pl.col("hour_rad").sin().alias("hour_sin"),
        pl.col("hour_rad").cos().alias("hour_cos")
    )
    
    # Day of week (1-7)
    df = df.with_columns(
        (2 * np.pi * pl.col("day_of_week") / 7).alias("dow_rad")
    ).with_columns(
        pl.col("dow_rad").sin().alias("dow_sin"),
        pl.col("dow_rad").cos().alias("dow_cos")
    )
    
    # Month (1-12)
    df = df.with_columns(
        (2 * np.pi * pl.col("month") / 12).alias("month_rad")
    ).with_columns(
        pl.col("month_rad").sin().alias("month_sin"),
        pl.col("month_rad").cos().alias("month_cos")
    )
    
    # Day of year (1-365)
    df = df.with_columns(
        (2 * np.pi * pl.col("day_of_year") / 365).alias("doy_rad")
    ).with_columns(
        pl.col("doy_rad").sin().alias("doy_sin"),
        pl.col("doy_rad").cos().alias("doy_cos")
    )
    
    return df


def add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add interaction features."""
    # Temperature x Hour interaction
    df = df.with_columns(
        (pl.col("avg_temperature") * pl.col("hour")).alias("temp_hour_interaction"),
        (pl.col("heatdegdays") * pl.col("is_weekend").cast(pl.Int8)).alias("hdd_weekend"),
        (pl.col("cooldegdays") * pl.col("is_weekend").cast(pl.Int8)).alias("cdd_weekend"),
    )
    
    return df


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply all feature engineering."""
    print("Adding lag features...")
    df = add_lag_features(df)
    
    print("Adding rolling features...")
    df = add_rolling_features(df)
    
    print("Adding cyclical features...")
    df = add_cyclical_features(df)
    
    print("Adding interaction features...")
    df = add_interaction_features(df)
    
    return df


def main():
    # Load processed data
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "merged_data.parquet"
    df = pl.read_parquet(data_path)
    
    print(f"Input: {df.height:,} rows, {df.width} columns")
    
    # Engineer features
    df = engineer_features(df)
    
    print(f"\nOutput: {df.height:,} rows, {df.width} columns")
    print(f"\nNew columns: {[c for c in df.columns if c not in ['date', 'hour', 'day_of_week', 'month', 'day_of_year', 'year', 'is_weekend', 'season', 'max_temperature', 'min_temperature', 'avg_temperature', 'precipitation', 'heatdegdays', 'cooldegdays', 'market_demand', 'ontario_demand']]}")
    
    # Drop rows with NaN from lags (first ~168 rows)
    print(f"\nDropping first 168 rows (NaN from lag features)...")
    df = df.slice(168)
    
    print(f"Final: {df.height:,} rows, {df.width} columns")
    
    # Save
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "features.parquet"
    df.write_parquet(output_path)
    print(f"\nSaved to: {output_path}")
    
    # Show sample
    print("\n" + "=" * 50)
    print("SAMPLE (first 5 rows, key columns)")
    print("=" * 50)
    key_cols = ["date", "hour", "avg_temperature", "ontario_demand", 
                "lag_1", "lag_24", "rolling_mean_24", "hour_sin", "hour_cos"]
    print(df.select(key_cols).head())


if __name__ == "__main__":
    main()
