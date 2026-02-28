"""
Load Ontario IESO hourly electricity demand data.
"""
import polars as pl
from pathlib import Path


def load_ieso_data(data_dir: str = None, years: list = None) -> pl.DataFrame:
    """Load IESO hourly demand data from CSV files.
    
    Args:
        data_dir: Path to data/raw/hourlydemanddata folder.
        years: List of years to load. None = all years.
    
    Returns:
        Polars DataFrame with hourly demand data
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "hourlydemanddata"
    else:
        data_dir = Path(data_dir)
    
    # Find all CSV files
    csv_files = sorted(data_dir.glob("PUB_Demand_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No demand CSV files found in {data_dir}")
    
    dfs = []
    for csv_file in csv_files:
        # Extract year from filename
        year = int(csv_file.stem.split("_")[-1])
        
        if years and year not in years:
            continue
        
        # Read CSV, skipping 3 metadata rows
        df = pl.read_csv(
            csv_file,
            skip_rows=3,
            try_parse_dates=True,
        )
        
        dfs.append(df)
    
    # Concatenate all years
    if not dfs:
        raise ValueError("No data loaded. Check year filter.")
    
    combined = pl.concat(dfs)
    
    # Clean column names
    combined.columns = [c.strip().lower().replace(" ", "_") for c in combined.columns]
    
    # Select only the columns we need
    cols_to_keep = ["date", "hour", "market_demand", "ontario_demand"]
    cols_available = [c for c in cols_to_keep if c in combined.columns]
    combined = combined.select(cols_available)
    
    # Sort
    combined = combined.sort(["date", "hour"])
    
    return combined


def get_demand_summary(df: pl.DataFrame) -> dict:
    """Get summary statistics for demand data."""
    return {
        "date_range": (df["date"].min(), df["date"].max()),
        "total_records": df.height,
        "unique_days": df["date"].n_unique(),
        "demand_stats": df["ontario_demand"].describe(),
    }


if __name__ == "__main__":
    df = load_ieso_data()
    print(f"Loaded {df.height} hourly records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(df.head(10))
