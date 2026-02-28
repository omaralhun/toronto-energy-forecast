"""
Grid demand forecasting using LightGBM.
"""
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from pathlib import Path
import pickle


def load_features_data() -> pl.DataFrame:
    """Load the engineered features dataset."""
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "features.parquet"
    df = pl.read_parquet(data_path)
    return df


def prepare_data(df: pl.DataFrame):
    """Prepare features and target for training."""
    # Target
    target = "ontario_demand"
    
    # Features to exclude from training
    exclude = ["date", "market_demand", "ontario_demand", "season"]
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Convert to pandas
    pdf = df.to_pandas()
    
    X = pdf[feature_cols]
    y = pdf[target]
    
    # Handle any remaining NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(method="ffill").fillna(method="bfill")
    
    return X, y, feature_cols


def train_model(X, y, feature_cols):
    """Train LightGBM model."""
    # Time-based split (use last 20% as validation)
    split_idx = int(len(X) * 0.8)
    
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    
    # LightGBM parameters
    params = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    print("\nTraining LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model, X_val, y_val


def evaluate_model(model, X_val, y_val, feature_cols):
    """Evaluate model performance."""
    y_pred = model.predict(X_val)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"RMSE: {rmse:,.2f} MW")
    print(f"MAE:  {mae:,.2f} MW")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    
    print("\n" + "=" * 50)
    print("TOP 15 FEATURES")
    print("=" * 50)
    for i, (feat, imp) in enumerate(feat_imp[:15]):
        print(f"{i+1:2}. {feat:30} {imp:,.0f}")
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "feature_importance": feat_imp
    }


def main():
    print("Loading features data...")
    df = load_features_data()
    print(f"Data: {df.height:,} rows, {df.width} columns")
    
    print("\nPreparing data...")
    X, y, feature_cols = prepare_data(df)
    print(f"Features: {len(feature_cols)}")
    print(f"Target: ontario_demand (MW)")
    
    # Train model
    model, X_val, y_val = train_model(X, y, feature_cols)
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val, feature_cols)
    
    # Save model
    output_dir = Path(__file__).parent.parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    
    model.save_model(str(output_dir / "grid_demand_model.txt"))
    
    # Save metrics
    with open(output_dir / "model_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    
    print(f"\n" + "=" * 50)
    print("MODEL SAVED")
    print("=" * 50)
    print(f"Model: {output_dir / 'grid_demand_model.txt'}")
    print(f"Metrics: {output_dir / 'model_metrics.pkl'}")


if __name__ == "__main__":
    main()
