from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "crop_yield.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
TARGET_COLUMN = "Yield_tons_per_hectare"

FEATURE_COLUMNS = [
    "Region",
    "Soil_Type",
    "Crop",
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Weather_Condition",
    "Days_to_Harvest",
]

CATEGORICAL_FEATURES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
NUMERICAL_FEATURES = [
    "Rainfall_mm",
    "Temperature_Celsius",
    "Days_to_Harvest",
    "Fertilizer_Used",
    "Irrigation_Used",
]


def create_preprocessor() -> ColumnTransformer:
    """Create a robust preprocessing pipeline for heterogeneous agricultural data."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Any]:
    """Define candidate regression models for benchmark comparison."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=60,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_and_evaluate(
    data_path: str | Path = DATA_PATH,
    sample_size: int | None = 100000,
    save_dir: str | Path = MODELS_DIR,
) -> dict[str, Any]:
    """Train candidate models, benchmark performance, and serialize the best pipeline."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    if sample_size:
        print(f"Using representative sample of {sample_size:,} rows for fast robust training...")
        df = pd.read_csv(data_path, nrows=sample_size)
    else:
        print("Using full dataset...")
        df = pd.read_csv(data_path)

    # Validate target and features
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    for col in ["Fertilizer_Used", "Irrigation_Used"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    candidate_models = build_models()
    benchmark_results: list[dict[str, Any]] = []
    trained_pipelines: dict[str, Pipeline] = {}

    for name, estimator in candidate_models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        train_duration = time.time() - t0

        # Predictions
        preds = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mape = float(np.mean(np.abs((y_test - preds) / np.maximum(y_test, 0.1))) * 100.0)

        result_entry = {
            "model": name,
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "train_time_sec": round(train_duration, 2),
        }
        benchmark_results.append(result_entry)
        trained_pipelines[name] = pipeline
        print(f"{name} -> R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Time: {train_duration:.2f}s")

    # Sort benchmark by R2 desc, RMSE asc
    benchmark_results.sort(key=lambda m: (m["r2"], -m["rmse"]), reverse=True)
    best_entry = benchmark_results[0]
    best_model_name = best_entry["model"]
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n==========================================")
    print(f"BEST MODEL: {best_model_name} (R2 = {best_entry['r2']:.4f})")
    print(f"==========================================")

    # Extract feature importances
    feature_importances: list[dict[str, Any]] = []
    try:
        prep = best_pipeline.named_steps["preprocessor"]
        cat_encoder = prep.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        all_features = NUMERICAL_FEATURES + encoded_cat_names

        model_step = best_pipeline.named_steps["model"]
        if hasattr(model_step, "coef_"):
            coefs = np.abs(model_step.coef_)
            total = float(np.sum(coefs))
            for f_name, raw_c in zip(all_features, coefs):
                feature_importances.append({
                    "feature": f_name,
                    "importance_pct": round((float(raw_c) / total) * 100, 2),
                })
        elif hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            total = float(np.sum(importances))
            for f_name, imp in zip(all_features, importances):
                feature_importances.append({
                    "feature": f_name,
                    "importance_pct": round((float(imp) / total) * 100, 2),
                })
        feature_importances.sort(key=lambda x: x["importance_pct"], reverse=True)
    except Exception as exc:
        print(f"Note: Could not extract feature importances: {exc}")

    # Serialize artifacts
    model_file = save_path / "best_model.joblib"
    joblib.dump(best_pipeline, model_file)
    print(f"Saved model to: {model_file}")

    benchmark_file = save_path / "benchmark.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Saved benchmark to: {benchmark_file}")

    metrics_file = save_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": best_entry,
            "sample_size": sample_size or len(df),
            "total_records": len(df),
            "features": FEATURE_COLUMNS,
        }, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    if feature_importances:
        feat_file = save_path / "feature_importance.json"
        with open(feat_file, "w", encoding="utf-8") as f:
            json.dump(feature_importances, f, indent=2)
        print(f"Saved feature importances to: {feat_file}")

    return {
        "best_model": best_model_name,
        "benchmark": benchmark_results,
        "metrics": best_entry,
        "model_path": str(model_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate crop yield prediction models.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to crop_yield.csv")
    parser.add_argument("--sample-size", type=int, default=100000, help="Number of rows to sample (default 100,000)")
    parser.add_argument("--full", action="store_true", help="Train on full dataset (1M rows)")
    parser.add_argument("--save-dir", default=str(MODELS_DIR), help="Output directory for model artifacts")

    args = parser.parse_args()
    sample = None if args.full else args.sample_size
    train_and_evaluate(data_path=args.data_path, sample_size=sample, save_dir=args.save_dir)
