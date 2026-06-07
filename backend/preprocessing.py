"""
preprocessing.py
────────────────
Clean raw dataframe, apply StandardScaler, perform
time-respecting train/test split, and save scaler artifact.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop nulls, reset index, sort by date."""
    df = df.copy()
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["deliveries"])
    return df


def split(df: pd.DataFrame,
          feature_cols: list,
          target_col: str,
          test_size: float = 0.20):
    """Time-respecting split (no shuffle)."""
    split_idx = int(len(df) * (1 - test_size))
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df["date"].iloc[split_idx:] if "date" in df.columns else None
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, dates_test


def scale(X_train: pd.DataFrame,
          X_test: pd.DataFrame,
          models_dir: str = "models/"):
    """Fit StandardScaler on train only; save to disk."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, Path(models_dir) / "scaler.joblib")
    print(f"Scaler saved → {models_dir}scaler.joblib")
    return X_train_sc, X_test_sc, scaler


def preprocess(df: pd.DataFrame,
               feature_cols: list,
               target_col: str = "deliveries",
               config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    df  = clean(df)
    X_train, X_test, y_train, y_test, dates_test = split(
        df, feature_cols, target_col,
        test_size=cfg["data"]["test_split"]
    )
    X_train_sc, X_test_sc, scaler = scale(
        X_train, X_test,
        models_dir=cfg["paths"]["models_dir"]
    )
    return (X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, scaler, dates_test)


if __name__ == "__main__":
    from data_ingestion import ingest
    from feature_engineering import engineer
    df = engineer(ingest())
    feature_cols = [c for c in df.columns
                    if c not in ("date", "year", "quarter",
                                 "deliveries", "production",
                                 "revenue_bn", "delivery_growth_yoy",
                                 "revenue_per_delivery")]
    preprocess(df, feature_cols)
