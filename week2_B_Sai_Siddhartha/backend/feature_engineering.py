"""
feature_engineering.py
───────────────────────
Build lag, rolling, cyclical, log, and ratio features
from the cleaned raw dataframe. Saves feature column list.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def engineer(df: pd.DataFrame,
             config_path: str = "config.yaml") -> pd.DataFrame:
    cfg  = load_config(config_path)
    fe   = cfg["feature_engineering"]
    df   = df.copy().sort_values("date").reset_index(drop=True)

    # ── Lag features ──────────────────────────────────────────────
    for lag in fe["lags"]:
        df[f"deliv_lag{lag}"]  = df["deliveries"].shift(lag)
        df[f"stock_lag{lag}"]  = df["stock_price"].shift(lag)

    # ── Rolling features ──────────────────────────────────────────
    for w in fe["rolling_windows"]:
        df[f"deliv_roll{w}_mean"] = df["deliveries"].rolling(w).mean()
        df[f"deliv_roll{w}_std"]  = df["deliveries"].rolling(w).std()

    # ── Cyclical quarter encoding ─────────────────────────────────
    p = fe["cyclical_period"]
    df["q_sin"] = np.sin(2 * np.pi * df["quarter"] / p)
    df["q_cos"] = np.cos(2 * np.pi * df["quarter"] / p)

    # ── Trend index ───────────────────────────────────────────────
    df["year_idx"] = df["year"] - df["year"].min()

    # ── Log-scale lag ─────────────────────────────────────────────
    df["log_deliv_lag1"] = np.log1p(df["deliveries"].shift(1))

    # ── Ratio features ────────────────────────────────────────────
    df["prod_deliv_ratio"] = df["production"] / (df["deliveries"] + 1)
    df["rev_per_deliv_norm"] = (
        df["revenue_per_delivery"] / df["revenue_per_delivery"].max()
    )

    df = df.dropna().reset_index(drop=True)
    print(f"After feature engineering: {df.shape}")
    return df


FEATURE_COLS = [
    "year_idx", "q_sin", "q_cos",
    "avg_selling_price", "energy_storage_gwh", "market_sentiment",
    "deliv_lag1", "deliv_lag2", "deliv_lag4",
    "stock_lag1", "stock_lag2",
    "deliv_roll4_mean", "deliv_roll4_std",
    "deliv_roll8_mean", "deliv_roll8_std",
    "prod_deliv_ratio", "rev_per_deliv_norm",
    "prod_delivery_gap", "log_deliv_lag1",
]


def save_feature_cols(feature_cols: list,
                      models_dir: str = "models/"):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_cols, Path(models_dir) / "feature_cols.joblib")
    print(f"Feature columns saved → {models_dir}feature_cols.joblib")


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return only those FEATURE_COLS that exist in df."""
    return [c for c in FEATURE_COLS if c in df.columns]


if __name__ == "__main__":
    from data_ingestion import ingest
    df = engineer(ingest())
    cols = get_feature_cols(df)
    save_feature_cols(cols)
    print("Features:", cols)
