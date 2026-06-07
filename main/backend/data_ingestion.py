"""
data_ingestion.py
─────────────────
Load real CSV from data/raw/ OR generate realistic synthetic
Tesla quarterly delivery/production data (2015–2025).
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_synthetic_data(cfg: dict) -> pd.DataFrame:
    """Build realistic synthetic Tesla quarterly data."""
    np.random.seed(42)
    quarters = pd.date_range(cfg["data"]["start_date"],
                             cfg["data"]["end_date"], freq="QS")
    n = len(quarters)

    # Real-ish Tesla delivery curve with noise
    real_deliveries = np.array([
        10045, 11532, 12955, 17478,
        12420, 14820, 24821, 22200,
        25418, 22026, 26137, 29870,
        29980, 40768, 83500, 90966,
        63000, 95200, 97000, 112095,
        88400, 90650, 139300, 180570,
        184800, 201250, 241300, 308650,
        310048, 254695, 343830, 405278,
        422875, 466140, 435059, 484507,
        386810, 443956, 462890, 495570,
        336000,
    ])
    deliveries = (real_deliveries + np.random.randint(-3000, 3000, n)).clip(5000)
    production = (deliveries * np.random.uniform(1.01, 1.06, n)).astype(int)

    stock_raw = np.array([
        42, 43, 49, 50, 45, 46, 48, 52, 52, 48, 55, 60,
        56, 50, 47, 50, 70, 55, 58, 70, 80, 78, 430, 700,
        800, 680, 800, 1100, 1100, 900, 800, 1200,
        150, 250, 270, 260, 200, 175, 250, 240, 180,
    ])
    stock_price = (stock_raw[:n] + np.random.normal(0, 8, n)).clip(10)
    asp = np.linspace(84000, 46000, n) + np.random.normal(0, 1200, n)
    revenue_bn = (deliveries * asp / 1e9).round(2)
    energy_gwh = (np.exp(0.25 * np.arange(n)) * 0.05
                  + np.random.normal(0, 0.2, n)).clip(0.1)
    sentiment = np.clip(
        50 + np.cumsum(np.random.normal(0, 3, n)), 5, 95
    )
    sentiment = ((sentiment - sentiment.min()) /
                 (sentiment.max() - sentiment.min()) * 100)

    df = pd.DataFrame({
        "date":              quarters,
        "year":              quarters.year,
        "quarter":           quarters.quarter,
        "deliveries":        deliveries.astype(int),
        "production":        production,
        "stock_price":       stock_price.round(2),
        "avg_selling_price": asp.round(0),
        "revenue_bn":        revenue_bn,
        "energy_storage_gwh": energy_gwh.round(2),
        "market_sentiment":  sentiment.round(1),
    })
    df["prod_delivery_gap"]   = df["production"] - df["deliveries"]
    df["delivery_growth_yoy"] = df["deliveries"].pct_change(4) * 100
    df["revenue_per_delivery"]= (df["revenue_bn"] * 1e9
                                 / df["deliveries"]).round(0)
    return df


def load_real_data(path: str) -> pd.DataFrame:
    """Load real Kaggle CSV and standardise column names."""
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def ingest(config_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    raw_path = Path(cfg["paths"]["raw_data"])

    if not cfg["data"]["synthetic"] and raw_path.exists():
        print(f"Loading real data from {raw_path}")
        df = load_real_data(str(raw_path))
    else:
        print("Generating synthetic Tesla data …")
        df = generate_synthetic_data(cfg)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"Saved to {raw_path}")

    print(f"Dataset shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = ingest()
    print(df.head())
