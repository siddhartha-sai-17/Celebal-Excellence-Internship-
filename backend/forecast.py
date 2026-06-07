"""
forecast.py
───────────
Holt-Winters exponential smoothing for time series forecasting.
Runs ADF stationarity test, 4-quarter backtest, and 8-quarter
forward forecast. Saves fitted model as hw_model.joblib.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def adf_test(series: pd.Series) -> dict:
    result = adfuller(series)
    info   = {
        "statistic": round(result[0], 4),
        "p_value":   round(result[1], 4),
        "stationary": result[1] < 0.05,
    }
    print(f"ADF statistic: {info['statistic']}, p={info['p_value']}"
          f" → {'stationary' if info['stationary'] else 'non-stationary'}")
    return info


def run_forecast(ts: pd.Series,
                 config_path: str = "config.yaml") -> dict:
    cfg  = load_config(config_path)
    fc   = cfg["forecasting"]
    mdir = Path(cfg["paths"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)

    # Stationarity
    adf_info = adf_test(ts)

    # Backtest on last 4 quarters
    n_test  = fc["seasonal_periods"]
    ts_tr   = ts.iloc[:-n_test]
    ts_te   = ts.iloc[-n_test:]

    hw = ExponentialSmoothing(
        ts_tr,
        trend           = fc["trend"],
        seasonal        = fc["seasonal"],
        seasonal_periods= fc["seasonal_periods"],
    ).fit(optimized=True)

    hw_fc = hw.forecast(n_test)
    mae   = mean_absolute_error(ts_te, hw_fc)
    rmse  = np.sqrt(mean_squared_error(ts_te, hw_fc))
    r2    = r2_score(ts_te, hw_fc)
    print(f"Holt-Winters backtest → MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}")

    # Refit on full series for forward forecast
    hw_full = ExponentialSmoothing(
        ts,
        trend           = fc["trend"],
        seasonal        = fc["seasonal"],
        seasonal_periods= fc["seasonal_periods"],
    ).fit(optimized=True)

    horizon       = fc["forecast_horizon"]
    future_fc     = hw_full.forecast(horizon)
    future_dates  = pd.date_range(
        ts.index[-1] + pd.DateOffset(months=3),
        periods=horizon, freq="QS"
    )

    print("\n🔮 Forward forecast:")
    for d, v in zip(future_dates, future_fc):
        print(f"  {d.strftime('%Y-%m')} → {int(v):,}")

    # Save
    joblib.dump(hw_full, mdir / "hw_model.joblib")
    print(f"\nHolt-Winters model saved → {mdir}/hw_model.joblib")

    return {
        "adf":            adf_info,
        "backtest_mae":   round(mae),
        "backtest_rmse":  round(rmse),
        "backtest_r2":    round(r2, 4),
        "future_dates":   [str(d.date()) for d in future_dates],
        "future_forecast": [int(v) for v in future_fc],
        "model":          hw_full,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from backend.data_ingestion import ingest
    df = ingest()
    ts = df.set_index("date")["deliveries"]
    run_forecast(ts)
