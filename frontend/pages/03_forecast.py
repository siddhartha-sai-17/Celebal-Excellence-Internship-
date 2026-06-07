"""
03_forecast.py — Forecast Streamlit page
"""
import sys
import json
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from frontend.components.charts import forecast_line
from frontend.components.metric_card import metric_row

st.set_page_config(page_title="Forecast · Tesla ML", layout="wide")
st.markdown("## Holt-Winters time series forecast")

DATA_PATH    = ROOT / "data" / "raw" / "tesla_deliveries.csv"
RESULTS_PATH = ROOT / "data" / "outputs" / "pipeline_results.json"

if not DATA_PATH.exists() or not RESULTS_PATH.exists():
    st.warning("Run `python pipeline.py` from the project root first.")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
with open(RESULTS_PATH) as f:
    results = json.load(f)

ts  = df.set_index("date")["deliveries"]
fq  = results["forecast_8q"]
future_dates = pd.to_datetime([x["date"] for x in fq])
future_fc    = [x["deliveries"] for x in fq]

metric_row([
    {"label": "Method",       "value": "Holt-Winters",          "sub": "additive trend + seasonal", "color": "blue"},
    {"label": "Backtest MAE", "value": f"{results['ts_backtest']['MAE']:,}", "sub": "last 4 quarters","color": "teal"},
    {"label": "Backtest R²",  "value": str(results["ts_backtest"]["R2"]),   "sub": "",               "color": "amber"},
    {"label": "Horizon",      "value": "8 quarters",             "sub": "through Q1 2027",           "color": "purple"},
])

st.markdown("---")
st.altair_chart(forecast_line(ts, future_dates, future_fc), use_container_width=True)

st.markdown("#### Quarterly forecast table")
fc_df = pd.DataFrame(fq)
fc_df["date"] = pd.to_datetime(fc_df["date"])
fc_df["quarter"] = fc_df["date"].dt.to_period("Q").astype(str)
fc_df = fc_df[["quarter", "date", "deliveries"]]
st.dataframe(
    fc_df.style.format({"deliveries": "{:,}"}),
    use_container_width=True,
)
