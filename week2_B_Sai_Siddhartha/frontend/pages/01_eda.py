"""
01_eda.py — EDA Streamlit page
"""
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from frontend.components.charts import deliveries_line, revenue_bar, yoy_bar
from frontend.components.metric_card import metric_row

st.set_page_config(page_title="EDA · Tesla ML", layout="wide")
st.markdown("## Exploratory data analysis")

DATA_PATH = ROOT / "data" / "raw" / "tesla_deliveries.csv"

if not DATA_PATH.exists():
    st.warning("Run `python pipeline.py` from the project root first.")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

metric_row([
    {"label": "Quarters",       "value": str(len(df)),       "sub": "2015–2025",          "color": "blue"},
    {"label": "Peak deliveries","value": f"{df['deliveries'].max():,}", "sub": "Q4 2024",  "color": "teal"},
    {"label": "Avg deliveries", "value": f"{int(df['deliveries'].mean()):,}", "sub": "per quarter","color": "amber"},
    {"label": "Delivery ↔ Rev", "value": f"{df['deliveries'].corr(df['revenue_bn']):.3f}",
     "sub": "correlation",    "color": "purple"},
])

st.markdown("---")
st.altair_chart(deliveries_line(df), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.altair_chart(revenue_bar(df), use_container_width=True)
with col2:
    st.altair_chart(yoy_bar(df), use_container_width=True)

st.markdown("#### Correlation matrix")
num_cols = ["deliveries", "production", "stock_price",
            "revenue_bn", "energy_storage_gwh", "market_sentiment"]
st.dataframe(df[num_cols].corr().round(3), use_container_width=True)

st.markdown("#### Raw data (last 12 quarters)")
st.dataframe(df.tail(12), use_container_width=True)
