"""
02_models.py — Model Results Streamlit page
"""
import sys
import json
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from frontend.components.charts import model_comparison_bar
from frontend.components.metric_card import metric_row

st.set_page_config(page_title="Models · Tesla ML", layout="wide")
st.markdown("## Regression model results")

RESULTS_PATH = ROOT / "data" / "outputs" / "pipeline_results.json"

if not RESULTS_PATH.exists():
    st.warning("Run `python pipeline.py` from the project root first.")
    st.stop()

with open(RESULTS_PATH) as f:
    results = json.load(f)

bm = results["best_model"]
metric_row([
    {"label": "Best model", "value": bm["model"],          "sub": "lowest MAPE",      "color": "teal"},
    {"label": "MAE",        "value": f"{bm['MAE']:,}",      "sub": "units",            "color": "blue"},
    {"label": "RMSE",       "value": f"{bm['RMSE']:,}",     "sub": "units",            "color": "amber"},
    {"label": "MAPE",       "value": f"{bm['MAPE']}%",      "sub": "mean abs % error", "color": "red"},
    {"label": "R²",         "value": str(bm["R2"]),         "sub": "test set",         "color": "purple"},
])

st.markdown("---")
mdf = pd.DataFrame(results["all_models"])
st.altair_chart(model_comparison_bar(mdf), use_container_width=True)

st.markdown("#### All model metrics")
st.dataframe(
    mdf.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                      "MAPE": "{:.2f}%", "R2": "{:.4f}"}),
    use_container_width=True,
)

st.markdown("#### Best hyperparameters (GridSearchCV)")
col1, col2 = st.columns(2)
with col1:
    st.json(results["best_hyperparams"])
with col2:
    cv = results["cv_r2"]
    st.markdown(f"""
    **5-fold cross-validation R²**
    - Mean: `{cv['mean']}`
    - Std:  `{cv['std']}`
    """)

st.markdown("#### Top features")
st.write(", ".join(results.get("top_features", [])))
