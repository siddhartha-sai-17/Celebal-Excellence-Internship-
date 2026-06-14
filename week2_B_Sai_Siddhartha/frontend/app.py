"""
app.py
──────
Streamlit entry point for the Tesla ML Pipeline dashboard.

Run with:
    streamlit run frontend/app.py
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from frontend.components.metric_card import metric_row
from frontend.components.charts import (
    deliveries_line, revenue_bar, yoy_bar,
    model_comparison_bar, feature_importance_bar, forecast_line,
)

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla ML Pipeline",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;1,9..144,300&display=swap');
html,body,[class*="css"]{font-family:'DM Mono',monospace!important}
.stApp{background:#07111c;color:#c9dbe8}
.main .block-container{padding:2rem 2.5rem 4rem;max-width:1300px}
section[data-testid="stSidebar"]{background:#0b1a27!important;border-right:1px solid #1a3048!important}
#MainMenu,footer,header{visibility:hidden}
h1,h2,h3{font-family:'Fraunces',Georgia,serif!important;font-weight:300!important}
.stButton>button{background:transparent!important;border:1px solid #2a7fcf!important;
  color:#5baef0!important;border-radius:6px!important;font-family:'DM Mono',monospace!important;
  font-size:0.78rem!important;letter-spacing:0.15em!important;text-transform:uppercase!important}
.stButton>button:hover{background:#0e2538!important;border-color:#1de9b6!important;color:#1de9b6!important}
.stTextArea textarea{background:#060f18!important;border:1px solid #1a3048!important;
  border-radius:8px!important;color:#c9dbe8!important;font-family:'DM Mono',monospace!important}
.stSelectbox>div>div{background:#060f18!important;border:1px solid #1a3048!important;color:#c9dbe8!important}
.stProgress>div>div>div{background:linear-gradient(90deg,#2a7fcf,#1de9b6)!important}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"


@st.cache_resource
def load_artifacts():
    arts = {}
    for name in ["xgb_tuned", "rf_model", "scaler", "hw_model"]:
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            arts[name] = joblib.load(p)
    p = MODELS_DIR / "feature_cols.joblib"
    if p.exists():
        arts["feature_cols"] = joblib.load(p)
    p = DATA_DIR / "outputs" / "pipeline_results.json"
    if p.exists():
        with open(p) as f:
            arts["results"] = json.load(f)
    return arts


@st.cache_data
def load_data():
    p = DATA_DIR / "raw" / "tesla_deliveries.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return None


arts = load_artifacts()
df   = load_data()

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 2rem;border-bottom:1px solid #1a3048;margin-bottom:2rem;">
    <div style="font-size:0.7rem;letter-spacing:0.22em;color:#e82127;text-transform:uppercase;margin-bottom:0.6rem;">
        ML Pipeline Dashboard
    </div>
    <h1 style="font-size:2.8rem;color:#e8f4ff;margin:0;line-height:1;">
        Tesla <span style="color:#e82127;font-style:italic;">Deliveries</span>
    </h1>
    <div style="font-size:0.78rem;color:#4a7a9b;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.6rem;">
        2015 – 2025 · End-to-End ML Pipeline
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", [
        "Overview",
        "EDA",
        "Model Results",
        "Feature Importance",
        "Forecast",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1a3048;margin:1.5rem 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem;color:#2a4a60;line-height:1.9;'>"
        "Run <code>python pipeline.py</code> first to generate model "
        "artifacts and results JSON.<br><br>"
        "Then <code>streamlit run frontend/app.py</code> to launch this dashboard."
        "</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("### Pipeline summary")

    if arts.get("results"):
        r = arts["results"]
        bm = r["best_model"]
        metric_row([
            {"label": "Best model",  "value": bm["model"].replace(" ", "\u202F"),
             "sub": "lowest MAPE", "color": "teal"},
            {"label": "MAPE",        "value": f"{bm['MAPE']}%",
             "sub": "mean abs % error", "color": "blue"},
            {"label": "MAE",         "value": f"{bm['MAE']:,}",
             "sub": "units", "color": "amber"},
            {"label": "R²",          "value": str(bm["R2"]),
             "sub": "test set", "color": "purple"},
        ])
    else:
        st.info("Run `python pipeline.py` to populate results.")

    if df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.altair_chart(deliveries_line(df), use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────────
elif page == "EDA":
    st.markdown("### Exploratory data analysis")
    if df is None:
        st.warning("Run pipeline.py first to generate the dataset.")
    else:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.altair_chart(revenue_bar(df), use_container_width=True)
        with col2:
            st.altair_chart(yoy_bar(df), use_container_width=True)

        st.markdown("#### Raw data")
        st.dataframe(df.tail(12), use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# MODEL RESULTS
# ─────────────────────────────────────────────────────────────────
elif page == "Model Results":
    st.markdown("### Regression model comparison")
    if arts.get("results"):
        r       = arts["results"]
        mdf     = pd.DataFrame(r["all_models"])
        st.altair_chart(model_comparison_bar(mdf), use_container_width=True)
        st.markdown("#### Full metrics table")
        st.dataframe(mdf.style.format({
            "MAE": "{:,.0f}", "RMSE": "{:,.0f}",
            "MAPE": "{:.2f}%", "R2": "{:.4f}"
        }), use_container_width=True)
        st.markdown("#### Best hyperparameters (GridSearchCV)")
        st.json(r["best_hyperparams"])
    else:
        st.info("Run `python pipeline.py` first.")

# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────
elif page == "Feature Importance":
    st.markdown("### Permutation feature importance")
    if arts.get("results"):
        top = arts["results"].get("top_features", [])
        st.markdown(f"**Top features:** {', '.join(top)}")
        st.info("Re-run `pipeline.py` with `evaluate.py` to regenerate "
                "the full importance DataFrame for charting.")
    else:
        st.info("Run `python pipeline.py` first.")

# ─────────────────────────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────────────────────────
elif page == "Forecast":
    st.markdown("### Holt-Winters 8-quarter forecast")
    if arts.get("results") and df is not None:
        r  = arts["results"]
        ts = df.set_index("date")["deliveries"]

        fq = r["forecast_8q"]
        future_dates = pd.to_datetime([x["date"] for x in fq])
        future_fc    = [x["deliveries"] for x in fq]

        st.altair_chart(forecast_line(ts, future_dates, future_fc),
                        use_container_width=True)

        metric_row([
            {"label": "Backtest MAE",  "value": f"{r['ts_backtest']['MAE']:,}",
             "sub": "last 4 quarters", "color": "blue"},
            {"label": "Backtest RMSE", "value": f"{r['ts_backtest']['RMSE']:,}",
             "sub": "last 4 quarters", "color": "amber"},
            {"label": "Backtest R²",   "value": str(r["ts_backtest"]["R2"]),
             "sub": "", "color": "teal"},
        ])

        st.markdown("#### Quarterly forecast table")
        fc_df = pd.DataFrame(fq)
        st.dataframe(fc_df.style.format({"deliveries": "{:,}"}),
                     use_container_width=True)
    else:
        st.info("Run `python pipeline.py` first.")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-size:0.68rem;letter-spacing:0.1em;color:#2a4a60;
            text-align:center;padding:2rem 0 0;border-top:1px solid #1a3048;margin-top:3rem;">
    Tesla ML Pipeline · scikit-learn · XGBoost · statsmodels · Streamlit
</div>
""", unsafe_allow_html=True)
