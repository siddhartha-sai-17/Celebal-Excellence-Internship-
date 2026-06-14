# Tesla ML Pipeline

End-to-end machine learning pipeline on Tesla delivery & production data (2015–2025).

## Project Structure

```
tesla_ml_pipeline/
├── data/
│   ├── raw/                    # Source CSV (generated or real)
│   ├── processed/              # Engineered feature table
│   └── outputs/                # pipeline_results.json
├── backend/
│   ├── data_ingestion.py       # Load/generate data
│   ├── preprocessing.py        # Clean, scale, split
│   ├── feature_engineering.py  # Lags, rolling, cyclical features
│   ├── train.py                # Fit 5 regression models
│   ├── tune.py                 # GridSearchCV tuning
│   ├── evaluate.py             # Metrics + feature importance
│   └── forecast.py             # Holt-Winters 8Q forecast
├── models/                     # Saved .joblib artifacts
│   ├── ridge_model.joblib
│   ├── lasso_model.joblib
│   ├── rf_model.joblib
│   ├── gb_model.joblib
│   ├── xgb_model.joblib
│   ├── xgb_tuned.joblib        # Best GridSearchCV model
│   ├── hw_model.joblib         # Holt-Winters
│   ├── scaler.joblib           # StandardScaler
│   └── feature_cols.joblib     # Feature name list
├── frontend/
│   ├── app.py                  # Streamlit entry point
│   ├── pages/
│   │   ├── 01_eda.py
│   │   ├── 02_models.py
│   │   └── 03_forecast.py
│   ├── components/
│   │   ├── charts.py           # Reusable Altair chart helpers
│   │   └── metric_card.py      # Styled metric card widgets
│   └── plots/                  # Static PNG outputs
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
├── pipeline.py                 # Master orchestrator
├── config.yaml                 # All hyperparams and paths
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (trains all models, saves artifacts)
python pipeline.py

# 3. Launch the Streamlit dashboard
streamlit run frontend/app.py
```

## Pipeline Stages

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_ingestion.py` | Load real CSV or generate synthetic Tesla data |
| 2 | `feature_engineering.py` | Lag (1,2,4Q), rolling stats, sin/cos quarter, log scale |
| 3 | `preprocessing.py` | StandardScaler fit on train only, 80/20 time split |
| 4 | `train.py` | Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost |
| 5 | `tune.py` | GridSearchCV (3-fold) over XGBoost hyperparameters |
| 6 | `evaluate.py` | MAE, RMSE, MAPE, R², permutation feature importance |
| 7 | `forecast.py` | Holt-Winters additive (trend + 4Q seasonal), 8Q ahead |
| 8 | `pipeline.py` | Orchestrates all steps, saves `pipeline_results.json` |

## Using Real Kaggle Data

1. Download from: https://www.kaggle.com/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025
2. Place CSV at `data/raw/tesla_deliveries.csv`
3. Set `data.synthetic: false` in `config.yaml`
4. Run `python pipeline.py`

## Key Design Decisions

- **`scaler.joblib`** is fitted on the training set only to prevent data leakage
- **`feature_cols.joblib`** ensures consistent column ordering at inference time
- **`pipeline.py`** is the single entry point; frontend only loads pre-fitted artifacts
- All hyperparameters live in **`config.yaml`** — no magic numbers in Python files
- Time-respecting 80/20 split (no shuffle) preserves temporal ordering

## Models Trained

| Model | Type | Scaled Input |
|-------|------|-------------|
| Ridge | Linear | Yes |
| Lasso | Linear | Yes |
| Random Forest | Ensemble | No |
| Gradient Boosting | Ensemble | No |
| XGBoost | Boosting | No |
| XGBoost (Tuned) | Boosting + GridSearch | No |

## Time Series Forecast

Holt-Winters exponential smoothing with:
- Additive trend
- Additive seasonality (4 quarter periods)
- ADF stationarity test before fitting
- 4-quarter backtest for validation
- 8-quarter forward forecast with ±15% confidence interval
