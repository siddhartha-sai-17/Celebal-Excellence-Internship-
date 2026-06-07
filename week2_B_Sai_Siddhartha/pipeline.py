"""
pipeline.py
───────────
Master orchestrator. Run this file to execute the full
end-to-end Tesla ML pipeline in sequence:

  1. Data ingestion
  2. Feature engineering
  3. Preprocessing / train-test split / scaling
  4. Model training (5 models)
  5. Hyperparameter tuning (XGBoost GridSearchCV)
  6. Model evaluation + feature importance
  7. Time series forecasting (Holt-Winters)
  8. Persist results JSON

Usage:
    python pipeline.py
    python pipeline.py --config path/to/config.yaml
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── allow running from project root ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from backend.data_ingestion      import ingest
from backend.feature_engineering import engineer, get_feature_cols, save_feature_cols
from backend.preprocessing       import preprocess
from backend.train               import train_all
from backend.tune                import tune
from backend.evaluate            import evaluate_all, feature_importance
from backend.forecast            import run_forecast

import yaml, joblib


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    out_path = Path(cfg["paths"]["outputs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 1 — DATA INGESTION")
    print("="*60)
    df = ingest(config_path)

    print("\n" + "="*60)
    print("STEP 2 — FEATURE ENGINEERING")
    print("="*60)
    df_fe  = engineer(df, config_path)
    f_cols = get_feature_cols(df_fe)
    save_feature_cols(f_cols, cfg["paths"]["models_dir"])

    print("\n" + "="*60)
    print("STEP 3 — PREPROCESSING")
    print("="*60)
    (X_tr, X_te, y_tr, y_te,
     X_tr_sc, X_te_sc, scaler, dates_test) = preprocess(
        df_fe, f_cols, config_path=config_path
    )

    print("\n" + "="*60)
    print("STEP 4 — MODEL TRAINING")
    print("="*60)
    base_results = train_all(
        X_tr, X_te, y_tr, y_te,
        X_tr_sc, X_te_sc, config_path
    )

    print("\n" + "="*60)
    print("STEP 5 — HYPERPARAMETER TUNING")
    print("="*60)
    tune_result = tune(X_tr, X_te, y_tr, y_te, config_path)

    print("\n" + "="*60)
    print("STEP 6 — EVALUATION + FEATURE IMPORTANCE")
    print("="*60)
    ridge   = joblib.load(Path(cfg["paths"]["models_dir"]) / "ridge_model.joblib")
    lasso   = joblib.load(Path(cfg["paths"]["models_dir"]) / "lasso_model.joblib")
    rf      = joblib.load(Path(cfg["paths"]["models_dir"]) / "rf_model.joblib")
    gb      = joblib.load(Path(cfg["paths"]["models_dir"]) / "gb_model.joblib")
    xgb_m   = joblib.load(Path(cfg["paths"]["models_dir"]) / "xgb_model.joblib")
    best    = tune_result["best_model"]

    models_dict = {
        "Ridge":          (ridge,  True),
        "Lasso":          (lasso,  True),
        "Random Forest":  (rf,     False),
        "Grad. Boosting": (gb,     False),
        "XGBoost":        (xgb_m,  False),
        "XGBoost (Tuned)":(best,   False),
    }
    eval_df = evaluate_all(models_dict, X_te, y_te, X_te_sc)

    fi_df = feature_importance(best, X_te, y_te, f_cols)
    print("\nTop 5 features by permutation importance:")
    print(fi_df.head(5).to_string(index=False))

    print("\n" + "="*60)
    print("STEP 7 — TIME SERIES FORECASTING")
    print("="*60)
    ts  = df.set_index("date")["deliveries"]
    fc  = run_forecast(ts, config_path)

    print("\n" + "="*60)
    print("STEP 8 — SAVING RESULTS")
    print("="*60)
    best_row = eval_df.iloc[0].to_dict()
    summary  = {
        "best_model":      best_row,
        "all_models":      eval_df.to_dict(orient="records"),
        "cv_r2":           {"mean": tune_result["cv_r2_mean"],
                            "std":  tune_result["cv_r2_std"]},
        "best_hyperparams":tune_result["best_params"],
        "top_features":    fi_df[fi_df["importance"] > 0]["feature"].tolist()[:5],
        "ts_backtest":     {"MAE":  fc["backtest_mae"],
                            "RMSE": fc["backtest_rmse"],
                            "R2":   fc["backtest_r2"]},
        "forecast_8q":     [{"date": d, "deliveries": v}
                            for d, v in zip(fc["future_dates"],
                                            fc["future_forecast"])],
    }

    with open(out_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Results saved → {out_path}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE ✅")
    print("="*60)
    print(f"Best model : {best_row['model']}")
    print(f"MAE        : {best_row['MAE']:,}")
    print(f"MAPE       : {best_row['MAPE']}%")
    print(f"R²         : {best_row['R2']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
