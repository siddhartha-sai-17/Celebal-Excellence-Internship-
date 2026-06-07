"""
train.py
────────
Fit five regression models (Ridge, Lasso, Random Forest,
Gradient Boosting, XGBoost) and persist each as a .joblib file.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_model(name: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    print(f"  {name:<32}  MAE={mae:>9,.0f}  RMSE={rmse:>9,.0f}"
          f"  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": round(mae), "RMSE": round(rmse),
            "R2": round(r2, 4), "MAPE": round(mape, 2)}


def train_all(X_train, X_test, y_train, y_test,
              X_train_sc, X_test_sc,
              config_path: str = "config.yaml") -> list:
    cfg   = load_config(config_path)
    mc    = cfg["models"]
    mdir  = Path(cfg["paths"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)

    results = []

    # ── Ridge ────────────────────────────────────────────────────
    ridge = Ridge(alpha=mc["ridge"]["alpha"])
    ridge.fit(X_train_sc, y_train)
    results.append(evaluate_model("Ridge", y_test, ridge.predict(X_test_sc)))
    joblib.dump(ridge, mdir / "ridge_model.joblib")

    # ── Lasso ────────────────────────────────────────────────────
    lasso = Lasso(alpha=mc["lasso"]["alpha"],
                  max_iter=mc["lasso"]["max_iter"])
    lasso.fit(X_train_sc, y_train)
    results.append(evaluate_model("Lasso", y_test, lasso.predict(X_test_sc)))
    joblib.dump(lasso, mdir / "lasso_model.joblib")

    # ── Random Forest ─────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators    = mc["random_forest"]["n_estimators"],
        max_depth       = mc["random_forest"]["max_depth"],
        min_samples_leaf= mc["random_forest"]["min_samples_leaf"],
        random_state    = mc["random_forest"]["random_state"],
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", y_test, rf.predict(X_test)))
    joblib.dump(rf, mdir / "rf_model.joblib")

    # ── Gradient Boosting ─────────────────────────────────────────
    gb = GradientBoostingRegressor(
        n_estimators    = mc["gradient_boosting"]["n_estimators"],
        learning_rate   = mc["gradient_boosting"]["learning_rate"],
        max_depth       = mc["gradient_boosting"]["max_depth"],
        subsample       = mc["gradient_boosting"]["subsample"],
        min_samples_leaf= mc["gradient_boosting"]["min_samples_leaf"],
        random_state    = mc["gradient_boosting"]["random_state"],
    )
    gb.fit(X_train, y_train)
    results.append(evaluate_model("Gradient Boosting", y_test, gb.predict(X_test)))
    joblib.dump(gb, mdir / "gb_model.joblib")

    # ── XGBoost ───────────────────────────────────────────────────
    xgb_m = xgb.XGBRegressor(
        n_estimators = mc["xgboost"]["n_estimators"],
        learning_rate= mc["xgboost"]["learning_rate"],
        max_depth    = mc["xgboost"]["max_depth"],
        subsample    = mc["xgboost"]["subsample"],
        reg_alpha    = mc["xgboost"]["reg_alpha"],
        reg_lambda   = mc["xgboost"]["reg_lambda"],
        random_state = mc["xgboost"]["random_state"],
        verbosity    = mc["xgboost"]["verbosity"],
    )
    xgb_m.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
    results.append(evaluate_model("XGBoost", y_test, xgb_m.predict(X_test)))
    joblib.dump(xgb_m, mdir / "xgb_model.joblib")

    print(f"\nAll models saved to {mdir}/")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from backend.data_ingestion    import ingest
    from backend.feature_engineering import engineer, get_feature_cols, save_feature_cols
    from backend.preprocessing     import preprocess

    df   = engineer(ingest())
    cols = get_feature_cols(df)
    save_feature_cols(cols)
    X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc, _, _ = preprocess(df, cols)
    results = train_all(X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc)
    for r in results:
        print(r)
