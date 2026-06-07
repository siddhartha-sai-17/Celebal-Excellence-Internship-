"""
tune.py
───────
GridSearchCV hyperparameter tuning for XGBoost.
Saves the best estimator as xgb_tuned.joblib.
"""

import yaml
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def tune(X_train, X_test, y_train, y_test,
         config_path: str = "config.yaml") -> dict:
    cfg   = load_config(config_path)
    tc    = cfg["tuning"]
    mdir  = Path(cfg["paths"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)

    print("Running GridSearchCV …")
    gs = GridSearchCV(
        estimator  = xgb.XGBRegressor(random_state=42, verbosity=0),
        param_grid = tc["param_grid"],
        cv         = tc["cv_folds"],
        scoring    = tc["scoring"],
        n_jobs     = -1,
        refit      = True,
        verbose    = 1,
    )
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    best_model  = gs.best_estimator_
    print(f"Best params: {best_params}")

    # 5-fold CV on full dataset
    cv_r2 = cross_val_score(
        best_model, X_train, y_train,
        cv=cfg["evaluation"]["cv_folds"], scoring="r2"
    )
    print(f"5-fold CV R²: {np.round(cv_r2,3)}  "
          f"mean={cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # Save
    joblib.dump(best_model, mdir / "xgb_tuned.joblib")
    print(f"Tuned model saved → {mdir}/xgb_tuned.joblib")

    return {
        "best_params": best_params,
        "cv_r2_mean" : round(float(cv_r2.mean()), 4),
        "cv_r2_std"  : round(float(cv_r2.std()),  4),
        "best_model" : best_model,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from backend.data_ingestion      import ingest
    from backend.feature_engineering import engineer, get_feature_cols, save_feature_cols
    from backend.preprocessing       import preprocess

    df   = engineer(ingest())
    cols = get_feature_cols(df)
    save_feature_cols(cols)
    X_tr, X_te, y_tr, y_te, *_ = preprocess(df, cols)
    result = tune(X_tr, X_te, y_tr, y_te)
    print(result["best_params"])
