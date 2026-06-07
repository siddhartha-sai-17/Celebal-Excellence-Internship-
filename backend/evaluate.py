"""
evaluate.py
───────────
Compute MAE, RMSE, MAPE, R² for any fitted model.
Also runs permutation importance analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


def compute_metrics(name: str, y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1)))) * 100
    return {
        "model": name,
        "MAE":   round(mae),
        "RMSE":  round(rmse),
        "R2":    round(r2, 4),
        "MAPE":  round(mape, 2),
    }


def evaluate_all(models: dict,
                 X_test, y_test,
                 X_test_sc=None) -> pd.DataFrame:
    """
    models: dict of {name: (model, needs_scaled)}
    e.g. {"Ridge": (ridge, True), "XGBoost": (xgb_m, False)}
    """
    rows = []
    for name, (model, scaled) in models.items():
        X = X_test_sc if scaled and X_test_sc is not None else X_test
        y_pred = model.predict(X)
        rows.append(compute_metrics(name, y_test.values, y_pred))
    df = pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)
    print(df.to_string(index=False))
    return df


def feature_importance(model, X_test, y_test,
                       feature_names: list,
                       n_repeats: int = 20,
                       random_state: int = 42) -> pd.DataFrame:
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state
    )
    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi_df


if __name__ == "__main__":
    print("Run from pipeline.py or import directly.")
