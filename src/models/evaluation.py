"""Forecasting Model Evaluation Module for SolarScope.

Implements rigorous, zero-leakage performance evaluation metrics for renewable energy forecasting.
Includes MAE, RMSE, R², nMAE, nRMSE, and WAPE.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecast(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    capacity_kw: float,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Calculates comprehensive forecasting evaluation metrics.

    Args:
        y_true: Ground truth power measurements (kW)
        y_pred: Predicted power values (kW)
        capacity_kw: Total plant installed AC capacity for metric normalization
        model_name: Identifier for the model evaluated
    """
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.maximum(0.0, np.asarray(y_pred).flatten())  # Physical constraint: power >= 0

    # Clean any NaNs
    valid = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_t = y_true_arr[valid]
    y_p = y_pred_arr[valid]

    if len(y_t) == 0:
        return {
            "model_name": model_name,
            "mae_kw": 0.0,
            "rmse_kw": 0.0,
            "r2_score": 0.0,
            "nmae_pct": 0.0,
            "nrmse_pct": 0.0,
            "wape_pct": 0.0,
        }

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    # Normalized metrics (percentage of nominal plant capacity)
    nmae = (mae / capacity_kw) * 100.0 if capacity_kw > 0 else 0.0
    nrmse = (rmse / capacity_kw) * 100.0 if capacity_kw > 0 else 0.0

    # Weighted Absolute Percentage Error (WAPE = sum(|y - y_hat|) / sum(y))
    sum_true = float(np.sum(y_t))
    wape = (float(np.sum(np.abs(y_t - y_p))) / sum_true * 100.0) if sum_true > 0 else 0.0

    return {
        "model_name": model_name,
        "mae_kw": round(mae, 2),
        "rmse_kw": round(rmse, 2),
        "r2_score": round(r2, 4),
        "nmae_pct": round(nmae, 2),
        "nrmse_pct": round(nrmse, 2),
        "wape_pct": round(wape, 2),
        "sample_count": int(len(y_t)),
    }


def compare_model_metrics(metrics_list: list) -> pd.DataFrame:
    """Formats a comparison table across multiple models."""
    df = pd.DataFrame(metrics_list)
    cols_order = [
        "model_name", "mae_kw", "rmse_kw", "r2_score",
        "nmae_pct", "nrmse_pct", "wape_pct", "sample_count"
    ]
    existing = [c for c in cols_order if c in df.columns]
    return df[existing].sort_values("mae_kw").reset_index(drop=True)
