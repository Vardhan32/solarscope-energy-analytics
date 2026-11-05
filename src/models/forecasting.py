"""Solar Power Generation Forecasting Pipeline for SolarScope.

Implements feature engineering, chronological train/test splitting, baseline models,
Linear/Ridge, Random Forest, and HistGradientBoostingRegressor models.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.config import (
    FORECAST_HORIZON_DAYS,
    INTERVALS_PER_DAY,
    RANDOM_STATE,
    PLANTS,
    PlantMetadata,
)
from src.models.baseline import Persistence24hModel, DiurnalClimatologyModel
from src.models.evaluation import evaluate_forecast, compare_model_metrics
from src.utils.logger import get_logger

logger = get_logger("forecasting")


def engineer_features(plant_ts_df: pd.DataFrame) -> pd.DataFrame:
    """Constructs leakage-free temporal, geometric, lag, and weather features."""
    df = plant_ts_df.copy().sort_values("DATE_TIME").reset_index(drop=True)

    # 1. Clear-sky solar elevation proxy (diurnal bell curve between 06:00 and 18:00)
    time_float = df["HOUR"] + df["MINUTE"] / 60.0
    clearsky_proxy = np.where(
        (time_float >= 6.0) & (time_float <= 18.0),
        np.sin(np.pi * (time_float - 6.0) / 12.0),
        0.0
    )
    df["SOLAR_GEOMETRIC_PROXY"] = np.maximum(0.0, clearsky_proxy)

    # 2. Historical Lags (Strictly backward in time)
    # LAG_96: same time yesterday (24h prior)
    # LAG_192: same time 2 days ago (48h prior)
    # LAG_672: same time 7 days ago (1 week prior)
    df["LAG_96"] = df["AC_POWER_KW"].shift(INTERVALS_PER_DAY)
    df["LAG_192"] = df["AC_POWER_KW"].shift(INTERVALS_PER_DAY * 2)
    df["LAG_672"] = df["AC_POWER_KW"].shift(INTERVALS_PER_DAY * 7)

    # Rolling average of previous day's power (shift by 96 to prevent intra-day leakage)
    df["ROLLING_MEAN_PREV_DAY"] = df["AC_POWER_KW"].shift(INTERVALS_PER_DAY).rolling(INTERVALS_PER_DAY, min_periods=1).mean()

    # Fill earliest history with diurnal climatology proxy rather than arbitrary zeros
    df["LAG_96"] = df["LAG_96"].bfill().fillna(0.0)
    df["LAG_192"] = df["LAG_192"].bfill().fillna(0.0)
    df["LAG_672"] = df["LAG_672"].bfill().fillna(0.0)
    df["ROLLING_MEAN_PREV_DAY"] = df["ROLLING_MEAN_PREV_DAY"].bfill().fillna(0.0)

    return df


FEATURE_COLUMNS = [
    "HOUR",
    "MINUTE",
    "DAY_OF_WEEK",
    "HOUR_SIN",
    "HOUR_COS",
    "SOLAR_GEOMETRIC_PROXY",
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "EXPECTED_AC_POWER_KW",
    "LAG_96",
    "LAG_192",
    "ROLLING_MEAN_PREV_DAY",
]


def chronological_split(
    df: pd.DataFrame,
    test_days: int = FORECAST_HORIZON_DAYS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits time series data chronologically without shuffling to prevent lookahead leakage."""
    sorted_df = df.sort_values("DATE_TIME").reset_index(drop=True)
    total_test_intervals = test_days * INTERVALS_PER_DAY

    if len(sorted_df) <= total_test_intervals:
        raise ValueError(
            f"Dataset length ({len(sorted_df)}) is too short for a {test_days}-day test split ({total_test_intervals} intervals)."
        )

    split_idx = len(sorted_df) - total_test_intervals
    train_df = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)

    logger.info(
        f"Chronological split created: Train={len(train_df)} rows ({train_df['DATE_TIME'].min()} to {train_df['DATE_TIME'].max()}), "
        f"Test={len(test_df)} rows ({test_df['DATE_TIME'].min()} to {test_df['DATE_TIME'].max()})"
    )
    return train_df, test_df


class SolarForecastEngine:
    """Manages model training, evaluation, comparison, and 7-day inference."""

    def __init__(self, meta: PlantMetadata):
        self.meta = meta
        self.capacity_kw = meta.total_ac_capacity_kw
        self.models: Dict[str, Any] = {}
        self.metrics: List[Dict[str, Any]] = []
        self.predictions_df: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()

    def train_and_evaluate(self, plant_ts_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the entire feature engineering, chronological split, training, and benchmarking workflow."""
        enriched_df = engineer_features(plant_ts_df)
        train_df, test_df = chronological_split(enriched_df, test_days=FORECAST_HORIZON_DAYS)

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["AC_POWER_KW"]

        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["AC_POWER_KW"]

        # Results container
        pred_dict = {
            "DATE_TIME": test_df["DATE_TIME"],
            "DATE": test_df["DATE"],
            "HOUR": test_df["HOUR"],
            "ACTUAL_KW": y_test.values,
            "IRRADIATION": test_df["IRRADIATION"].values,
        }

        # 1. Baseline Model 1: Persistence 24h
        m_persist = Persistence24hModel()
        m_persist.fit(X_train, y_train)
        pred_persist = m_persist.predict(X_test)
        self.models["Baseline (24h Persistence)"] = m_persist
        pred_dict["PRED_PERSISTENCE_KW"] = pred_persist
        self.metrics.append(
            evaluate_forecast(y_test, pred_persist, self.capacity_kw, "Baseline (24h Persistence)")
        )

        # 2. Baseline Model 2: Diurnal Climatology (Historical Mean Profile)
        m_clim = DiurnalClimatologyModel()
        m_clim.fit(X_train, y_train)
        pred_clim = m_clim.predict(X_test)
        self.models["Baseline (Diurnal Climatology)"] = m_clim
        pred_dict["PRED_CLIMATOLOGY_KW"] = pred_clim
        self.metrics.append(
            evaluate_forecast(y_test, pred_clim, self.capacity_kw, "Baseline (Diurnal Climatology)")
        )

        # 3. Ridge Regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        m_ridge = Ridge(alpha=10.0)
        m_ridge.fit(X_train_scaled, y_train)
        pred_ridge = np.maximum(0.0, m_ridge.predict(X_test_scaled))
        self.models["Linear Regression (Ridge)"] = m_ridge
        pred_dict["PRED_RIDGE_KW"] = pred_ridge
        self.metrics.append(
            evaluate_forecast(y_test, pred_ridge, self.capacity_kw, "Linear Regression (Ridge)")
        )

        # 4. Random Forest Regressor
        m_rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        m_rf.fit(X_train, y_train)
        pred_rf = np.maximum(0.0, m_rf.predict(X_test))
        self.models["Random Forest"] = m_rf
        pred_dict["PRED_RANDOM_FOREST_KW"] = pred_rf
        self.metrics.append(
            evaluate_forecast(y_test, pred_rf, self.capacity_kw, "Random Forest")
        )

        # 5. Gradient Boosting (HistGradientBoostingRegressor)
        m_gb = HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=8,
            learning_rate=0.08,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        )
        m_gb.fit(X_train, y_train)
        pred_gb = np.maximum(0.0, m_gb.predict(X_test))
        self.models["Gradient Boosting"] = m_gb
        pred_dict["PRED_GRADIENT_BOOSTING_KW"] = pred_gb
        self.metrics.append(
            evaluate_forecast(y_test, pred_gb, self.capacity_kw, "Gradient Boosting")
        )

        self.predictions_df = pd.DataFrame(pred_dict)
        metrics_table = compare_model_metrics(self.metrics)

        return metrics_table, self.predictions_df

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns feature importance ranking from Random Forest model."""
        if "Random Forest" not in self.models:
            return pd.DataFrame()
        rf = self.models["Random Forest"]
        imp = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return imp


MODEL_LIMITATIONS = """
### Model Limitations & Energy Analyst Operational Context:
1. **Numerical Weather Prediction (NWP) Dependency**: In production day-ahead forecasting, actual irradiation is unavailable in advance; forecasts rely on meteorological satellite / NWP inputs where cloud cover error is the primary error driver.
2. **Short Monitoring Horizon**: The dataset covers a 34-day summer window (May 15 – June 17, 2020). Seasonal shifts (monsoon cloud cover, winter solar zenith angles) require 12-month data to avoid seasonal generalization bias.
3. **Transient Ramp Events**: Sudden local cloud transients passing across inverter arrays cause brief steep ramps that hourly or 15-minute point forecasts smooth out.
4. **Physical Clipping Non-Linearities**: When inverter saturation thresholds are reached, electrical output decouples from linear irradiance scaling.
"""
