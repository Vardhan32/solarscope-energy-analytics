"""Baseline Forecasting Models for SolarScope.

Implements standard benchmark models for diurnal solar power forecasting:
1. Persistence (24-hour lag: Today's profile = Yesterday's profile)
2. Diurnal Climatology (Historical Mean Profile by time of day)
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class Persistence24hModel:
    """Predicts power at time t as the observed power at time t - 24 hours."""

    def __init__(self, intervals_per_day: int = 96):
        self.intervals_per_day = intervals_per_day
        self.last_day_profile_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Persistence24hModel":
        # Cache the most recent 24-hour profile from training data
        if len(y) >= self.intervals_per_day:
            self.last_day_profile_ = y.iloc[-self.intervals_per_day:].values
        else:
            self.last_day_profile_ = np.zeros(self.intervals_per_day)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions using 24-hour lag from input feature if available,
        or rolling persistence.
        """
        if "LAG_96" in X.columns:
            return np.maximum(0.0, X["LAG_96"].fillna(0.0).values)

        # Fallback if lag column not provided directly
        n = len(X)
        if self.last_day_profile_ is None:
            return np.zeros(n)
        repeats = int(np.ceil(n / self.intervals_per_day))
        tiled = np.tile(self.last_day_profile_, repeats)[:n]
        return np.maximum(0.0, tiled)


class DiurnalClimatologyModel:
    """Predicts power as the historical mean for each (hour, minute) interval."""

    def __init__(self):
        self.hourly_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DiurnalClimatologyModel":
        df = X.copy()
        df["TARGET"] = y.values
        df["TIME_KEY"] = df["HOUR"].astype(str) + ":" + df["MINUTE"].astype(str)

        self.hourly_means_ = df.groupby("TIME_KEY")["TARGET"].mean().to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        df["TIME_KEY"] = df["HOUR"].astype(str) + ":" + df["MINUTE"].astype(str)
        preds = df["TIME_KEY"].map(self.hourly_means_).fillna(self.global_mean_).values
        return np.maximum(0.0, preds)
