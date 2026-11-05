"""Forecasting Studio View for SolarScope."""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import PLANTS, MODELS_DIR
from src.models.forecasting import SolarForecastEngine, MODEL_LIMITATIONS
from app.components.plots import plot_forecast_actual_vs_predicted, THEME_LAYOUT, COLOR_PRIMARY, COLOR_SUCCESS


def render_forecasting(
    plant_id: int,
    predictions_df: pd.DataFrame,
    metrics_table: pd.DataFrame,
    engine: SolarForecastEngine
) -> None:
    """Renders 7-day chronological forecast benchmarking, actual vs predicted curves, and limitations."""
    meta = PLANTS[plant_id]

    st.markdown(f"### 🤖 Machine Learning Forecasting Studio: {meta.name}")
    st.markdown(
        f"7-Day Day-Ahead Chronological Generation Forecast for **{meta.total_ac_capacity_mw:.2f} MW** fleet. "
        "Evaluated on a strict, out-of-time chronological holdout test set with **zero lookahead leakage**."
    )

    # Headline Model Evaluation Metrics Table
    st.markdown("#### 🏆 Benchmark Evaluation Matrix (7-Day Chronological Holdout)")
    st.markdown(
        "Standardized model performance evaluated against actual ground truth generation across **672 consecutive 15-minute intervals**:"
    )

    formatted_table = metrics_table.copy()
    formatted_table = formatted_table.rename(columns={
        "model_name": "Model Architecture",
        "mae_kw": "MAE (kW)",
        "rmse_kw": "RMSE (kW)",
        "r2_score": "R² Score",
        "nmae_pct": "nMAE (% Cap)",
        "nrmse_pct": "nRMSE (% Cap)",
        "wape_pct": "WAPE (%)",
        "sample_count": "Test Intervals"
    })
    st.dataframe(formatted_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Interactive Actual vs Predicted Plot
    st.markdown("#### 📈 Interactive 7-Day Holdout Forecast vs Actual Generation")

    model_options = [
        "Gradient Boosting",
        "Random Forest",
        "Linear Regression (Ridge)",
        "Baseline (Diurnal Climatology)",
        "Baseline (24h Persistence)"
    ]
    selected_models = st.multiselect(
        "Select Models to Display on Horizon Curve:",
        options=model_options,
        default=["Gradient Boosting", "Random Forest", "Baseline (Diurnal Climatology)"]
    )

    if selected_models:
        st.plotly_chart(
            plot_forecast_actual_vs_predicted(predictions_df, selected_models),
            use_container_width=True
        )
    else:
        st.warning("Please select at least one model to display predictions.")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 🎯 Daily Aggregated Forecast Energy Accuracy")
        # Compute daily actual vs predicted MWh over the 7 days
        pred_daily = predictions_df.groupby("DATE").agg(
            ACTUAL_MWH=("ACTUAL_KW", lambda x: (x * 0.25).sum() / 1000.0),
            PRED_GB_MWH=("PRED_GRADIENT_BOOSTING_KW", lambda x: (x * 0.25).sum() / 1000.0),
            PRED_RF_MWH=("PRED_RANDOM_FOREST_KW", lambda x: (x * 0.25).sum() / 1000.0),
            PRED_RIDGE_MWH=("PRED_RIDGE_KW", lambda x: (x * 0.25).sum() / 1000.0),
        ).reset_index()

        pred_daily["GB_DAILY_ERROR_PCT"] = (
            ((pred_daily["PRED_GB_MWH"] - pred_daily["ACTUAL_MWH"]) / pred_daily["ACTUAL_MWH"]) * 100.0
        ).round(2)

        st.dataframe(
            pred_daily.rename(columns={
                "DATE": "Forecast Date",
                "ACTUAL_MWH": "Actual (MWh)",
                "PRED_GB_MWH": "Gradient Boosting (MWh)",
                "PRED_RF_MWH": "Random Forest (MWh)",
                "GB_DAILY_ERROR_PCT": "GB Error (%)"
            }).round(2),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("#### 🌲 Feature Importance Ranking (Random Forest)")
        importances = engine.get_feature_importances()
        if not importances.empty:
            fig_imp = px.bar(
                importances.head(8),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top Predictive Drivers"
            )
            fig_imp.update_layout(**THEME_LAYOUT, height=320, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    # Operational Model Limitations
    st.markdown(MODEL_LIMITATIONS)
