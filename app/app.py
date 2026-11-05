"""SolarScope – Solar Power Generation Analytics & Forecasting Platform.

Main Streamlit Application Entrypoint.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Setup python path to include repository root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    PLANTS,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    PlantMetadata,
)
from src.models.pipeline import run_full_pipeline
from src.models.forecasting import SolarForecastEngine
from app.views.overview import render_overview
from app.views.analytics import render_analytics
from app.views.plant_perf import render_plant_perf
from app.views.weather_view import render_weather
from app.views.anomalies_view import render_anomalies
from app.views.forecasting_view import render_forecasting
from app.views.sql_explorer import render_sql_explorer

# Page configuration
st.set_page_config(
    page_title="SolarScope | Solar Analytics & Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Top Header Accent */
    .solar-header {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0px;
    }
    
    .solar-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 6px 6px 0px 0px;
        color: #94a3b8;
        font-weight: 500;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(245, 158, 11, 0.15) !important;
        color: #f59e0b !important;
        border-bottom: 2px solid #f59e0b !important;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_cached_plant_data(plant_id: int):
    """Loads processed parquet data or triggers pipeline if missing."""
    inv_file = DATA_PROCESSED_DIR / f"plant_{plant_id}_inverter.parquet"
    ts_file = DATA_PROCESSED_DIR / f"plant_{plant_id}_timeseries.parquet"
    anom_file = DATA_PROCESSED_DIR / f"plant_{plant_id}_anomalies.parquet"
    pred_file = DATA_PROCESSED_DIR / f"plant_{plant_id}_forecast_predictions.parquet"

    if not (inv_file.exists() and ts_file.exists() and anom_file.exists() and pred_file.exists()):
        with st.spinner("Initializing SolarScope datasets and training ML models..."):
            run_full_pipeline()

    inverter_df = pd.read_parquet(inv_file)
    plant_ts_df = pd.read_parquet(ts_file)
    anomalies_df = pd.read_parquet(anom_file)
    predictions_df = pd.read_parquet(pred_file)

    return inverter_df, plant_ts_df, anomalies_df, predictions_df


@st.cache_resource(show_spinner=False)
def load_cached_engine(plant_id: int) -> SolarForecastEngine:
    """Loads serialized forecasting engine and models."""
    model_file = MODELS_DIR / f"plant_{plant_id}_engine.joblib"
    if not model_file.exists():
        run_full_pipeline()
    return joblib.load(model_file)


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:#f59e0b; margin-bottom:0;'>☀️ SolarScope</h2>", unsafe_allow_html=True)
        st.caption("Solar Power Generation Analytics & Forecasting Platform")
        st.markdown("---")

        # Plant Selector
        plant_options = {
            4135001: "Plant 1 (Gandhinagar - 31.02 MW)",
            4136001: "Plant 2 (Gandhinagar - 30.47 MW)",
        }
        selected_plant_id = st.selectbox(
            "📍 Select Solar Asset:",
            options=list(plant_options.keys()),
            format_func=lambda x: plant_options[x]
        )

        st.markdown("---")
        # Navigation
        views = [
            "Executive Overview",
            "Energy & Generation Analytics",
            "Inverter Fleet Benchmarking",
            "Weather & Thermal Dynamics",
            "Asset Health & Anomalies",
            "ML Forecasting Studio",
            "SQL Analytics Console"
        ]
        selected_view = st.radio("Navigate Module:", views)

        st.markdown("---")
        st.markdown("### ℹ️ Asset Specs")
        meta = PLANTS[selected_plant_id]
        st.write(f"**Capacity:** {meta.total_ac_capacity_mw:.2f} MW AC")
        st.write(f"**Inverters:** {meta.inverter_count} units")
        st.write(f"**Interval:** 15-minute telemetry")
        st.write(f"**Location:** {meta.location}")

        st.markdown("---")
        st.caption("Built for Renewable Energy Analysts & Grid Operators.")

    # Load Data
    inverter_df, plant_ts_df, anomalies_df, predictions_df = load_cached_plant_data(selected_plant_id)
    engine = load_cached_engine(selected_plant_id)

    # Also load the other plant's timeseries for side-by-side comparison in Overview
    other_plant_id = 4136001 if selected_plant_id == 4135001 else 4135001
    _, other_plant_ts_df, _, _ = load_cached_plant_data(other_plant_id)

    # Date Range Filter in Header (if applicable)
    min_date = plant_ts_df["DATE"].min()
    max_date = plant_ts_df["DATE"].max()

    # Route Views
    if selected_view == "Executive Overview":
        render_overview(selected_plant_id, inverter_df, plant_ts_df, other_plant_ts_df)
    elif selected_view == "Energy & Generation Analytics":
        render_analytics(selected_plant_id, plant_ts_df)
    elif selected_view == "Inverter Fleet Benchmarking":
        render_plant_perf(selected_plant_id, inverter_df)
    elif selected_view == "Weather & Thermal Dynamics":
        render_weather(selected_plant_id, plant_ts_df)
    elif selected_view == "Asset Health & Anomalies":
        render_anomalies(selected_plant_id, anomalies_df)
    elif selected_view == "ML Forecasting Studio":
        # Get metrics table from engine
        metrics_table = pd.DataFrame(engine.metrics).sort_values("mae_kw").reset_index(drop=True)
        render_forecasting(selected_plant_id, predictions_df, metrics_table, engine)
    elif selected_view == "SQL Analytics Console":
        render_sql_explorer()


if __name__ == "__main__":
    main()
