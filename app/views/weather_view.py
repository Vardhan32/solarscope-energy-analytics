"""Weather Impact and Thermal Analysis View for SolarScope."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PLANTS
from src.analytics.metrics import calculate_daily_metrics
from src.analytics.weather import (
    analyze_weather_correlation,
    calculate_thermal_derating_curve,
    calculate_cell_heating_model,
    classify_sky_conditions,
)
from app.components.plots import (
    plot_weather_generation_scatter,
    plot_thermal_derating,
    THEME_LAYOUT,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
)


def render_weather(plant_id: int, plant_ts_df: pd.DataFrame) -> None:
    """Renders weather impact, solar irradiation sensitivity, and thermal loss analytics."""
    meta = PLANTS[plant_id]
    daily_df = calculate_daily_metrics(plant_ts_df, meta)

    corr_df = analyze_weather_correlation(plant_ts_df)
    derate_df = calculate_thermal_derating_curve(plant_ts_df)
    heating_model = calculate_cell_heating_model(plant_ts_df)
    sky_df = classify_sky_conditions(daily_df)

    st.markdown(f"### ⛅ Weather & Thermal Dynamics: {meta.name}")
    st.markdown(
        "Empirical evaluation of atmospheric conditions on solar photovoltaic energy conversion. "
        "Examines solar irradiance drive, cell temperature rise, and module thermal derating."
    )

    # Key Thermal Insights Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Thermal Heating Rate",
            f"{heating_model['thermal_heating_coefficient']:.1f} °C / (kW/m²)",
            help="Temperature increase above ambient per unit solar irradiation"
        )
    with c2:
        st.metric(
            "Model R² Fit",
            f"{heating_model['r2_score']:.3f}",
            help="Quality of Ross empirical cell heating fit"
        )
    with c3:
        st.metric(
            "Max Module Temp Rise",
            f"{heating_model['max_delta_t_deg_c']:.1f} °C",
            help="Peak difference between module cell and ambient temperature"
        )
    with c4:
        st.metric(
            "Clear Sky Days",
            f"{(sky_df['SKY_CONDITION'] == 'Clear Sky').sum()} / {len(sky_df)}",
            help="Days with Peak Sun Hours >= 6.5 kWh/m²"
        )

    st.markdown("---")

    # Irradiance vs Power Scatter
    st.plotly_chart(plot_weather_generation_scatter(plant_ts_df), use_container_width=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 🌡️ Temperature Derating Analysis")
        st.markdown(
            "As solar modules heat up past STC (25°C), semiconductor bandgap contraction reduces open-circuit voltage ($V_{oc}$), "
            "causing thermal power derating (~ -0.4% per °C). Below is the empirical fleet response across module temperature bins."
        )
        st.plotly_chart(plot_thermal_derating(derate_df), use_container_width=True)

    with col2:
        st.markdown("#### 🔗 Meteorological Correlation Matrix")
        st.markdown("Pearson correlation matrix between environmental drivers and electrical output:")

        fig_corr = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            title="Correlation Matrix (Daylight Hours)"
        )
        fig_corr.update_layout(**THEME_LAYOUT, height=360)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🌤️ Daily Sky Condition Breakdown")
    fig_sky = px.histogram(
        sky_df,
        x="SKY_CONDITION",
        color="SKY_CONDITION",
        color_discrete_map={
            "Clear Sky": "#10b981",
            "Partly Cloudy": "#f59e0b",
            "Overcast / Rainy": "#64748b"
        },
        title="Distribution of Monitoring Days by Solar Resource Classification",
        labels={"SKY_CONDITION": "Resource Category", "count": "Days"}
    )
    fig_sky.update_layout(**THEME_LAYOUT, height=320, showlegend=False)
    st.plotly_chart(fig_sky, use_container_width=True)
