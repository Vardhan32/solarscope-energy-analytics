"""Overview Dashboard View for SolarScope."""

import streamlit as st
import pandas as pd

from src.config import PLANTS
from src.analytics.metrics import calculate_daily_metrics, calculate_kpi_summary
from src.analytics.plant_perf import compare_plants
from app.components.kpi_cards import render_kpi_card
from app.components.plots import plot_generation_timeseries, plot_daily_generation_bar


def render_overview(
    plant_id: int,
    inverter_df: pd.DataFrame,
    plant_ts_df: pd.DataFrame,
    other_plant_ts_df: pd.DataFrame
) -> None:
    """Renders executive overview with KPIs, comparisons, and generation profiles."""
    meta = PLANTS[plant_id]
    other_plant_id = 4136001 if plant_id == 4135001 else 4135001
    other_meta = PLANTS[other_plant_id]

    daily_df = calculate_daily_metrics(plant_ts_df, meta)
    other_daily_df = calculate_daily_metrics(other_plant_ts_df, other_meta)
    kpis = calculate_kpi_summary(plant_ts_df, daily_df, meta)

    st.markdown(f"### ☀️ Executive Summary: {meta.name}")
    st.markdown(f"📍 **Location:** `{meta.location}` &nbsp;|&nbsp; ⚡ **Installed Capacity:** `{meta.total_ac_capacity_mw:.2f} MW AC` &nbsp;|&nbsp; 🔌 **Inverters:** `{meta.inverter_count} units`")

    # KPI Metric Cards Grid
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card(
            title="Total Energy",
            value=f"{kpis['total_energy_mwh']:.1f} MWh",
            subtitle=f"{kpis['monitoring_days']} monitoring days",
            icon="🔋"
        )
    with col2:
        render_kpi_card(
            title="Avg Daily Yield",
            value=f"{kpis['avg_daily_mwh']:.1f} MWh",
            subtitle="Per day fleet total",
            icon="📈"
        )
    with col3:
        render_kpi_card(
            title="Peak Generation",
            value=f"{kpis['peak_ac_kw'] / 1000.0:.2f} MW",
            subtitle=f"{(kpis['peak_ac_kw']/meta.total_ac_capacity_kw)*100:.1f}% of nameplate",
            icon="⚡"
        )
    with col4:
        render_kpi_card(
            title="Capacity Factor",
            value=f"{kpis['avg_capacity_factor_pct']:.1f}%",
            subtitle="Fleet 24h utilization",
            icon="🎯"
        )
    with col5:
        render_kpi_card(
            title="Performance Ratio",
            value=f"{kpis['avg_performance_ratio']:.3f}",
            subtitle="IEC 61724 standard",
            icon="💎"
        )

    st.markdown("---")

    # Time series plots
    st.plotly_chart(
        plot_generation_timeseries(
            plant_ts_df,
            title=f"{meta.name} – 15-Minute Generation Curve vs Theoretical Expected Power"
        ),
        use_container_width=True
    )

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(
            plot_daily_generation_bar(daily_df),
            use_container_width=True
        )

    with col_right:
        st.markdown("#### ⚖️ Plant 1 vs Plant 2 Benchmarking")
        st.markdown("Comparative operational performance between the two sister solar plants at Gandhinagar Solar Park.")
        
        comp_df = compare_plants(
            daily_df if plant_id == 4135001 else other_daily_df,
            other_daily_df if plant_id == 4135001 else daily_df
        )
        st.dataframe(
            comp_df,
            use_container_width=True,
            hide_index=True
        )
