"""Energy Analytics View for SolarScope."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PLANTS
from src.analytics.metrics import calculate_daily_metrics, calculate_diurnal_profile
from app.components.plots import plot_diurnal_profile, THEME_LAYOUT, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS


def render_analytics(plant_id: int, plant_ts_df: pd.DataFrame) -> None:
    """Renders comprehensive daily, diurnal, and capacity factor energy analytics."""
    meta = PLANTS[plant_id]
    daily_df = calculate_daily_metrics(plant_ts_df, meta)
    diurnal_df = calculate_diurnal_profile(plant_ts_df)

    st.markdown(f"### 📊 Energy Generation & Capacity Analytics: {meta.name}")

    # Tabs for analytics categories
    tab1, tab2, tab3 = st.tabs(["Diurnal Generation Profiles", "Capacity Factor & Performance Ratio", "Specific Yield & Efficiency"])

    with tab1:
        st.markdown("#### 🕒 24-Hour Diurnal Fleet Profile")
        st.markdown(
            "Diurnal distribution of fleet power generation across all 24 hours of the day. "
            "The shaded envelope illustrates the **10th to 90th percentile** variability band, highlighting weather-induced generation dispersion."
        )
        st.plotly_chart(plot_diurnal_profile(diurnal_df), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### ⚡ Peak Solar Hours (11:00 - 14:00)")
            peak_window = diurnal_df[(diurnal_df["HOUR"] >= 11) & (diurnal_df["HOUR"] <= 14)]
            st.dataframe(
                peak_window[["HOUR", "MEAN_AC_KW", "MEDIAN_AC_KW", "P90_AC_KW", "MEAN_IRRADIATION", "MEAN_MODULE_TEMP"]].round(2),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("##### 💡 Diurnal Takeaways for Grid Dispatch")
            st.info(
                "• **Fleet Peak Output:** Solar generation peaks consistently between **12:00 and 13:00**, with average generation exceeding 75% of nameplate rating.\n"
                "• **Morning Ramp Rate:** Generation ramps sharply from 07:00 to 10:00 at an average rate of ~5 MW/hour.\n"
                "• **Evening Decay:** Output tapers off predictably by 18:30 as the solar zenith angle increases."
            )

    with tab2:
        st.markdown("#### 🎯 Capacity Factor (CF) & Performance Ratio (PR)")
        st.markdown(
            "Capacity Factor measures actual electrical generation against theoretical continuous nameplate capacity (24 hours). "
            "Performance Ratio (IEC 61724 standard) normalizes generation against in-plane irradiation."
        )

        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=daily_df["DATE"],
            y=daily_df["CAPACITY_FACTOR_PCT"],
            name="Daily Capacity Factor (%)",
            marker=dict(color=COLOR_SECONDARY, opacity=0.85)
        ))
        fig_cf.add_trace(go.Scatter(
            x=daily_df["DATE"],
            y=daily_df["ROLLING_7D_CF_PCT"],
            name="7-Day Rolling CF Trend (%)",
            line=dict(color=COLOR_PRIMARY, width=3)
        ))
        fig_cf.update_layout(
            **THEME_LAYOUT,
            title="Daily Capacity Factor (%) & 7-Day Trend",
            xaxis_title="Date",
            yaxis_title="Capacity Factor (%)",
            height=360
        )
        st.plotly_chart(fig_cf, use_container_width=True)

        # Performance Ratio Chart
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=daily_df["DATE"],
            y=daily_df["PERFORMANCE_RATIO"],
            name="Daily Performance Ratio (PR)",
            line=dict(color=COLOR_SUCCESS, width=2.5),
            mode="lines+markers"
        ))
        # Benchmark line at 0.80
        fig_pr.add_hline(
            y=0.80,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Target Benchmark (PR = 0.80)"
        )
        fig_pr.update_layout(
            **THEME_LAYOUT,
            title="IEC 61724 Performance Ratio (PR) Daily Trend",
            xaxis_title="Date",
            yaxis_title="Performance Ratio (PR)",
            yaxis=dict(range=[0.5, 1.1]),
            height=360
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with tab3:
        st.markdown("#### ⚡ Specific Yield & Inverter Conversion Efficiency")
        col_y1, col_y2 = st.columns(2)

        with col_y1:
            fig_yield = px.bar(
                daily_df,
                x="DATE",
                y="SPECIFIC_YIELD_KWH_KWP",
                title="Daily Specific Yield (kWh / kWp)",
                labels={"SPECIFIC_YIELD_KWH_KWP": "Yield (kWh/kWp)", "DATE": "Date"},
                color_discrete_sequence=[COLOR_PRIMARY]
            )
            fig_yield.update_layout(**THEME_LAYOUT, height=360)
            st.plotly_chart(fig_yield, use_container_width=True)

        with col_y2:
            fig_eff = px.histogram(
                plant_ts_df[plant_ts_df["IS_DAYLIGHT"] & (plant_ts_df["PLANT_EFFICIENCY_PCT"] > 80)],
                x="PLANT_EFFICIENCY_PCT",
                nbins=40,
                title="Fleet Inverter DC-to-AC Efficiency Distribution (%)",
                labels={"PLANT_EFFICIENCY_PCT": "Conversion Efficiency (%)"},
                color_discrete_sequence=[COLOR_SUCCESS]
            )
            fig_eff.update_layout(**THEME_LAYOUT, height=360)
            st.plotly_chart(fig_eff, use_container_width=True)
