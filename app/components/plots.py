"""SolarScope Interactive Plotly Visualization Components."""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# SolarScope consistent energy theme palette
COLOR_PRIMARY = "#f59e0b"    # Solar Amber
COLOR_SECONDARY = "#06b6d4"  # Solar Cyan
COLOR_SUCCESS = "#10b981"    # Emerald
COLOR_DANGER = "#ef4444"     # Rose Red
COLOR_PURPLE = "#a855f7"     # Violet
COLOR_BG = "#0f172a"         # Slate Dark
COLOR_GRID = "#334155"       # Slate Muted

THEME_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.4)",
    font=dict(color="#f1f5f9", family="Inter, system-ui, sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
    legend=dict(bgcolor="rgba(15, 23, 42, 0.6)", bordercolor="rgba(255,255,255,0.1)"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def plot_generation_timeseries(
    df: pd.DataFrame,
    title: str = "Plant Power Generation (15-min Intervals)",
    show_expected: bool = True
) -> go.Figure:
    """Renders 15-minute aggregate power generation curves."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["DATE_TIME"],
        y=df["AC_POWER_KW"],
        name="Actual AC Power (kW)",
        line=dict(color=COLOR_PRIMARY, width=2),
        fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.15)"
    ))

    if show_expected and "EXPECTED_AC_POWER_KW" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["DATE_TIME"],
            y=df["EXPECTED_AC_POWER_KW"],
            name="Expected AC Power (STC Derated)",
            line=dict(color=COLOR_SECONDARY, width=1.5, dash="dot")
        ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Timestamp",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        height=420
    )
    return fig


def plot_daily_generation_bar(daily_df: pd.DataFrame) -> go.Figure:
    """Renders daily generation bar chart with rolling 7-day average."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=daily_df["DATE"],
        y=daily_df["TOTAL_AC_ENERGY_KWH"] / 1000.0,
        name="Daily Generation (MWh)",
        marker=dict(color=COLOR_PRIMARY, opacity=0.85)
    ))

    if "ROLLING_7D_AC_KWH" in daily_df.columns:
        fig.add_trace(go.Scatter(
            x=daily_df["DATE"],
            y=daily_df["ROLLING_7D_AC_KWH"] / 1000.0,
            name="7-Day Moving Average",
            line=dict(color=COLOR_SUCCESS, width=2.5)
        ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(text="Daily Electricity Generation (MWh) & 7-Day Trend", font=dict(size=15)),
        xaxis_title="Date",
        yaxis_title="Energy (MWh)",
        hovermode="x unified",
        height=380
    )
    return fig


def plot_diurnal_profile(diurnal_df: pd.DataFrame) -> go.Figure:
    """Renders 24-hour diurnal curve with 10th-90th percentile confidence bounds."""
    fig = go.Figure()

    # 10th-90th percentile confidence ribbon
    fig.add_trace(go.Scatter(
        x=diurnal_df["HOUR"],
        y=diurnal_df["P90_AC_KW"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=diurnal_df["HOUR"],
        y=diurnal_df["P10_AC_KW"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(245, 158, 11, 0.15)",
        name="10th - 90th Percentile Range"
    ))

    # Mean Profile
    fig.add_trace(go.Scatter(
        x=diurnal_df["HOUR"],
        y=diurnal_df["MEAN_AC_KW"],
        name="Mean Fleet AC Power",
        line=dict(color=COLOR_PRIMARY, width=3)
    ))

    # Median Profile
    fig.add_trace(go.Scatter(
        x=diurnal_df["HOUR"],
        y=diurnal_df["MEDIAN_AC_KW"],
        name="Median Fleet AC Power",
        line=dict(color=COLOR_SECONDARY, width=2, dash="dash")
    ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(text="Diurnal Generation Profile (24-Hour Profile & Variability)", font=dict(size=15)),
        xaxis=dict(tickmode="linear", tick0=0, dtick=2, title="Hour of Day (0-23)"),
        yaxis_title="Fleet Power (kW)",
        hovermode="x unified",
        height=400
    )
    return fig


def plot_inverter_rankings(bench_df: pd.DataFrame) -> go.Figure:
    """Renders inverter ranking bar chart with health status colors."""
    status_colors = {
        "High Performer": COLOR_SUCCESS,
        "Normal": COLOR_SECONDARY,
        "Underperforming": COLOR_PRIMARY,
        "Critical Underperformance": COLOR_DANGER
    }

    fig = go.Figure()
    for status, color in status_colors.items():
        subset = bench_df[bench_df["HEALTH_STATUS"] == status]
        if not subset.empty:
            fig.add_trace(go.Bar(
                x=subset["SOURCE_KEY"],
                y=subset["TOTAL_ENERGY_KWH"] / 1000.0,
                name=status,
                marker=dict(color=color)
            ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(text="Inverter Fleet Energy Yield Ranking (MWh)", font=dict(size=15)),
        xaxis_title="Inverter Key",
        yaxis_title="Total Energy (MWh)",
        barmode="group",
        height=420
    )
    return fig


def plot_weather_generation_scatter(df: pd.DataFrame) -> go.Figure:
    """Renders Irradiance vs AC Power scatter colored by Module Temperature."""
    daylight = df[df["IS_DAYLIGHT"]].copy()
    if len(daylight) > 3000:
        daylight = daylight.sample(3000, random_state=42)

    fig = px.scatter(
        daylight,
        x="IRRADIATION",
        y="AC_POWER_KW",
        color="MODULE_TEMPERATURE",
        color_continuous_scale="Viridis",
        labels={
            "IRRADIATION": "Solar Irradiation (kW/m²)",
            "AC_POWER_KW": "Fleet AC Power (kW)",
            "MODULE_TEMPERATURE": "Module Temp (°C)"
        },
        title="Solar Irradiation vs AC Power (Color: Module Temperature °C)"
    )

    fig.update_layout(
        **THEME_LAYOUT,
        height=450
    )
    return fig


def plot_thermal_derating(derate_df: pd.DataFrame) -> go.Figure:
    """Renders thermal efficiency derating curve across temperature bins."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=derate_df["TEMP_BIN"].astype(str),
        y=derate_df["NORMALIZED_KW_PER_IRRAD"],
        name="Generation Intensity (kW / [kW/m²])",
        marker=dict(color=COLOR_SECONDARY, opacity=0.85)
    ))

    fig.add_trace(go.Scatter(
        x=derate_df["TEMP_BIN"].astype(str),
        y=derate_df["MEAN_EFFICIENCY_PCT"],
        name="Inverter Conversion Efficiency (%)",
        yaxis="y2",
        line=dict(color=COLOR_PRIMARY, width=3)
    ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(text="Thermal Derating: Output Intensity & Efficiency vs Module Temperature", font=dict(size=15)),
        xaxis_title="Module Temperature Range (°C)",
        yaxis=dict(title="Output per unit Irradiance (kW / [kW/m²])"),
        yaxis2=dict(
            title="Inverter Efficiency (%)",
            overlaying="y",
            side="right",
            range=[85, 100],
            showgrid=False
        ),
        height=400
    )
    return fig


def plot_forecast_actual_vs_predicted(
    pred_df: pd.DataFrame,
    selected_models: List[str]
) -> go.Figure:
    """Renders 7-day chronological actual vs predicted generation curves."""
    fig = go.Figure()

    # Actual Ground Truth
    fig.add_trace(go.Scatter(
        x=pred_df["DATE_TIME"],
        y=pred_df["ACTUAL_KW"],
        name="Actual Generation (kW)",
        line=dict(color="#ffffff", width=2.5),
    ))

    model_config = {
        "Gradient Boosting": ("PRED_GRADIENT_BOOSTING_KW", COLOR_SUCCESS, "solid"),
        "Random Forest": ("PRED_RANDOM_FOREST_KW", COLOR_PRIMARY, "solid"),
        "Linear Regression (Ridge)": ("PRED_RIDGE_KW", COLOR_PURPLE, "dash"),
        "Baseline (Diurnal Climatology)": ("PRED_CLIMATOLOGY_KW", COLOR_SECONDARY, "dot"),
        "Baseline (24h Persistence)": ("PRED_PERSISTENCE_KW", "#94a3b8", "dashdot"),
    }

    for model_name in selected_models:
        if model_name in model_config:
            col, color, dash = model_config[model_name]
            if col in pred_df.columns:
                fig.add_trace(go.Scatter(
                    x=pred_df["DATE_TIME"],
                    y=pred_df[col],
                    name=f"Predicted ({model_name})",
                    line=dict(color=color, width=2, dash=dash)
                ))

    fig.update_layout(
        **THEME_LAYOUT,
        title=dict(text="7-Day Holdout Forecast: Actual vs Predicted Power (15-min Intervals)", font=dict(size=15)),
        xaxis_title="Forecast Horizon Timestamp",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        height=450
    )
    return fig


def plot_anomaly_timeline(anomalies_df: pd.DataFrame) -> go.Figure:
    """Renders anomaly events categorized by severity over time."""
    if anomalies_df.empty:
        fig = go.Figure()
        fig.update_layout(**THEME_LAYOUT, title="No Anomalies Detected in Selected Range")
        return fig

    severity_colors = {
        "Critical": COLOR_DANGER,
        "Warning": COLOR_PRIMARY,
        "Info": COLOR_SECONDARY,
    }

    fig = px.strip(
        anomalies_df,
        x="DATE_TIME",
        y="ANOMALY_TYPE",
        color="SEVERITY",
        color_discrete_map=severity_colors,
        hover_data=["SOURCE_KEY", "AC_POWER_KW", "IRRADIATION"],
        title="Detected Operational Anomalies Timeline"
    )

    fig.update_layout(
        **THEME_LAYOUT,
        height=380,
        xaxis_title="Timestamp",
        yaxis_title="Anomaly Category"
    )
    return fig
