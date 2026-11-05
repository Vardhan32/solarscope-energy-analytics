"""SolarScope KPI Component."""

import streamlit as st


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    delta: str = "",
    delta_color: str = "normal",  # "normal", "inverse", "off"
    icon: str = "⚡"
) -> None:
    """Renders a modern, responsive metric card with custom styling."""
    delta_html = ""
    if delta:
        color = "#10b981" if delta_color == "normal" else ("#ef4444" if delta_color == "inverse" else "#94a3b8")
        delta_html = f"<div style='color: {color}; font-size: 0.82rem; font-weight: 600; margin-top: 4px;'>{delta}</div>"

    card_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{title}</span>
            <span style="font-size: 1.2rem;">{icon}</span>
        </div>
        <div style="color: #f8fafc; font-size: 1.75rem; font-weight: 700; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            {value}
        </div>
        {delta_html}
        <div style="color: #64748b; font-size: 0.75rem; margin-top: 6px;">
            {subtitle}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
