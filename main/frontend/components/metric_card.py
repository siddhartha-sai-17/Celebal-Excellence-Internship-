"""
metric_card.py
──────────────
Styled metric card helpers for the Streamlit dark-theme UI.
Renders raw HTML via st.markdown(unsafe_allow_html=True).
"""

import streamlit as st


def metric_card(label: str, value: str, sub: str = "",
                color: str = "teal") -> None:
    """
    Render a single styled metric card.

    color options: "teal" | "blue" | "amber" | "red" | "purple"
    """
    color_map = {
        "teal":   "#1de9b6",
        "blue":   "#5baef0",
        "amber":  "#f5a623",
        "red":    "#e82127",
        "purple": "#9b5de5",
    }
    accent = color_map.get(color, "#1de9b6")

    st.markdown(f"""
    <div style="
        background:#0b1a27;
        border:1px solid #1a3048;
        border-top:2px solid {accent};
        border-radius:8px;
        padding:1.2rem 1.4rem;
        height:100%;
    ">
        <div style="font-size:0.68rem;letter-spacing:0.15em;
                    text-transform:uppercase;color:#4a7a9b;
                    margin-bottom:0.5rem;">{label}</div>
        <div style="font-size:1.8rem;font-weight:500;
                    color:{accent};line-height:1;">{value}</div>
        <div style="font-size:0.72rem;color:#3a6a8a;
                    margin-top:0.3rem;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_row(items: list) -> None:
    """
    items: list of dicts with keys: label, value, sub, color
    Renders all items side by side in equal-width columns.
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric_card(
                label=item.get("label", ""),
                value=item.get("value", ""),
                sub  =item.get("sub",   ""),
                color=item.get("color", "blue"),
            )


def result_banner(specialty: str, confidence: float) -> None:
    """Green banner showing predicted specialty + confidence."""
    conf_pct = f"{confidence*100:.1f}%"
    st.markdown(f"""
    <div style="
        background:#071f14;
        border:1px solid #0d4a30;
        border-left:4px solid #1de9b6;
        border-radius:8px;
        padding:1.4rem 1.8rem;
        margin:1.2rem 0;
        display:flex;align-items:center;gap:1.2rem;
    ">
        <div style="width:12px;height:12px;background:#1de9b6;
                    border-radius:50%;flex-shrink:0;
                    animation:pulse 1.5s infinite;"></div>
        <div>
            <div style="font-size:0.65rem;letter-spacing:0.2em;
                        text-transform:uppercase;color:#2aaa80;
                        margin-bottom:0.2rem;">Predicted Specialty</div>
            <div style="font-size:1.6rem;color:#1de9b6;
                        font-weight:300;">{specialty}</div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:0.65rem;letter-spacing:0.15em;
                        text-transform:uppercase;color:#2aaa80;
                        margin-bottom:0.2rem;">Confidence</div>
            <div style="font-size:2rem;font-weight:500;
                        color:#5baef0;">{conf_pct}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
