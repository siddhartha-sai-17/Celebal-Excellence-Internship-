"""
charts.py
─────────
Reusable Altair chart helpers for the Streamlit frontend.
All functions return an alt.Chart ready for st.altair_chart().
"""

import altair as alt
import pandas as pd

DARK_BG   = "#0b1a27"
GRID_COL  = "#0f2535"
LABEL_COL = "#3a6a8a"
TEXT_COL  = "#7a9ab5"
ACCENT    = "#1de9b6"
BLUE      = "#2a7fcf"
GOLD      = "#ffc940"
RED       = "#e82127"


def _base_config():
    return {
        "config": {
            "view":       {"strokeWidth": 0, "fill": DARK_BG},
            "axis":       {"labelColor": TEXT_COL, "titleColor": LABEL_COL,
                           "gridColor": GRID_COL, "domainColor": GRID_COL,
                           "tickColor": GRID_COL},
            "legend":     {"labelColor": TEXT_COL, "titleColor": LABEL_COL},
            "title":      {"color": TEXT_COL},
        }
    }


def deliveries_line(df: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title="Quarter"),
        tooltip=["date:T", "deliveries:Q", "production:Q"],
    )
    deliv = base.mark_line(color=RED, strokeWidth=2).encode(
        y=alt.Y("deliveries:Q", title="Units")
    )
    prod  = base.mark_line(color=BLUE, strokeWidth=1.5, strokeDash=[5, 3]).encode(
        y="production:Q"
    )
    return (deliv + prod).properties(
        title="Quarterly deliveries & production",
        height=280,
        background="transparent",
    ).configure(**_base_config()["config"])


def revenue_bar(df: pd.DataFrame) -> alt.Chart:
    df = df.copy()
    df["color"] = df["quarter"].apply(lambda q: RED if q == 4 else BLUE)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("date:T", title="Quarter"),
            y=alt.Y("revenue_bn:Q", title="Revenue ($B)"),
            color=alt.Color("quarter:N",
                            scale=alt.Scale(domain=[1, 2, 3, 4],
                                            range=[BLUE, BLUE, BLUE, RED]),
                            legend=None),
            tooltip=["date:T", "revenue_bn:Q"],
        )
        .properties(title="Quarterly revenue", height=240, background="transparent")
        .configure(**_base_config()["config"])
    )


def yoy_bar(df: pd.DataFrame) -> alt.Chart:
    df = df.dropna(subset=["delivery_growth_yoy"]).copy()
    df["color"] = df["delivery_growth_yoy"].apply(
        lambda v: ACCENT if v >= 0 else RED
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("date:T", title="Quarter"),
            y=alt.Y("delivery_growth_yoy:Q", title="YoY Growth (%)"),
            color=alt.condition(
                alt.datum.delivery_growth_yoy >= 0,
                alt.value(ACCENT), alt.value(RED)
            ),
            tooltip=["date:T",
                     alt.Tooltip("delivery_growth_yoy:Q", format=".1f")],
        )
        .properties(title="Year-over-year delivery growth", height=240,
                    background="transparent")
        .configure(**_base_config()["config"])
    )


def prob_distribution(prob_df: pd.DataFrame,
                      top_specialty: str) -> alt.Chart:
    return (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Probability:Q", axis=alt.Axis(format=".0%")),
            y=alt.Y("Specialty:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.Specialty == top_specialty,
                alt.value(ACCENT), alt.value(BLUE)
            ),
            tooltip=["Specialty",
                     alt.Tooltip("Probability:Q", format=".3f")],
        )
        .properties(height=280, background="transparent")
        .configure(**_base_config()["config"])
    )


def model_comparison_bar(metrics_df: pd.DataFrame) -> alt.Chart:
    melted = metrics_df.melt(
        id_vars="model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric", value_name="Value"
    )
    melted["Value_K"] = melted["Value"] / 1000
    return (
        alt.Chart(melted)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("model:N", title=None),
            y=alt.Y("Value_K:Q", title="Value (K units)"),
            color=alt.Color("Metric:N",
                            scale=alt.Scale(domain=["MAE", "RMSE"],
                                            range=[BLUE, ACCENT])),
            xOffset="Metric:N",
            tooltip=["model:N", "Metric:N",
                     alt.Tooltip("Value_K:Q", format=".1f")],
        )
        .properties(title="Model MAE & RMSE comparison",
                    height=300, background="transparent")
        .configure(**_base_config()["config"])
    )


def feature_importance_bar(fi_df: pd.DataFrame) -> alt.Chart:
    top = fi_df.head(12)
    return (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("importance:Q", title="Permutation importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.importance > 0,
                alt.value(ACCENT), alt.value(RED)
            ),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
        )
        .properties(title="Feature importances (top 12)",
                    height=340, background="transparent")
        .configure(**_base_config()["config"])
    )


def forecast_line(ts: pd.Series,
                  future_dates,
                  future_fc) -> alt.Chart:
    hist = pd.DataFrame({"date": ts.index, "value": ts.values / 1000,
                         "type": "Historical"})
    fwd  = pd.DataFrame({"date": future_dates,
                         "value": [v / 1000 for v in future_fc],
                         "type": "Forecast"})
    combined = pd.concat([hist, fwd], ignore_index=True)

    return (
        alt.Chart(combined)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Quarter"),
            y=alt.Y("value:Q", title="Deliveries (000s)"),
            color=alt.Color("type:N",
                            scale=alt.Scale(domain=["Historical", "Forecast"],
                                            range=[BLUE, ACCENT])),
            tooltip=["date:T", alt.Tooltip("value:Q", format=".1f"), "type"],
        )
        .properties(title="Holt-Winters delivery forecast",
                    height=300, background="transparent")
        .configure(**_base_config()["config"])
    )
