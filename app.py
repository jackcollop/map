"""Interactive Dash app for the Cotton On-Call report.

Replaces the static figures in plot.py with an interactive dashboard:
  * View toggle: seasonality (faceted by futures month) vs marketing-year totals
  * Metric selector: Net / Sales / Purchases / Open futures
  * Filters: min futures year, |DTE| limit, month selection
  * Highlight a chosen year across all facets

Run:  python app.py   then open http://127.0.0.1:8050
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cotton_on_call_history.csv")

TEMPLATE = "plotly_dark"
MONTH_ORDER = ["March", "May", "July", "December"]  # October dropped
METRICS = ["Net", "Sales", "Purchases", "Open Futures"]


def _to_int(series):
    """Contract counts are strings that may carry a trailing '*'."""
    return series.astype(str).str.strip("*").str.replace(",", "", regex=False)\
        .replace({"": np.nan, "nan": np.nan}).astype(float)


def load_data():
    oc = pd.read_csv(CSV_PATH, parse_dates=True, index_col=0)
    oc = oc.dropna(subset=["Futures Month", "Futures Year", "MY"])
    oc = oc[oc["Futures Month"] != "October"]  # October contracts excluded

    oc["Sales"] = _to_int(oc["Unfixed Call Sales (Contracts)"])
    oc["Purchases"] = _to_int(oc["Unfixed Call Purchases (Contracts)"])
    oc["Open Futures"] = _to_int(oc["Open Futures Contracts at Close"])
    oc["Net"] = oc["Sales"] - oc["Purchases"]
    oc = oc.dropna(subset=["Sales", "Purchases"])

    oc["Futures Year"] = oc["Futures Year"].astype(int)
    oc["MY"] = oc["MY"].astype(int)
    # plot.py flips DTE so pre-expiry days are negative -> approaching expiry
    oc["DTE"] = -oc["DTE"]
    oc.index.name = "As Of Date"
    return oc


OC = load_data()
YEARS = sorted(OC["Futures Year"].unique())
MYS = sorted(int(y) for y in OC["MY"].unique())
MONTHS = [m for m in MONTH_ORDER if m in OC["Futures Month"].unique()]

app = Dash(__name__, title="Cotton On-Call")

# ----------------------------------------------------------------------------- layout
_control = {"marginBottom": "18px"}
_label = {"fontSize": "12px", "textTransform": "uppercase",
          "letterSpacing": "0.08em", "color": "#9aa0a6", "marginBottom": "6px"}

app.layout = html.Div(
    style={"backgroundColor": "#111", "color": "#e8eaed", "minHeight": "100vh",
           "fontFamily": "Segoe UI, Roboto, sans-serif", "display": "flex"},
    children=[
        # ---- sidebar
        html.Div(
            style={"width": "270px", "padding": "24px", "backgroundColor": "#1b1b1b",
                   "boxSizing": "border-box"},
            children=[
                html.H2("Cotton On-Call", style={"marginTop": 0}),
                html.Div(style=_control, children=[
                    html.Div("View", style=_label),
                    dcc.RadioItems(
                        id="view", value="seasonality",
                        options=[{"label": " Seasonality by month", "value": "seasonality"},
                                 {"label": " Marketing-year totals", "value": "my"}],
                        labelStyle={"display": "block", "marginBottom": "4px"}),
                ]),
                html.Div(style=_control, children=[
                    html.Div("Metric", style=_label),
                    dcc.Dropdown(id="metric", value="Net", clearable=False,
                                 options=[{"label": m, "value": m} for m in METRICS]),
                ]),
                html.Div(style=_control, children=[
                    html.Div("Highlight marketing year", style=_label),
                    dcc.Dropdown(id="highlight", value=2026, clearable=True,
                                 placeholder="None",
                                 options=[{"label": str(y), "value": y} for y in MYS]),
                ]),
                html.Div(style=_control, children=[
                    html.Div(id="minyear-label", style=_label),
                    dcc.Slider(id="minyear", min=min(MYS), max=max(MYS),
                               step=1, value=2003,
                               marks={int(y): {"label": f"'{str(y)[2:]}"}
                                      for y in MYS[::2]},
                               tooltip={"placement": "bottom"}),
                ]),
                html.Div(style=_control, children=[
                    html.Div(id="dte-label", style=_label),
                    dcc.Slider(id="dte", min=50, max=800, step=50, value=800,
                               marks={v: str(v) for v in range(100, 801, 200)},
                               tooltip={"placement": "bottom"}),
                ]),
                html.Div(id="month-block", style=_control, children=[
                    html.Div("Months", style=_label),
                    dcc.Checklist(id="months", value=MONTHS,
                                  options=[{"label": " " + m, "value": m} for m in MONTHS],
                                  labelStyle={"display": "block", "marginBottom": "2px"}),
                ]),
            ],
        ),
        # ---- main panel
        html.Div(style={"flex": 1, "padding": "16px"}, children=[
            dcc.Graph(id="graph", style={"height": "88vh"},
                      config={"displayModeBar": True, "scrollZoom": True}),
        ]),
    ],
)


# ----------------------------------------------------------------------------- helpers
def _apply_highlight(fig, metric, highlight):
    """Emphasise the highlighted year's trace(s), fade the rest."""
    if highlight is None:
        fig.for_each_trace(lambda t: t.update(line_width=1.8, opacity=0.85))
        return fig

    def restyle(tr):
        try:
            is_target = int(float(tr.name)) == int(highlight)
        except (TypeError, ValueError):
            is_target = False
        if is_target:
            tr.update(line_width=4, opacity=1)
        else:
            tr.update(line_width=1.5, opacity=0.35)

    fig.for_each_trace(restyle)
    return fig


def build_seasonality(metric, highlight, min_year, dte_limit, months):
    df = OC[(OC["MY"] >= min_year)
            & (OC["DTE"].abs() < dte_limit)
            & (OC["Futures Month"].isin(months))].sort_index()
    if df.empty:
        return px.line(template=TEMPLATE, title="No data for the current filters")

    df = df.copy()
    df["MY"] = df["MY"].astype(str)
    cols = [m for m in months if m in df["Futures Month"].unique()]
    fig = px.line(
        df, x="DTE", y=metric, color="MY",
        facet_col="Futures Month",
        facet_col_wrap=min(len(cols), 2) or 1,  # 2 columns -> 2x2 grid for 4 months
        category_orders={"Futures Month": cols,
                         "MY": [str(y) for y in MYS]},
        template=TEMPLATE,
        hover_data={"Futures Year": True, "Futures Month": True},
        labels={"DTE": "Days to expiry", metric: f"{metric} (contracts)",
                "MY": "Marketing year"},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=40),
                      legend_title_text="Marketing year",
                      title=f"{metric} vs days-to-expiry, by futures month")
    return _apply_highlight(fig, metric, highlight)


def build_marketing_year(metric, highlight, min_year, dte_limit):
    base = OC[OC["MY"] >= min_year]
    my_sum = base.groupby([base.index, "MY"])[metric].sum().reset_index()
    my_sum = my_sum.rename(columns={"level_0": "As Of Date"})
    date_col = my_sum.columns[0]
    # each marketing year (Aug-Jul) terminates with the July (MY+1) contract
    my_end = pd.to_datetime((my_sum["MY"] + 1).astype(str) + "-07-01")
    my_sum["DTE"] = (my_sum[date_col] - my_end).dt.days
    my_sum = my_sum[my_sum["DTE"].abs() < dte_limit]
    my_sum["MY"] = my_sum["MY"].astype(str)
    my_sum = my_sum.sort_values(["MY", "DTE"])
    if my_sum.empty:
        return px.line(template=TEMPLATE, title="No data for the current filters")

    fig = px.line(
        my_sum, x="DTE", y=metric, color="MY", template=TEMPLATE,
        labels={"DTE": "Days to MY end (Jul)", metric: f"{metric} (contracts)",
                "MY": "Marketing year"},
    )
    fig.update_layout(margin=dict(l=50, r=20, t=50, b=40),
                      title=f"Marketing-year {metric}: total across contracts vs days-to-MY-end")
    return _apply_highlight(fig, metric, highlight)


# ----------------------------------------------------------------------------- callbacks
@app.callback(
    Output("graph", "figure"),
    Output("month-block", "style"),
    Output("minyear-label", "children"),
    Output("dte-label", "children"),
    Input("view", "value"),
    Input("metric", "value"),
    Input("highlight", "value"),
    Input("minyear", "value"),
    Input("dte", "value"),
    Input("months", "value"),
)
def update(view, metric, highlight, min_year, dte_limit, months):
    minyear_label = f"Min marketing year: {min_year}"
    dte_label = f"|DTE| limit: {dte_limit} days"

    if view == "my":
        fig = build_marketing_year(metric, highlight, min_year, dte_limit)
        month_style = {"display": "none"}
    else:
        fig = build_seasonality(metric, highlight, min_year, dte_limit, months or MONTHS)
        month_style = {"marginBottom": "18px"}

    return fig, month_style, minyear_label, dte_label


if __name__ == "__main__":
    app.run(debug=True)
