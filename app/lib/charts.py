"""One visual system for every chart in the app.

Colors come from a validated categorical palette, assigned in fixed order and
by entity (legit is always slot 1, fraud always slot 2), never by rank.
Gridlines and axes are kept recessive so the data carries the ink.
"""

from __future__ import annotations

import plotly.graph_objects as go

# Categorical slots in fixed order (validated for colour-vision deficiency on
# adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LEGIT = SERIES[0]
FRAUD = SERIES[1]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    """Apply the shared layout: recessive grid, system font, legend on top."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if legend else 8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        bargap=0.25,
    )
    fig.update_xaxes(
        showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK_2)
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=AXIS,
        tickfont=dict(color=MUTED),
        title_font=dict(color=INK_2),
    )
    return fig


PLOTLY_CONFIG = {"displayModeBar": False}
