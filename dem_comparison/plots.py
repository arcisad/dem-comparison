import plotly.graph_objects as go
import pickle
from pathlib import Path
from dem_comparison.utils import read_metrics


def plot_metrics(
    metric_files: list[Path],
    is_error: bool = True,
    polar: bool = False,
    save_path: Path | None = None,
) -> go.Figure:
    """Plots the metrics for the DEMS as an interactive plot

    Parameters
    ----------
    metric_files : list[Path]
        List of pickle files with the metric values stored in them.
    is_error : bool, optional
        If the data is an error(DEMs difference) array or a single DEM, by default True
    polar: bool, optional
        Polar plot, by default False
    save_path: Path | None, optional,
        Path to save the html plot, by default None

    Returns
    -------
    go.Figure
        Plotly figure
    """

    labels = ["ME" if is_error else "MEAN", "STD", "MSE", "NMAD"]
    metrics, x, y = read_metrics(metric_files, numerical_axes=polar)
    # buttons to create
    buttons = [
        {
            "method": "restyle",
            "label": labels[i],
            "args": [
                {"visible": [l == labels[i] for l in labels]},
                {
                    "z": [metric],
                },
            ],
        }
        for i, metric in enumerate(metrics)
    ]

    # Add dropdown
    updatemenu = [{"buttons": buttons, "direction": "down", "showactive": True}]

    fig = go.Figure()
    fig.update_layout(xaxis_title="Lat", yaxis_title="Lon")
    for i, metric in enumerate(metrics):
        if polar:
            fig.add_trace(
                go.Scatterpolar(
                    r=x,
                    theta=y,
                    mode="markers",
                    marker=dict(color=metric, colorbar=dict(thickness=20)),
                    name="",
                    hovertext=metric,
                    visible=True if i == 0 else False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=metric, colorbar=dict(thickness=20)),
                    name="",
                    hovertext=metric,
                    visible=True if i == 0 else False,
                )
            )

    fig.update_layout(updatemenus=updatemenu)
    if polar:
        polar_layout = {
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickvals": [0, 45, 90, 135, 180, 225, 270, 315],
                "ticktext": ["0", "45", "90", "135", "180", "-135", "-90", "-45"],
                "tickmode": "array",
            },
            "radialaxis": {"range": [90, 50]},
        }
        fig.update_layout(polar=polar_layout)

    if save_path:
        fig.write_html(save_path)

    return fig
