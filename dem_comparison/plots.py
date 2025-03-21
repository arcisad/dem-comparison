import plotly.graph_objects as go
import pickle
from pathlib import Path


def plot_metrics(
    metric_files: list[Path],
    is_error: bool = True,
) -> go.Figure:
    """Plots the metrics for the DEMS as an interactive plot

    Parameters
    ----------
    metric_files : list[Path]
        List of pickle files with the metric values stored in them.
    is_error : bool, optional
        If the data is an error(DEMs difference) array or a single DEM, by default True

    Returns
    -------
    go.Figure
        Plotly figure
    """
    data = []
    for pkl in metric_files:
        with open(pkl, "rb") as f:
            data.append(pickle.load(f))
    x = []
    y = []
    for pkl in metric_files:
        parts = str(Path(pkl).stem).split("_")
        x.append(parts[3])
        y.append(parts[4])

    mae = [i[0] for i in data]
    std = [i[1] for i in data]
    mse = [i[2] for i in data]
    nmad = [i[3] for i in data]
    labels = ["MAE" if is_error else "MEAN", "STD", "MSE", "NMAD"]
    metrics = [mae, std, mse, nmad]
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

    return fig
