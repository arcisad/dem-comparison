import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from dem_comparison.utils import read_metrics


def plot_metrics(
    metric_files: list[Path],
    is_error: bool = True,
    polar: bool = False,
    save_path: Path | None = None,
    data_bounds: tuple | list[tuple] | None = (-50, 50),
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
    data_bounds: tuple | list[tuple] | None, optional
        Filters out the data beyond the given bounds (left and right inclusive), by default (-50, 50),
        If list of bounds provided the number of bounds should be the same as the number of metrics.
    Returns
    -------
    go.Figure
        Plotly figure
    """

    def filter_data(met, db):
        validity_list = [True if db[0] <= el <= db[1] else False for el in met]
        new_metric = [el for j, el in enumerate(met) if validity_list[j]]
        new_x = [el for j, el in enumerate(x) if validity_list[j]]
        new_y = [el for j, el in enumerate(y) if validity_list[j]]
        return new_x, new_y, new_metric

    labels = ["ME" if is_error else "MEAN", "STD", "MSE", "NMAD"]
    metrics, x, y = read_metrics(metric_files, numerical_axes=polar)

    if type(data_bounds) is list:
        assert len(data_bounds) == len(
            metrics
        ), "If used as a list, the number of data bounds should be the same as the number of metrics."

    # buttons to create
    buttons = [
        {
            "method": "restyle",
            "label": labels[i],
            "args": [
                {
                    "visible": [
                        el for el in [l == labels[i] for l in labels] for _ in range(2)
                    ]
                },
            ],
        }
        for i in range(len(metrics))
    ]

    # Add dropdown
    updatemenu = [{"buttons": buttons, "direction": "down", "showactive": True}]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar" if polar else "xy"}, {"type": "xy"}]],
        column_widths=[0.7, 0.3],
    )
    if not polar:
        fig.update_layout(xaxis_title="Lat", yaxis_title="Lon")
    for i, metric in enumerate(metrics):
        if data_bounds is not None:
            if type(data_bounds) is list:
                x, y, metric = filter_data(metric, data_bounds[i])
            else:
                x, y, metric = filter_data(metric, data_bounds)
        if polar:
            fig.add_trace(
                go.Scatterpolar(
                    r=x,
                    theta=y,
                    mode="markers",
                    marker=dict(color=metric, colorbar=dict(thickness=20)),
                    name="",
                    hovertemplate="<i>Lat</i>: %{theta}, "
                    + "<i>Lon</i>: %{r}"
                    + "<br></br>"
                    "<b>%{text}</b>",
                    text=[f"{labels[i]}: {j}" for j in metric],
                    visible=True if i == 0 else False,
                ),
                row=1,
                col=1,
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
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Histogram(
                x=metric,
                visible=True if i == 0 else False,
                showlegend=False,
            ),
            row=1,
            col=2,
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
