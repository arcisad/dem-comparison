import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from dem_comparison.utils import read_metrics
import numpy as np
import rasterio as rio
from dem_comparison.utils import enhance_image, get_cross_section_data, bin_metrics
from PIL import Image
import matplotlib.pyplot as plt
from shapely import Polygon
import shutil


def plot_metrics(
    metric_files: list[Path],
    is_error: bool = True,
    polar: bool = False,
    save_path: Path | None = None,
    data_bounds: tuple | list[tuple] | None = (-50, 50),
    num_bins: int | None = 15,
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
        color = metric
        hover_text = metric
        if num_bins is not None:
            bin_vals, bin_edges, bin_step = bin_metrics(metric, num_bins=num_bins)
            color = bin_vals
            hover_text = []
            for bv in bin_vals:
                bin_start = np.round(bv - bin_step / 2, 2)
                bin_end = np.round(bv + bin_step / 2, 2)
                if bin_start == bin_end:
                    hover_text.append(f"{bin_start}")
                else:
                    hover_text.append(f"{bin_start} to {bin_end}")
        if polar:
            fig.add_trace(
                go.Scatterpolar(
                    r=x,
                    theta=y,
                    mode="markers",
                    marker=dict(
                        color=color,
                        colorbar=dict(thickness=20),
                        colorscale="temps_r",
                    ),
                    name="",
                    hovertemplate="<i>Lat</i>: %{r}°, "
                    + "<i>Lon</i>: %{theta}"
                    + "<br></br>"
                    "<b>%{text}</b>",
                    text=[f"{labels[i]}: {j}" for j in hover_text],
                    visible=True if i == 0 else False,
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=sorted(x),
                    y=sorted(y),
                    mode="markers",
                    marker=dict(
                        color=color,
                        colorbar=dict(thickness=20),
                        colorscale="temps_r",
                    ),
                    name="",
                    hovertext=hover_text,
                    visible=True if i == 0 else False,
                ),
                row=1,
                col=1,
            )
        if num_bins is None:
            fig.add_trace(
                go.Histogram(
                    x=metric,
                    y=y,
                    visible=True if i == 0 else False,
                    showlegend=False,
                    name=labels[i],
                    hovertemplate="<i>Bin range</i>: %{x}"
                    + "<br></br>"
                    + "<b>Freq: %{y}</b>",
                ),
                row=1,
                col=2,
            )
        else:
            hover_text_hist = [
                f"{np.round(bv, 2)}-{np.round(bv + bin_step, 2)}"
                for bv in bin_edges[:-1]
            ]
            fig.add_trace(
                go.Bar(
                    x=bin_edges,
                    y=np.histogram(metric, num_bins)[0],
                    visible=True if i == 0 else False,
                    showlegend=False,
                    name=labels[i],
                    hovertemplate="<i>Bin range</i>: %{customdata}"
                    + "<br></br>"
                    + "<b>Freq: %{y}</b>",
                    customdata=hover_text_hist,
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
    fig.update_layout(bargap=0)

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_cross_sections(
    raster: Path | list[Path],
    bounds_poly: Polygon,
    diff_raster: Path,
    dist_step: int = 500,
    average_steps: list[int] = [5, 15, 30, 60, 100],
    temp_path: Path = Path("TMP"),
    raster_names: list[str] | None = None,
    raster_colours: list[str] | None = None,
    save_path: Path | None = None,
    diff_opacity: float = 1.0,
):
    if type(raster) is not list:
        raster = [raster]

    if raster_names is None:
        raster_names = [f"Raster_{i}" for i in range(len(raster))]

    if raster_colours is None:
        raster_colours = ["black"] * len(raster)

    plot_image = raster[0]
    if diff_raster:
        plot_image = diff_raster
    with rio.open(plot_image) as ds:
        transform = ds.transform
        img = enhance_image(ds.read(1))
        imh, imw, _ = img.shape
        if not temp_path.exists():
            temp_path.mkdir(parents=True, exist_ok=True)
            plt.imsave(temp_path / "temp_img.jpg", img)

    av_step_names = ["Original"] + ["Smoothing: " + str(i) for i in average_steps]
    average_steps = [None] + average_steps
    # buttons to create
    buttons = [
        {
            "method": "restyle",
            "label": av_step_names[i],
            "args": [
                {
                    "visible": len(raster)
                    * [
                        el
                        for el in [l == av_step_names[i] for l in av_step_names]
                        for _ in range(2)
                    ]
                    + [
                        el
                        for el in [l == av_step_names[i] for l in av_step_names]
                        for _ in range(2 if diff_opacity == 0 else 4)
                    ]
                    + [True]
                },
            ],
        }
        for i in range(len(av_step_names))
    ]

    updatemenu = [{"buttons": buttons, "direction": "down", "showactive": True}]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy", "secondary_y": True}, {"type": "xy", "rowspan": 2}],
            [{"type": "xy", "secondary_y": True}, None],
        ],
        column_widths=[0.7, 0.3],
    )

    vals_list_windows_per_raster = []
    for r in raster:
        vals_list_windows = []
        for w in average_steps:
            vals_list_windows.append(
                get_cross_section_data(r, bounds_poly, dist_step, w)
            )
        vals_list_windows_per_raster.append(vals_list_windows)

    diff_vals_list_windows = []
    for w in average_steps:
        diff_vals_list_windows.append(
            get_cross_section_data(diff_raster, bounds_poly, dist_step, w)
        )

    for j, vlw in enumerate(vals_list_windows_per_raster):
        for i, vals_list in enumerate(vlw):
            d, v, _, psx = vals_list
            fig.add_trace(
                go.Scatter(
                    x=d[0],
                    y=v[0],
                    mode="lines",
                    marker=dict(
                        color=raster_colours[j],
                    ),
                    name=raster_names[j],
                    visible=True if i == 0 else False,
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=d[1],
                    y=v[1],
                    mode="lines",
                    marker=dict(
                        color=raster_colours[j],
                    ),
                    name=raster_names[j],
                    visible=True if i == 0 else False,
                    showlegend=False,
                ),
                row=2,
                col=1,
                secondary_y=False,
            )

    for i, vals_list in enumerate(diff_vals_list_windows):
        d, v, _, psx = vals_list
        if diff_opacity != 0:
            fig.add_trace(
                go.Scatter(
                    x=d[0],
                    y=[abs(i) for i in v[0]],
                    mode="lines",
                    marker=dict(
                        color="blue",
                    ),
                    name="",
                    visible=True if i == 0 else False,
                    opacity=diff_opacity,
                    showlegend=False,
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=d[1],
                    y=[abs(i) for i in v[1]],
                    mode="lines",
                    marker=dict(
                        color="red",
                    ),
                    name="",
                    visible=True if i == 0 else False,
                    opacity=diff_opacity,
                    showlegend=False,
                ),
                row=2,
                col=1,
                secondary_y=True,
            )
        fig.add_trace(
            go.Scatter(
                x=[i[0] for i in psx[0]],
                y=[imh - i[1] for i in psx[0]],
                mode="lines",
                marker=dict(
                    color="blue",
                ),
                name="Absolute diff along",
                hovertemplate="%{text}<extra></extra>",
                text=[f"{v[0][i]}" for i in range(len(v[0]))],
                visible=True if i == 0 else False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[i[0] for i in psx[1]],
                y=[imh - i[1] for i in psx[1]],
                mode="lines",
                marker=dict(
                    color="red",
                ),
                name="Absolute diff across",
                hovertemplate="%{text}<extra></extra>",
                text=[f"{v[1][i]}" for i in range(len(v[1]))],
                visible=True if i == 0 else False,
            ),
            row=1,
            col=2,
        )

    x, y = bounds_poly.exterior.xy
    x = [int(np.floor((i - transform.c) / transform.a)) for i in x.tolist()]
    y = [imh - int(np.floor((transform.f - i) / abs(transform.e))) for i in y.tolist()]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            marker=dict(
                color="magenta",
            ),
            name="AOI boundary",
            visible=True,
            hoverinfo="none",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(updatemenus=updatemenu)
    fig.add_layout_image(
        dict(
            source=Image.open(temp_path / "temp_img.jpg"),
            xref="x",
            yref="y",
            x=0,
            y=imh,
            sizex=imw,
            sizey=imh,
            sizing="stretch",
            opacity=0.5,
            visible=True,
            layer="below",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title_text="Distance(m)", row=2, col=1)
    fig.update_yaxes(
        title_text="Elevation(m)",
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Elevation(m)",
        row=2,
        col=1,
        secondary_y=False,
    )
    if diff_opacity != 0.0:
        fig.update_yaxes(
            title_text="ABS Difference(m)",
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="ABS Difference(m)",
            row=2,
            col=1,
            secondary_y=True,
        )
    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=2)
    fig.update_layout(hovermode="x unified")

    shutil.rmtree(temp_path)

    if save_path:
        fig.write_html(save_path)

    return fig
