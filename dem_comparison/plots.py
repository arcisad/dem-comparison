import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
import rasterio as rio
from dem_comparison.utils import (
    enhance_image,
    get_cross_section_data,
    bin_metrics,
    read_metrics,
    hillshade,
)
from dem_handler.utils.spatial import resize_bounds, BoundingBox
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from shapely import Polygon
import shutil
import cv2 as cv
import pandas as pd
from scipy import stats

PIL.Image.MAX_IMAGE_PIXELS = 933120000


def rescale_intensities(img: np.ndarray, new_range: tuple = (0, 1)) -> np.ndarray:
    """Rescales the intensities of an image to a new range

    Parameters
    ----------
    img : np.ndarray
    new_range : tuple, optional
        by default (0, 1)

    Returns
    -------
    np.ndarray
    """
    new_image_range = new_range[1] - new_range[0]
    img_max = img[~np.isnan(img)].max()
    img_min = img[~np.isnan(img)].min()
    img_range = img_max - img_min
    new_img = (((img - img_min) * new_image_range) / img_range) + new_range[0]
    return new_img


def plot_metrics(
    metric_files: list[Path],
    is_error: bool = True,
    polar: bool = True,
    save_path: Path | None = None,
    data_bounds: tuple | list[tuple] | None = None,
    bins: int | list | None = None,
    percentiles_bracket: tuple | list[tuple] | None = None,
    plot_resolution: tuple | None = None,
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
        Filters out the data beyond the given bounds (left and right inclusive), by default None,
        If list of bounds provided the number of bounds should be the same as the number of metrics.
    bins: int | list | None, optional
        Bins the metrics if passed, either the number of bins or list of list of left edges/list of bin numbers or a combination of both, by default None
    percentiles_bracket: tuple | None, optional
        Only uses the data within the given percentals bracket (lower, upper). If used with `data_bounds`, the percentiles are applied on filtered data by the bounds, by default None.
    plot_resolution: tuple | None, optional,
        Turns autosize off and force the resolution (h, w), by default None

    Returns
    -------
    go.Figure
        Plotly figure
    """

    def filter_data(met, data_x, data_y, db, is_percentile=False):
        if is_percentile:
            pl = np.percentile(met, db[0])
            pu = np.percentile(met, db[1])
            validity_list = [True if pl < el < pu else False for el in met]
        else:
            validity_list = [True if db[0] <= el <= db[1] else False for el in met]
        new_metric = [el for j, el in enumerate(met) if validity_list[j]]
        new_x = [el for j, el in enumerate(data_x) if validity_list[j]]
        new_y = [el for j, el in enumerate(data_y) if validity_list[j]]
        return new_x, new_y, new_metric

    labels = ["ME" if is_error else "MEAN", "STD", "MSE", "NMAD"]
    metrics, x, y = read_metrics(metric_files, numerical_axes=polar)

    if type(data_bounds) is list:
        assert len(data_bounds) == len(
            metrics
        ), "If used as a list, the number of data bounds should be the same as the number of metrics."

    if type(percentiles_bracket) is list:
        assert len(percentiles_bracket) == len(
            metrics
        ), "If used as a list, the number of percentiles brackets should be the same as the number of metrics."

    if type(bins) is list:
        assert len(bins) == len(
            metrics
        ), "If used as a list, the number of bins should be the same as the number of metrics."

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
        horizontal_spacing=0.1,
    )
    if not polar:
        fig.update_layout(xaxis_title="Lat", yaxis_title="Lon")
    min_lat = 90
    for i, metric in enumerate(metrics):
        if data_bounds is not None:
            if type(data_bounds) is list:
                x, y, metric = filter_data(metric, x, y, data_bounds[i])
            else:
                x, y, metric = filter_data(metric, x, y, data_bounds)
        if percentiles_bracket is not None:
            if type(percentiles_bracket) is list:
                x, y, metric = filter_data(metric, x, y, percentiles_bracket[i], True)
            else:
                x, y, metric = filter_data(metric, x, y, percentiles_bracket, True)
        if min(x) < min_lat:
            min_lat = min(x)
        color = metric
        hover_text = metric
        if bins is not None:
            if type(bins) is list:
                bin_vals, bin_edges, bin_steps, diff_list = bin_metrics(
                    metric, bins=bins[i]
                )
                if type(bin_steps) is not list:
                    bin_steps = [bin_steps] * len(bin_edges)
            else:
                bin_vals, bin_edges, bin_steps, diff_list = bin_metrics(
                    metric, bins=bins
                )
                bin_steps = [bin_steps] * len(bin_edges)
            if type(bin_steps) is not list:
                bin_steps = [bin_steps] * len(bin_edges)
            color = bin_vals
            hover_text = []
            for k, bv in enumerate(bin_vals):
                bin_start = np.round(bv - diff_list[k] / 2, 2)
                bin_end = np.round(bv + diff_list[k] / 2, 2)
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
        if bins is None:
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
            available_bins = np.array(bin_edges).tolist()
            if np.max(metric) < max(available_bins):
                available_bins = [be for be in available_bins if be < np.max(metric)]
            available_bins = available_bins + [np.max(metric).tolist()]
            if np.min(metric) > min(available_bins):
                available_bins = [be for be in available_bins if be > np.min(metric)]
            available_bins = [np.min(metric).tolist()] + available_bins
            hover_text_hist = [
                f"{np.round(be, 2)}-{np.round(be + bin_steps[k], 2)}"
                for k, be in enumerate(np.array(bin_edges).tolist())
            ]
            fig.add_trace(
                go.Bar(
                    x=bin_edges,
                    y=np.histogram(metric, available_bins)[0],
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
            "radialaxis": {"range": [90, np.floor(min_lat).tolist() - 1]},
        }
        fig.update_layout(polar=polar_layout)
    if bins is not None:
        fig.update_layout(bargap=0)

    titles_list = [bins, data_bounds, percentiles_bracket]
    alone_cond = titles_list.count(None) == 2

    if data_bounds is not None:
        if alone_cond:
            ending = ".</span>"
        elif None in titles_list:
            if titles_list.index(None) == 0:
                ending = ", "
            elif titles_list.index(None) == 2:
                ending = ".</span>"
        else:
            ending = ", "

    fig_title = "Elevation difference metrics. <span style='font-size: 12px;'>With "
    if bins is not None:
        fig_title += (
            f"num bins for each metirc: {bins}{'.</span>' if alone_cond else ', '}"
        )
    if data_bounds is not None:
        fig_title += f"thresholds for each metric: {data_bounds}{ending}"
    if percentiles_bracket is not None:
        fig_title += (
            f"percentile brackets for each metric: {percentiles_bracket}.</span>"
        )
    fig.update_layout(title=dict(text=fig_title))
    if plot_resolution:
        fig.update_layout(
            height=plot_resolution[0],
            width=plot_resolution[1],
        )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_cross_sections(
    raster: Path | list[Path],
    bounds_poly: Polygon,
    diff_raster: Path,
    dist_step: int = 30,
    average_steps: list[int] = [5, 15, 30, 60, 100],
    temp_path: Path = Path("TMP"),
    raster_names: list[str] | None = None,
    raster_colours: list[str] | None = None,
    axes_label: str = "Elevation(m)",
    diff_unit: str = "m",
    save_path: Path | None = None,
    diff_opacity: float = 1.0,
    full_map: Path = Path("resources/mosaic_downsampled_3200m.tif"),
    map_intensity_range: tuple = (-50, 50),
    map_color_steps: int = 15,
    aoi_name: str = "",
    along_line_ratio: float = 0.5,
    across_line_ratio: float = 0.5,
    hillshade_index: int | None = None,
    plot_resolution: tuple | None = None,
    dynamic_spacing: bool = True,
    aoi_buffer: int = 0,
):
    """Plots the cross section changes of an area of interest

    Parameters
    ----------
    raster : Path | list[Path]
        Path ot list of Paths to the required raster files
    bounds_poly : Polygon
        Bounds for the area of interest in the rater files
    diff_raster : Path
        Difference raster (diff between input rasters)
    dist_step : int, optional
        Steps of spatial intervals in the cross sections in meters, by default 30
    average_steps : list[int], optional
        Smoothing levels for the plots, by default [5, 15, 30, 60, 100]
    temp_path : Path, optional
        Temporary path for intermediate outputs, by default Path("TMP")
    raster_names : list[str] | None, optional
        Name of the rasters in the plot, by default None
    raster_colours : list[str] | None, optional
        Color of the rasters in the plot, by default None
    axes_label: str, optional
        Label for the plot axes, by default "Elevation(m)"
    diff_unit: str, optional
        Unit string to go on absolute difference axis, by defaul "m"
    save_path : Path | None, optional
        Path to sabve the outputs, by default None
    diff_opacity : float, optional
        Opacity of the difference plot, by default 1.0
    full_map: Path, optional
        Path to the full map of the region, by default Path("resources/mosaic_downsampled_3200m.tif"),
    map_intensity_range : tuple, optional
        Intensity clipping range in the background map, by default (-50, 50)
    map_color_steps : int, optional
        Number of color steps in the background map, by default 15
    aoi_name : str, optional
        Name of the area of interest to go into plot title, by default ""
    along_line_ratio: float, optional
        Location of the line along the bounding box on the shorter side, by default 0.5
    along_line_ratio: float, optional
        Location of the line across the bounding box on the longer side, by default 0.5
    hillshade_index: int | None, optional
        If provided, the raster with this index from the input list will be used for the map with the diff map overlaid on it, by default None
    plot_resolution: tuple | None, optional,
        Turns autosize off and force the resolution (h, w), by default None
    dynamic_spacing: bool, optional
        Dynamic spacing between subplots, by default True
    aoi_buffer: float, optional
        Buffering the aoi locaiton on the full map in pixels, by default 0
    Returns
    -------
    _type_
        _description_
    """
    if type(raster) is not list:
        raster = [raster]

    if hillshade_index is not None:
        assert hillshade_index < len(raster), "Wrong hillshade index"

    if raster_names is None:
        raster_names = [f"Raster_{i}" for i in range(len(raster))]

    if raster_colours is None:
        raster_colours = ["black"] * len(raster)

    plot_image = raster[0]
    if diff_raster:
        plot_image = diff_raster

    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)

    with rio.open(plot_image) as ds:
        full_map_width = int(ds.profile["width"] / 3.3)
        full_map_height = int(ds.profile["height"] / 3.3)

    if hillshade_index is not None:
        with rio.open(raster[hillshade_index]) as ds:
            hillshade_img = hillshade(ds.read(1), skip_negative=False)
            hillshade_img[np.isnan(hillshade_img)] = 0.0
            hillshade_img = hillshade_img.astype("uint8")
            hillshade_img = np.dstack([hillshade_img] * 3)

    with rio.open(full_map, "r") as ds:
        full_map_transform = ds.transform
        full_map_img = ds.read(1)
        plot_bounds = resize_bounds(BoundingBox(*bounds_poly.bounds), 2).bounds
        minx = (
            int(
                np.floor((plot_bounds[0] - full_map_transform.c) / full_map_transform.a)
            )
            - aoi_buffer
        )
        miny = (
            int(
                np.floor(
                    (full_map_transform.f - plot_bounds[1]) / abs(full_map_transform.e)
                )
            )
            - aoi_buffer
        )
        maxx = (
            int(
                np.floor((plot_bounds[2] - full_map_transform.c) / full_map_transform.a)
            )
            + aoi_buffer
        )
        maxy = (
            int(
                np.floor(
                    (full_map_transform.f - plot_bounds[3]) / abs(full_map_transform.e)
                )
            )
            + aoi_buffer
        )
        full_map_img_rect = full_map_img.copy()
        full_map_img_rect = cv.rectangle(
            full_map_img_rect,
            (minx, miny),
            (maxx, maxy),
            float(full_map_img[~np.isnan(full_map_img)].max()),
            -1,
        )
        mask = np.zeros_like(full_map_img_rect)
        mask = cv.rectangle(
            mask,
            (minx, miny),
            (maxx, maxy),
            1.0,
            -1,
        )
        full_map_shape = (
            full_map_width,
            full_map_height,
        )
        full_map_img = cv.resize(
            full_map_img, dsize=full_map_shape, interpolation=cv.INTER_LINEAR
        )
        full_map_img_rect = cv.resize(
            full_map_img_rect, dsize=full_map_shape, interpolation=cv.INTER_LINEAR
        )
        mask = cv.resize(mask, dsize=full_map_shape, interpolation=cv.INTER_LINEAR)
        full_map_img = np.dstack([full_map_img_rect, full_map_img, full_map_img_rect])
        full_map_img = rescale_intensities(full_map_img)
        full_map_img[mask > 0, 0] = 255.0
        full_map_img[mask > 0, 1] = 0.0
        full_map_img[mask > 0, 2] = 255.0

    with rio.open(plot_image) as ds:
        transform = ds.transform
        plot_image_data = ds.read(1)
        img, nan_mask = enhance_image(
            plot_image_data,
            return_nan_mask=True,
            intensity_range=map_intensity_range,
            color_steps=map_color_steps,
        )
        colorbar_colors = img[~nan_mask]
        unique_colors = np.unique(colorbar_colors, axis=0)[::-1]
        colorbar_colors = [f"rgb{tuple(cc.tolist())}" for cc in unique_colors]
        colorbar_data = plot_image_data[~np.isnan(plot_image_data)]
        binned_data, _, _, _ = bin_metrics(
            colorbar_data, len(unique_colors), map_intensity_range
        )
        binned_data = np.unique(binned_data)
        if hillshade_index is not None:
            img = cv.addWeighted(hillshade_img, 0.5, img, 0.5, 0.0)
        for i in range(3):
            img[0 : full_map_shape[1], 0 : full_map_shape[0]][
                ~np.isnan(full_map_img)
            ] = (full_map_img[~np.isnan(full_map_img)]).astype("uint8")
        img[nan_mask] = stats.mode(img[~nan_mask]).mode
        imh, imw, _ = img.shape
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

    spacing_factor = 0.4 * imw / imh
    if not dynamic_spacing:
        spacing_factor = 0.4

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy", "secondary_y": True, "rowspan": 2}, None],
            [None, {"type": "xy", "rowspan": 2}],
            [{"type": "xy", "secondary_y": True, "rowspan": 2}, None],
            [None, None],
        ],
        column_widths=[1 - spacing_factor, spacing_factor],
        horizontal_spacing=0.1,
    )

    vals_list_windows_per_raster = []
    for r in raster:
        vals_list_windows = []
        for w in average_steps:
            vals_list_windows.append(
                get_cross_section_data(
                    r, bounds_poly, dist_step, w, along_line_ratio, across_line_ratio
                )
            )
        vals_list_windows_per_raster.append(vals_list_windows)

    diff_vals_list_windows = []
    for w in average_steps:
        diff_vals_list_windows.append(
            get_cross_section_data(
                diff_raster,
                bounds_poly,
                dist_step,
                w,
                along_line_ratio,
                across_line_ratio,
            )
        )

    for j, vlw in enumerate(vals_list_windows_per_raster):
        for i, vals_list in enumerate(vlw):
            d, v, _, _ = vals_list
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
                row=3,
                col=1,
                secondary_y=False,
            )

    for i, vals_list in enumerate(diff_vals_list_windows):
        d, v, _, pxs = vals_list
        if diff_opacity != 0:
            fig.add_trace(
                go.Scatter(
                    x=d[0],
                    y=[abs(i) for i in v[0]],
                    mode="lines",
                    marker=dict(
                        color="blue",
                    ),
                    name="Absolute diff along",
                    visible=True if i == 0 else False,
                    opacity=diff_opacity,
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
                    name="Absolute diff across",
                    visible=True if i == 0 else False,
                    opacity=diff_opacity,
                ),
                row=3,
                col=1,
                secondary_y=True,
            )
        fig.add_trace(
            go.Scatter(
                x=[i[0] for i in pxs[0]],
                y=[imh - i[1] for i in pxs[0]],
                mode="lines",
                marker=dict(
                    color="blue",
                ),
                name="Map diff (Blue line)",
                hovertemplate="%{text}",
                text=[f"{v[0][i]}" for i in range(len(v[0]))],
                visible=True if i == 0 else False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[i[0] for i in pxs[1]],
                y=[imh - i[1] for i in pxs[1]],
                mode="lines",
                marker=dict(
                    color="red",
                ),
                name="Map diff (Red line)",
                hovertemplate="%{text}",
                text=[f"{v[1][i]}" for i in range(len(v[1]))],
                visible=True if i == 0 else False,
            ),
            row=2,
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
        row=2,
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
            opacity=0.65,
            visible=True,
            layer="below",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(imw // 2, imw // 2 + len(colorbar_colors))),
            y=list(range(imh // 2, imh // 2 + len(colorbar_colors))),
            mode="markers",
            marker=dict(
                opacity=0.0,
                colorscale=colorbar_colors,
                showscale=True,
                cmin=binned_data.min(),
                cmax=binned_data.max(),
                colorbar=dict(
                    x=0.95,
                    thickness=15,
                    tickvals=[
                        binned_data.min(),
                        binned_data.mean(),
                        binned_data.max(),
                    ],
                    ticktext=[
                        "{:.2f}".format(colorbar_data.min()),
                        "{:.2f}".format(colorbar_data.mean()),
                        "{:.2f}".format(colorbar_data.max()),
                    ],
                    outlinewidth=0,
                    len=(
                        int(np.round(imh * (plot_resolution[0] / plot_resolution[1])))
                        if plot_resolution is not None
                        else imh
                    ),
                    lenmode="pixels",
                ),
            ),
            showlegend=False,
            visible=True,
        ),
        row=2,
        col=2,
    )
    fig.update_layout(showlegend=True)
    if diff_opacity != 0.0:
        fig.update_xaxes(title_text="Distance(m)", row=3, col=1)
    else:
        fig.update_xaxes(title_text="Distance(m) (Blue line)", row=1, col=1)
        fig.update_xaxes(title_text="Distance(m) (Red Line)", row=3, col=1)
    if hillshade_index is not None:
        fig.update_xaxes(
            title_text=f"Hillshade map: {raster_names[hillshade_index]}", row=2, col=2
        )
    fig.update_yaxes(
        title_text=axes_label,
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=axes_label,
        row=3,
        col=1,
        secondary_y=False,
    )
    if diff_opacity != 0.0:
        fig.update_yaxes(
            title_text=f"ABS Difference({diff_unit})",
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text=f"ABS Difference({diff_unit})",
            row=3,
            col=1,
            secondary_y=True,
        )
    fig.update_xaxes(showgrid=False, showticklabels=False, row=2, col=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=2, col=2)
    fig.update_layout(hovermode="x unified")
    aoi_str = f" for {aoi_name}" if aoi_name != "" else ""
    fig.update_layout(
        title=dict(text=f"Cross section plot{aoi_str}, resolution: {dist_step}m")
    )
    if plot_resolution:
        fig.update_layout(
            height=plot_resolution[0],
            width=plot_resolution[1],
        )

    shutil.rmtree(temp_path)

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_slope_vs_height(
    slope_rasters: list[Path],
    height_diff_raster: Path,
    slope_diff_raster: Path,
    bounds_poly: Polygon,
    slope_diff_clip_range: tuple = (-10, 10),
    save_path: Path | None = None,
    use_full_data: bool = False,
    raster_names: list[str] | None = None,
    precision: int = 1,
    return_all_figs: bool = False,
    aoi_name: str = "",
    along_line_ratio: float = 0.5,
    across_line_ratio: float = 0.5,
    plot_trendline: bool = False,
) -> go.Figure | list[go.Figure]:
    """Plots height diff vs raster slopes

    Parameters
    ----------
    slope_rasters : list[Path]
        List of rasters with slope data
    height_diff_raster : Path
    slope_diff_raster : Path
    bounds_poly : Polygon
        Plots data inside this bounding box
    slope_diff_clip_range : tuple, optional
        Clips all data based od the values for which their corresponding slope diffs are whithin this range, by default (-10, 10)
    save_path : Path | None, optional
        Path to save the plot, by default None
    use_full_data : bool, optional
        USe full data instead of the cross lines, by default False
    raster_names : list[str] | None, optional
        Names of the ratsers to be used in the plots, by default None
    precision : int, optional
        Number of decimal points for precision of the slopes, by default 1
    return_all_figs : bool, optional
        Returns list of figures instead of a single figure with subplots, by default False
    aoi_name : str, optional
        Name of the area of interest to go into plot title, by default ""
    along_line_ratio: float, optional
        Location of the line along the bounding box on the shorter side, by default 0.5
    along_line_ratio: float, optional
        Location of the line across the bounding box on the longer side, by default 0.5
    plot_trendline: bool, optional
        Plots the trendlines using Ordinary Least Squares, by default False
    Returns
    -------
    go.Figure | list[go.Figure]
    """
    if raster_names is None:
        raster_names = [p.stem for p in slope_rasters]

    if use_full_data:
        height_diff_data = rio.open(height_diff_raster).read(1)
        slope_diff_data = rio.open(slope_diff_raster).read(1)

        idx = np.logical_and(
            (slope_diff_clip_range[0] < slope_diff_data),
            (slope_diff_data < slope_diff_clip_range[1]),
        )
        height_diff_data = np.abs(height_diff_data[idx])
        slope_diff_data = slope_diff_data[idx]

        slope_data = []
        for sr in slope_rasters:
            data = rio.open(sr).read(1)[idx]
            slope_data.append(data)

        diff_df = pd.DataFrame(
            {
                "Direction": np.repeat("Everywhere", len(slope_diff_data)),
                "Slope_Diff": slope_diff_data,
                "Height_Diff": height_diff_data,
            }
        )

        names_list = []
        slopes_list = []
        for i, d in enumerate(slope_data):
            slopes_list.extend(np.round(d, precision))
            names_list.extend([raster_names[i]] * len(d))

        slope_dict = {
            "Name": names_list,
            "Slope": slopes_list,
            "Height_Diff": np.tile(height_diff_data, len(slope_data)),
            "Direction": np.repeat("Everywhere", len(slopes_list)),
        }
        slope_df = pd.DataFrame(slope_dict)
        slope_df_grouped_mean = (
            slope_df[["Slope", "Height_Diff"]].groupby("Slope").mean().reset_index()
        )
        slope_df_grouped_std = (
            slope_df[["Slope", "Height_Diff"]].groupby("Slope").std().reset_index()
        )
    else:
        height_diff_data = get_cross_section_data(
            height_diff_raster,
            bounds_poly,
            along_line_ratio=along_line_ratio,
            across_line_ratio=across_line_ratio,
        )
        slope_diff_data = get_cross_section_data(
            slope_diff_raster,
            bounds_poly,
            along_line_ratio=along_line_ratio,
            across_line_ratio=across_line_ratio,
        )

        slope_data = []
        for sr in slope_rasters:
            slope_data.append(
                get_cross_section_data(
                    sr,
                    bounds_poly,
                    along_line_ratio=along_line_ratio,
                    across_line_ratio=across_line_ratio,
                )
            )

        idx = np.logical_and(
            (slope_diff_clip_range[0] < slope_diff_data[1][0]),
            (slope_diff_data[1][0] < slope_diff_clip_range[1]),
        )
        along_hdiff = height_diff_data[1][0][idx]
        along_sdiff = slope_diff_data[1][0][idx]
        along_name = np.repeat("Along", len(along_hdiff))

        slope_data_along = []
        for d in slope_data:
            slope_data_along.append(np.round(d[1][0][idx], precision))

        idx = np.logical_and(
            (slope_diff_clip_range[0] < slope_diff_data[1][1]),
            (slope_diff_data[1][1] < slope_diff_clip_range[1]),
        )
        across_hdiff = height_diff_data[1][1][idx]
        across_sdiff = slope_diff_data[1][1][idx]
        across_name = np.repeat("Across", len(across_hdiff))

        slope_data_across = []
        for d in slope_data:
            slope_data_across.append(np.round(d[1][1][idx], precision))

        hdiff = np.abs(np.concat([along_hdiff, across_hdiff]))
        sdiff = np.concat([along_sdiff, across_sdiff])
        direction = np.concat([along_name, across_name])

        slopes_list = []
        names_list = []
        for i, (dal, daa) in enumerate(zip(slope_data_along, slope_data_across)):
            data = np.concat([dal, daa])
            slopes_list.extend(data)
            names_list.extend([raster_names[i]] * len(data))

        diff_df = pd.DataFrame(
            {
                "Direction": direction,
                "Slope_Diff": sdiff,
                "Height_Diff": hdiff,
            }
        )
        slope_dict = {
            "Name": names_list,
            "Slope": slopes_list,
            "Height_Diff": np.tile(hdiff, len(slope_data)),
            "Direction": np.tile(direction, len(slope_data)),
        }
        slope_df = pd.DataFrame(slope_dict)
        slope_df_grouped_mean = (
            slope_df.drop("Name", axis=1)
            .groupby(["Direction", "Slope"])
            .mean()
            .reset_index()
            .iloc[::-1]
        )
        slope_df_grouped_std = (
            slope_df.drop("Name", axis=1)
            .groupby(["Direction", "Slope"])
            .std()
            .reset_index()
            .iloc[::-1]
        )

    trendline = "ols" if plot_trendline else None

    dif_fig = px.scatter(
        diff_df,
        x="Slope_Diff",
        y="Height_Diff",
        color="Direction",
        trendline=trendline,
    )

    if use_full_data:
        slope_fig_mean = px.scatter(
            slope_df_grouped_mean,
            x="Slope",
            y="Height_Diff",
            labels={
                "Height_Diff": "Mean height diff",
            },
            trendline=trendline,
        )
        slope_fig_std = px.scatter(
            slope_df_grouped_std,
            x="Slope",
            y="Height_Diff",
            labels={
                "Height_Diff": "Height diff STD",
            },
            trendline=trendline,
        )
    else:
        slope_fig_mean = px.scatter(
            slope_df_grouped_mean,
            x="Slope",
            y="Height_Diff",
            color="Direction",
            labels={
                "Height_Diff": "Mean height diff",
            },
            trendline=trendline,
        )
        slope_fig_std = px.scatter(
            slope_df_grouped_std,
            x="Slope",
            y="Height_Diff",
            color="Direction",
            labels={
                "Height_Diff": "Height diff STD",
            },
            trendline=trendline,
        )

    figures = [dif_fig, slope_fig_mean, slope_fig_std]
    fig = make_subplots(
        rows=len(figures),
        cols=1,
        specs=[[{"type": "xy"}]] * len(figures),
        # vertical_spacing=0.1,
    )
    for i, figure in enumerate(figures):
        for trace in range(len(figure["data"])):
            if i != 0:
                figure["data"][trace]["showlegend"] = False
            fig.add_trace(figure["data"][trace], row=i + 1, col=1)

    fig.update_xaxes(title="Slope diff (°)", row=1, col=1)
    fig.update_yaxes(title="ABS Height diff (m)", row=1, col=1)

    fig.update_xaxes(title="Slope (°)", row=2, col=1)
    fig.update_yaxes(
        title="Mean ABS height diff (m)<br></br>Mean Absolute Error (MAE)", row=2, col=1
    )

    fig.update_xaxes(title="Slope (°)", row=3, col=1)
    fig.update_yaxes(
        title="Height diff STD (m)<br></br>Standard Error of Mean (SEM)", row=3, col=1
    )
    aoi_str = f" for {aoi_name}" if aoi_name != "" else ""
    fig.update_layout(title=dict(text=f"Elevation vs Slope plot{aoi_str}"))

    if save_path:
        fig.write_html(save_path)

    return figures if return_all_figs else fig
