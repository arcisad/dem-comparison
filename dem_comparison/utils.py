from rasterio.enums import Resampling
from rasterio.crs import CRS
from pathlib import Path
import numpy as np
import rioxarray
import rasterio as rio
from osgeo import gdal
from dem_handler.utils.spatial import resize_bounds, BoundingBox
import os
import pickle
import shutil
from itertools import product
from shapely import LineString, Point, Polygon
from rasterio.transform import Affine
from osgeo import ogr
from shapely import ops, from_wkt

TEST_PATH = Path("tests")
GEOID_PATH = TEST_PATH / "data/geoid/egm_08_geoid.tif"
REMA_INDEX_PATH = TEST_PATH / "rema/data/REMA_Mosaic_Index_v2_gpkg.gpkg"
TMP_PATH = TEST_PATH / "TMP"
TEST_DATA_PATH = TEST_PATH / "data"

get_pairs = lambda seq, ws: [seq[i : i + ws] for i in range(len(seq) - ws + 1)]

buffer_range = lambda seq, bi: seq + [
    seq[-1] + (seq[1] - seq[0]) * (i + 1) for i in range(bi)
]


def hillshade(
    array: np.ndarray,
    azimuth: float = 30.0,
    angle_altitude: float = 30.0,
    skip_negative: bool = True,
) -> np.ndarray:
    """Creates a hillshade view of a given image array.

    Parameters
    ----------
    array : np.ndarray
        Numpy array of the image
    azimuth : float, optional
        Azimuth angle, by default 30.0
    angle_altitude : float, optional
        Altitude of the angle, by default 30.0
    skip_negative : bool, optional
        Skip negative values, by default True

    Returns
    -------
    np.ndarray
        Hillshade of the input image.
    """

    assert (
        azimuth <= 360.0
    ), "Azimuth angle should be lass than or equal to 360 degrees."
    assert (
        angle_altitude <= 90.0
    ), "Altitude angle should be lass than or equal to 90 degrees."

    if skip_negative:
        array[array < 0] = np.nan

    azimuth = 360.0 - azimuth
    azi_rad = azimuth * np.pi / 180.0  # azimuth in radians

    alt_rad = angle_altitude * np.pi / 180.0  # altitude in radians

    x, y = np.gradient(array)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)

    shaded = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos(
        (azi_rad - np.pi / 2.0) - aspect
    )

    return 255 * (shaded + 1) / 2


def reproject_match_tifs(
    tif_1,
    tif_2,
    target_crs,
    roi_poly=None,
    save_path_1="",
    save_path_2="",
    convert_dB=True,
    set_nodata=False,
    resampling=Resampling.bilinear,
    verbose=False,
):

    tif_1_f = rioxarray.open_rasterio(tif_1)
    tif_2_f = rioxarray.open_rasterio(tif_2)
    crs_1 = tif_1_f.rio.crs
    crs_2 = tif_2_f.rio.crs
    nodata_1 = tif_1_f.rio.nodata
    nodata_2 = tif_2_f.rio.nodata

    # reproject raster to target crs if not already
    if verbose:
        print("reprojecting arrays if not in target crs")
        print(f"target crs: {target_crs}, crs_1: {crs_1}, crs_2: {crs_2}")
    tif_1_reproj = (
        tif_1_f.rio.reproject(f"EPSG:{target_crs}")
        if str(tif_1_f.rio.crs).split(":")[-1] != str(target_crs)
        else tif_1_f
    )
    tif_2_reproj = (
        tif_2_f.rio.reproject(f"EPSG:{target_crs}")
        if str(tif_2_f.rio.crs).split(":")[-1] != str(target_crs)
        else tif_2_f
    )
    del tif_1_f, tif_2_f
    if verbose:
        print("Shape of arrays after reprojection to target crs")
        print(tif_1_reproj.shape, tif_2_reproj.shape)
    # clip by the scene geometry
    if roi_poly is not None:
        if verbose:
            print("clip arrays by the roi")
        tif_1_clipped = tif_1_reproj.rio.clip([roi_poly], CRS.from_epsg(target_crs))
        tif_2_clipped = tif_2_reproj.rio.clip([roi_poly], CRS.from_epsg(target_crs))
        if verbose:
            print("Shape of arrays after being clipped by the roi")
            print(tif_1_clipped.shape, tif_2_clipped.shape)
    else:
        tif_1_clipped = tif_1_reproj.copy()
        tif_2_clipped = tif_2_reproj.copy()
    del tif_1_reproj, tif_2_reproj
    # match the shape and resolution of the two tifs as they may be slighly off
    # match tif 2 to tif 1 if tif 1 did not require reprojection originally
    if verbose:
        print("matching shape and resolutions")
        print("reprojecting tif 2 to tif 1")
    tif_2_matched = tif_2_clipped.rio.reproject_match(
        tif_1_clipped, resampling=resampling
    )
    tif_1_matched = tif_1_clipped.copy()
    del tif_1_clipped, tif_2_clipped
    if verbose:
        print("Shape of arrays after matching tifs")
        print(tif_1_matched.shape, tif_2_matched.shape)

    if set_nodata:
        if verbose:
            print(f"setting nodata values to {set_nodata}")
        tif_1_matched = tif_1_matched.where(nodata_1, set_nodata)
        tif_2_matched = tif_2_matched.where(nodata_2, set_nodata)
        tif_1_matched.rio.write_nodata(set_nodata, encoded=True, inplace=True)
        tif_2_matched.rio.write_nodata(set_nodata, encoded=True, inplace=True)

    if convert_dB:
        if verbose:
            print("converting from linear to db")
        tif_1_matched = 10 * np.log10(tif_1_matched)
        tif_2_matched = 10 * np.log10(tif_2_matched)

    if save_path_1:
        if verbose:
            print(f"Saving to: {save_path_1}")
        tif_1_matched.rio.to_raster(save_path_1)
    if save_path_2:
        if verbose:
            print(f"Saving to: {save_path_2}")
        tif_2_matched.rio.to_raster(save_path_2)

    return tif_1_matched, tif_2_matched


def simple_mosaic(
    dem_rasters: list[Path],
    save_path: Path,
    resolution: str | tuple = "lowest",
    bounds_scale_factor: float = 1.02,
    keep_vrt: bool = False,
    fill_value: float | int | list[float] | list[int] | None = None,
    force_bounds: tuple | None = None,
) -> None:
    """Making simple mosaic of all given raster files

    Parameters
    ----------
    dem_rasters : list[Path]
        List of raster files
    save_path : Path
        Output file path.
    resolution : str, optional
        Resolution to be used, by default "lowest"
    bounds_scale_factor: float, optional
        Scales the bounds by this factor, by default 1.02
    keep_vrt: bool, optional
        Keeps the vrt file.
    fill_value: float | int | list[float] | list[int] | None, optional
        Fills the mosaic with a singl value if provided, or fill each raster with the corresponding value from list, by default None
    force_bounds: tuple | None, optional
        Forces the mosaic to this bounding box if provided, otherwise it uses the outer bounds of all rasters, by default None
    """
    force_resolution = type(resolution) is not str

    rasters = [rio.open(ds) for ds in dem_rasters]
    lefts = [r.bounds.left for r in rasters]
    rights = [r.bounds.right for r in rasters]
    bottoms = [r.bounds.bottom for r in rasters]
    tops = [r.bounds.top for r in rasters]
    profiles = [r.profile for r in rasters]

    if type(fill_value) is list:
        temp_dir = Path("temp_filled_rasters_dir")
        temp_dir.mkdir(exist_ok=True)
        for i in range(len(rasters)):
            img = rasters[i].read(1)
            img[~np.isnan(img)] = fill_value[i]
            temp_path = temp_dir / f"raster_filles_{i}.tif"
            r_profile = profiles[i]
            with rio.open(temp_path, "w", **r_profile) as ds:
                ds.write(img, 1)
            dem_rasters[i] = temp_path
    for r in rasters:
        r.close()

    bounds = (min(lefts), min(bottoms), max(rights), max(tops))
    if force_bounds is not None:
        bounds = force_bounds
    bounds = resize_bounds(BoundingBox(*bounds), bounds_scale_factor).bounds
    xRes = resolution[0] if force_resolution else None
    yRes = resolution[1] if force_resolution else None
    resolution = "user" if force_resolution else resolution
    VRT_options = gdal.BuildVRTOptions(
        resolution=resolution,
        outputBounds=bounds,
        VRTNodata=np.nan,
        xRes=xRes,
        yRes=yRes,
    )
    vrt_path = str(save_path).replace(".tif", ".vrt")
    ds = gdal.BuildVRT(vrt_path, dem_rasters, options=VRT_options)
    ds.FlushCache()

    src = rio.open(vrt_path)
    profile = src.profile
    profile.update({"driver": "GTiff"})
    array = src.read(1)
    if (type(fill_value) is float) or (type(fill_value) is int):
        array[~np.isnan(array)] = fill_value
    src.close()
    with rio.open(save_path, "w", **profile) as dst:
        dst.write(array, 1)

    if not keep_vrt:
        os.remove(vrt_path)

    shutil.rmtree("temp_filled_rasters_dir", ignore_errors=True)
    print(f"Mosaic created.")
    return None


def build_diff_mosaics(
    dem_diffs: list[Path], mosaic_dir: Path, interval: int = 10
) -> None:
    """Builds mosaics from elevation diffeence rasters.

    Parameters
    ----------
    dem_diffs : list[Path]
        List of paths to diff rasters.
    mosaic_dir : Path
        Ouput mosaics directory.
    interval: int, optional,
        Interval by degrees for determining the maximum size for each mosaic.
        For an interval of I, a maximum size of IxI degrees mosaic will be queried and created.

    Returns
    -------
    None
    """

    LAT_RANGE = range(-90, -50)
    LON_RANGE = range(-180, 180)

    splits = [f.stem.split("_") for f in dem_diffs]
    lats = [-int(el[0][:-1]) for el in splits]
    lons = [int(el[1][:-1]) * (-1 if el[1][-1] == "W" else 1) for el in splits]

    lat_range = list(range(LAT_RANGE.start, LAT_RANGE.stop, interval))
    lon_range = list(range(LON_RANGE.start, LON_RANGE.stop, interval))

    lat_range = buffer_range(lat_range, 1)
    lon_range = buffer_range(lon_range, 1)

    lat_pairs = [range(*l) for l in get_pairs(lat_range, 2)]
    lon_pairs = [range(*l) for l in get_pairs(lon_range, 2)]
    combinations = list(product(lat_pairs, lon_pairs))

    for i, c in enumerate(combinations):
        lon_start = "W" if c[1].start < 0 else "E"
        lon_stop = "W" if c[1].stop < 0 else "E"
        combination_str = f"{abs(c[0].start)}S_{abs(c[1].start)}{lon_start}_{abs(c[0].stop)}S_{abs(c[1].stop)}{lon_stop}"
        lat_cond = np.logical_and(
            np.array(lats) >= c[0].start, np.array(lats) < c[0].stop
        )
        lon_cond = np.logical_and(
            np.array(lons) >= c[1].start, np.array(lons) < c[1].stop
        )
        chunk = np.array(dem_diffs)[np.logical_and(lat_cond, lon_cond)].tolist()
        if len(chunk) == 0:
            print(f"No diff files found for {combination_str}. Skipping...")
            continue
        print(
            f"Mosaicing chunk {combination_str}. {i + 1} of {len(combinations)} combinations."
        )
        mosaic_name = f"{combination_str}.tif"
        simple_mosaic(chunk, mosaic_dir / mosaic_name)

    return None


def reproject_dem(
    dem: Path,
    target_crs: int,
    save_path: Path,
) -> None:
    """Reprojects DEM to given crs.

    Parameters
    ----------
    dem : Path
        DEM path
    target_crs : int
        Target crs.
    save_path : Path
        Output path

    """
    raster = rioxarray.open_rasterio(dem)
    raster_rep = raster.rio.reproject(f"EPSG:{target_crs}")
    raster_rep.rio.to_raster(save_path)
    raster_rep.close()
    return None


def read_metrics(metric_files: list[Path], numerical_axes: bool = False) -> tuple:
    """Reads metrics from pickle files

    Parameters
    ----------
    metric_files : list[Path]
        List of Paths to pickle files
    numerical_axes : bool, optional
        Returns axes in numeric format, by default False

    Returns
    -------
    tuple
        Metrics
    """
    data = []
    for pkl in metric_files:
        with open(pkl, "rb") as f:
            data.append(pickle.load(f))
    x = []
    y = []
    for pkl in metric_files:
        parts = str(Path(pkl).stem).split("_")
        if numerical_axes:
            x.append(int(parts[0][:-1]))
            y.append(int(parts[1][:-1]) * (-1 if parts[1][-1] == "W" else 1))
        else:
            x.append(parts[0])
            y.append(parts[1])

    me = [i[0] for i in data]
    std = [i[1] for i in data]
    mse = [i[2] for i in data]
    nmad = [i[3] for i in data]

    return [me, std, mse, nmad], x, y


def resample_dataset(
    dataset_path: Path,
    scale_factor: float | list[float] = 1.0,
    output_file: Path | None = None,
    force_shape: tuple | None = None,  # (height, width)
) -> tuple:
    """
    Resamples the output data and returns the new data and its new affine transformation according to `scale_factor`
    The output shape could also be forced using `forced_shape` parameter.

    Parameters
    ----------
    dataset_path : Path
        Path to input dataset
    scale_factor : float | list[float], optional
        Scale factor to resample the data, by default 1.0
    output_file : Path | None, optional
        If provided the output will be written to a raster file on this path, by default None
    force_shape : tuple | None, optional
        Forcing the output shape. Will make `scale_factor` ineffective, by default None

    Returns
    -------
    tuple
        _description_
    """
    with rio.open(dataset_path) as dataset:
        # resample data to target shape
        if type(scale_factor) == float:
            scale_factor = [scale_factor] * 2
        if force_shape is not None:
            output_shape = force_shape
        else:
            output_shape = (
                int(dataset.height * scale_factor[0]),
                int(dataset.width * scale_factor[1]),
            )
        data = dataset.read(
            out_shape=(
                dataset.count,
                *output_shape,
            ),
            resampling=Resampling.bilinear,
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )

        profile = dataset.profile
        profile.update(
            transform=transform,
            width=data.shape[2],
            height=data.shape[1],
            dtype=data.dtype,
        )

    if output_file:
        with rio.open(output_file, "w", **profile) as ds:
            for i in range(0, profile["count"]):
                ds.write(data[i], i + 1)

    print(f"Resampling finished for {dataset_path.name}")
    return data, transform


def get_cross_lines(poly: Polygon) -> tuple:
    """Extracts the crossing line from the centre of the sides of a rectangle

    Parameters
    ----------
    poly : Polygon
        Shapley Polygon

    Returns
    -------
    tuple
        Tuple of LineStrings
    """
    points = [np.array(el) for el in list(zip(*poly.exterior.xy))[:-1]]
    l1_points = [(points[0] + points[1]) / 2, (points[2] + points[3]) / 2]
    l2_points = [(points[0] + points[3]) / 2, (points[1] + points[2]) / 2]
    l1 = LineString([Point(*l1_points[0]), Point(*l1_points[1])])
    l2 = LineString([Point(*l2_points[0]), Point(*l2_points[1])])
    return (l1, l2) if l1.length > l2.length else (l2, l1)


def get_line_points(l: LineString, step_size: float = 30) -> tuple:
    """Finds point coords in a line at given step size

    Parameters
    ----------
    l : LineString
    step_size : float, optional
        step size, by default 30

    Returns
    -------
    list[Point]
    """
    dists = np.arange(0, l.length + step_size, step_size)
    return [l.interpolate(d) for d in dists], dists


def get_pixel_coords(p: Point, t: Affine) -> tuple:
    """Returns the pixel coords of a point

    Parameters
    ----------
    p : Point
    t : Affine
        Affine transform of the corresponding raster

    Returns
    -------
    Point
    """
    x, y = p.xy
    px = int(np.floor((x[0] - t.c) / t.a))
    py = int(np.floor((t.f - y[0]) / abs(t.e)))
    return (px, py)


def get_cross_section_data(
    raster: Path,
    bounds_poly: Polygon,
    step_size: float = 30.0,
    average_window: int | None = None,
):
    """Finds values of a raster along the crossing lines at the centre of either side of a given bounding box.

    Parameters
    ----------
    raster : Path
    bounds_poly : Polygon
    step_size: float, optional
        Steps for calculating the values along the crossing lines, by default 30
    average_window:int | None, optional,
        If passed, a moving average of the data by this window size will be returned, by default None

    Returns
    -------
    Tuple of distances along each line, the values as each step size, coords of the points in the world and coords of the points in pixels
    """
    with rio.open(raster, "r") as ds:
        transform = ds.transform
        img = ds.read(1)
        l1, l2 = get_cross_lines(bounds_poly)

        value_list = []
        dist_list = []
        ps_list = []
        psx_list = []
        for l in [l1, l2]:
            ps, dists = get_line_points(l, step_size)
            psx = [get_pixel_coords(p, transform) for p in ps]
            x = [i[0] for i in psx]
            y = [i[1] for i in psx]
            idx = (np.array(y), np.array(x))
            vals = img[idx]
            if average_window:
                vals = moving_average(vals, average_window)
            value_list.append(vals)
            dist_list.append(dists)
            ps_list.append([(i.xy[0][0], i.xy[1][0]) for i in ps])
            psx_list.append(psx)
    return dist_list, value_list, ps_list, psx_list


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w


def enhance_image(
    img: np.ndarray,
    intensity_range: tuple = (-50, 50),
    color_steps: int = 15,
    white_background: bool = False,
    return_nan_mask: bool = False,
) -> tuple | np.ndarray:
    """Enhances an elevation data image

    Parameters
    ----------
    img : np.ndarray
        input image
    intensity_range : tuple, optional
        Intensity clipping range, by default (-50, 50)
    color_steps : int, optional
        Number of color steps, by default 15
    white_background: bool, optional,
        Set the background to white, by default False
    return_nan_mask : bool,
        Returns NaN mask of the image, by default False

    Returns
    -------
    new enhanced image and its NaN mask if `return_nan_mask` is set.
    """
    nan_mask = np.isnan(img)

    img = np.clip(img, intensity_range[0], intensity_range[1])
    r_levels = np.clip(np.arange(255, -color_steps, -color_steps), 0, 255)
    g_levels = np.concat(
        [
            np.arange(0, 255, color_steps * 2),
            np.arange(240, -color_steps * 2, -color_steps * 2),
        ]
    )
    b_levels = np.clip(np.arange(0, 260, color_steps), 0, 255)
    color_levels = [np.array([el[0], 0, el[1]]) for el in zip(r_levels, b_levels)]
    intensity_step = (intensity_range[1] - intensity_range[0]) / len(color_levels)
    left_edges = np.arange(intensity_range[0], intensity_range[1], intensity_step)
    new_r = 255 * np.ones_like(img) if white_background else np.zeros_like(img)
    new_b = 255 * np.ones_like(img) if white_background else np.zeros_like(img)
    new_g = 255 * np.ones_like(img) if white_background else np.zeros_like(img)
    for i, edge in enumerate(left_edges):
        new_r[np.logical_and(img >= edge, img <= edge + intensity_step)] = r_levels[i]
        new_g[np.logical_and(img >= edge, img <= edge + intensity_step)] = g_levels[i]
        new_b[np.logical_and(img >= edge, img <= edge + intensity_step)] = b_levels[i]

    new_img = np.stack((new_r, new_g, new_b), axis=2).astype("uint8")
    return (new_img, nan_mask) if return_nan_mask else new_img


def bin_metrics(
    metric: list | np.ndarray,
    bins: int | list = 15,
    bounds: tuple | None = None,
) -> np.ndarray:
    """Creates bins of metrics values

    Parameters
    ----------
    metric : list | np.ndarray
    bins : int | list, optional
       Number of bins or list of left edges, by default 15
    bounds : tuple
        Bounds to clip the metric values

    Returns
    -------
    new metric, bin left edges and bin interval
    """

    if type(metric) is list:
        metric = np.array(metric)

    if bounds is not None:
        metric = np.clip(metric, bounds[0], bounds[1])
    else:
        bounds = (np.min(metric), np.max(metric))

    if type(bins) is list:
        left_edges = bins
        round_left_edges = [np.round(e, 2) for e in left_edges]
        if np.round(np.max(metric), 2) not in round_left_edges:
            left_edges = left_edges + [np.max(metric).tolist()]
        if np.round(np.min(metric), 2) not in round_left_edges:
            left_edges = [np.min(metric).tolist()] + left_edges
        step_vals = np.diff(left_edges).tolist()
        left_edges = left_edges[:-1]
    else:
        step_val = (bounds[1] - bounds[0]) / bins
        left_edges = np.arange(bounds[0], bounds[1], step_val)
        step_vals = [step_val.tolist()] * len(left_edges)

    new_metric = np.zeros_like(metric).astype("float")
    diff_list = np.zeros_like(metric).astype("float")
    for i, edge in enumerate(left_edges):
        right_condition = metric < edge + step_vals[i]
        if i == len(left_edges) - 1:
            right_condition = metric <= edge + step_vals[i]
        new_metric[np.logical_and(metric >= edge, right_condition)] = (
            edge + step_vals[i] / 2
        )
        diff_list[np.logical_and(metric >= edge, right_condition)] = step_vals[i]

    return (
        new_metric,
        left_edges,
        (step_vals[0] if type(bins) is int else step_vals),
        diff_list.tolist(),
    )


def kml_to_poly(
    kml_file: Path,
) -> Polygon:
    """Reads KML file and returns Shaply Polygon

    Parameters
    ----------
    kml_file : Path

    Returns
    -------
    Polygon
    """
    driver = ogr.GetDriverByName("KML")
    datasource = driver.Open(kml_file)
    layer = datasource.GetLayer()
    feat = layer.GetNextFeature()
    geom = feat.geometry()
    poly = geom.ExportToIsoWkt()
    return ops.transform(lambda x, y, z=None: (x, y), from_wkt(poly))
