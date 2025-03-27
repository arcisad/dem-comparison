from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.profiles import Profile
from pathlib import Path
import numpy as np
import rioxarray
import rasterio as rio
import multiprocessing as mp
from osgeo import gdal
from dem_handler.utils.spatial import resize_bounds, BoundingBox
import os
import pickle
import shutil

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
        tif_1_clipped = tif_1_reproj.rio.clip([roi_poly], CRS.from_epsg(4326))
        tif_2_clipped = tif_2_reproj.rio.clip([roi_poly], CRS.from_epsg(4326))
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
    resolution: str = "lowest",
    bounds_scale_factor: float = 1.02,
    keep_vrt: bool = False,
    fill_value: float | int | list[float] | list[int] | None = None,
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
    """
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
    bounds = resize_bounds(BoundingBox(*bounds), bounds_scale_factor).bounds
    VRT_options = gdal.BuildVRTOptions(
        resolution=resolution,
        outputBounds=bounds,
        VRTNodata=np.nan,
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


def downsample_dataset(
    dataset_path: str,
    scale_factor: float | list[float] = 1.0,
    output_file: str = "",
    enhance_function=None,
    force_shape: tuple = (),  # (height, width)
):
    """
    Downsamples the output data and returns the new downsampled data and its new affine transformation according to `scale_factor`
    The output shape could also be forced using `forced_shape` parameter.
    """
    with rio.open(dataset_path) as dataset:
        # resample data to target shape
        if type(scale_factor) == float:
            scale_factor = [scale_factor] * 2
        if len(force_shape) != 0:
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

        if enhance_function is not None:
            data = enhance_function(data)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )

        if len(force_shape) != 0:
            scale_factor = [
                force_shape[0] / scale_factor[0],
                force_shape[1] / scale_factor[1],
            ]

        profile = dataset.profile
        profile.update(
            transform=transform,
            width=data.shape[2],
            height=data.shape[1],
            dtype=data.dtype,
        )

    if output_file != "":
        with rio.open(output_file, "w", **profile) as ds:
            for i in range(0, profile["count"]):
                ds.write(data[i], i + 1)

    return data, transform
