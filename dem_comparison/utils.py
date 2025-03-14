from rasterio.enums import Resampling
from rasterio.crs import CRS
from pathlib import Path
import numpy as np
import rioxarray

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
