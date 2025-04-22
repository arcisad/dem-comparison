from dem_handler.dem.rema import BBox
from dem_handler.utils.spatial import BoundingBox
from pathlib import Path
from dem_comparison.utils import *
from dem_handler.utils.spatial import resize_bounds, transform_polygon
from dem_handler.dem.rema import get_rema_dem_for_bounds
from dem_handler.dem.geoid import remove_geoid
from dem_handler.dem.cop_glo30 import get_cop30_dem_for_bounds
from dem_handler.download.aws import download_egm_08_geoid
from dem_handler.download.aws import download_rema_tiles, download_cop_glo30_tiles
import multiprocessing as mp
from itertools import product
import shutil
import pickle
import geopandas as gpd
from dem_handler.download.aio_aws import bulk_upload_dem_tiles
from dem_handler.utils.rio_tools import reproject_arr_to_new_crs
import aioboto3
import glob
from shapely import from_wkt
import gemgis as gg
from rasterio.enums import Resampling
import yaml


def analyse_difference(
    lat_range: range | list[float] = range(-90, -50),
    lon_range: range | list[float] = range(-180, 180),
    rema_resolution: int = 10,
    interval: int = 10,
    range_index_buffer: int = 1,
    save_dir_path: Path | None = None,
    use_multiprocessing: bool = True,
    num_cpus: int = mp.cpu_count(),
    mp_chunk_size: int = 20,
    query_num_cpus: int = 1,
    query_num_tasks: int | None = -1,
    return_outputs: bool = False,
    download_files_first: bool = True,
    session: aioboto3.Session | None = None,
) -> tuple | None:
    """Analyse the difference between REMA and COP30 DEMs

    Parameters
    ----------
    lat_range : range | list[float], optional
        Latitude range, by default range(-90, -50)
    lon_range : range | list[float], optional
        Longitude range, by default range(-180, 180)
    rema_resolution : int, optional
        Desired resolution of REMA DEM (2, 10, 32), by default 10
    interval : int, optional
        Interval for bacth processing in degrees, by default 10
    range_index_buffer : int, optional
        Buffers the desired ramge by the number of indexes, by default 1
    save_dir_path : Path | None, optional
        Directory to save output data to, by default None
    use_multiprocessing : bool, optional
        Use multi-processing, by default True
    num_cpus : int, optional
        Number of cpus if using multi-processing, by default mp.cpu_count()
    mp_chunk_size : int, optional
        Chunk size for multi-processing, by default 20
    query_num_cpus : int, optional
        Number of cpus for paralle downloads of REMA DEMs, by default 1
        If set to -1, all available cpus will be used.
    query_num_tasks: int | None, optional
        Number of tasks. Each task will run in async mode on each cpu, by default None, which turns off async mode.
        If set to -1, all DEMs will be downloaded in async mode in on single task on one cpu only.
    return_outputs: bool, optional
        Returns the output of anlysis, by default Fasle.
    download_files_first, bool, optional
        Downloads all the required input files first, by default True
    session: aioboto3.Session | None, optional
        If passed (with right credentials), the results will be uploaded to s3 and the local data will be removed, by default None

    Returns
    -------
    tuple
        REMA DEMs and COP30 DEMs if `return_output` is set otherwise None.
    """
    if type(lat_range) is range:
        lat_range = list(range(lat_range.start, lat_range.stop, interval))
    if type(lon_range) is range:
        lon_range = list(range(lon_range.start, lon_range.stop, interval))

    lat_range = buffer_range(lat_range, range_index_buffer)
    lon_range = buffer_range(lon_range, range_index_buffer)

    lat_pairs = [range(*l) for l in get_pairs(lat_range, 2)]
    lon_pairs = [range(*l) for l in get_pairs(lon_range, 2)]
    combinations = list(product(lat_pairs, lon_pairs))

    final_rema_array_list = []
    final_cop_array_list = []
    for c in combinations:
        outputs = analyse_difference_for_interval(
            c[0],
            c[1],
            rema_resolution=rema_resolution,
            save_dir_path=save_dir_path,
            use_multiprocessing=use_multiprocessing,
            num_cpus=num_cpus,
            mp_chunk_size=mp_chunk_size,
            query_num_cpus=query_num_cpus,
            query_num_tasks=query_num_tasks,
            download_files_first=download_files_first,
        )

        if session:
            s3_dir = Path(f"dem_comparison_results/{save_dir_path}")
            local_dir = Path(save_dir_path)
            bulk_upload_dem_tiles(
                s3_dir,
                local_dir,
                session=session,
                num_cpus=num_cpus,
            )
            shutil.rmtree(save_dir_path)

        if return_outputs:
            final_rema_array_list.append(outputs[0])
            final_cop_array_list.append(outputs[1])

    return (
        (
            final_rema_array_list,
            final_cop_array_list,
        )
        if return_outputs
        else None
    )


def analyse_difference_for_interval(
    lat_range: range | list[float],
    lon_range: range | list[float],
    rema_index_path: Path = Path("dem_comparison/data/REMA_Mosaic_Index_v2.gpkg"),
    cop30_index_path: Path = Path("dem_comparison/data/copdem_tindex_filename.gpkg"),
    rema_resolution: int = 10,
    save_dir_path: Path | None = None,
    range_index_buffer: int = 1,
    temp_path: Path = Path("TMP"),
    use_multiprocessing: bool = True,
    num_cpus: int = mp.cpu_count(),
    mp_chunk_size: int = 20,
    query_num_cpus: int = 1,
    query_num_tasks: int | None = -1,
    keep_temp_files: bool = False,
    return_outputs: bool = False,
    download_files_first: bool = True,
) -> tuple | None:
    """Analyse the difference between REMA and COP30 DEMs fir the given interval in degrees.

    Parameters
    ----------
    lat_range : range | list[float]
        Latitude range
    lon_range : range | list[float]
        Longitude range
    rema_index_path : Path, optional
        REMA index file, by default Path("dem_comparison/data/REMA_Mosaic_Index_v2.gpkg")
    cop30_index_path : Path, optional
        COP30 index file, by default Path("dem_comparison/data/copdem_tindex_filename.gpkg")
    rema_resolution : int, optional
        Desired REMA resolution (2, 20, 32), by default 10
    save_dir_path : Path | None, optional
        Directory to save output data to, by default None
    range_index_buffer : int, optional
        Number of elements as buffer for the range, by default 1
    temp_path : Path, optional
        Temporary path, by default Path("TMP")
    use_multiprocessing : bool, optional
        Use multi-processing, by default True
    num_cpus : int, optional
        Number of cpus if using multi-processing, by default mp.cpu_count()
    mp_chunk_size : int, optional
        Chunk size for multi-processing, by default 20
    query_num_cpus : int, optional
        Number of cpus for paralle downloads of REMA DEMs, by default 1
        If set to -1, all available cpus will be used.
    query_num_tasks: int | None, optional
        Number of tasks. Each task will run in async mode on each cpu, by default None, which turns off async mode.
        If set to -1, all DEMs will be downloaded in async mode in on single task on one cpu only.
    keep_temp_files: bool, optional,
        Flag to keep temporary files, by default False.
    return_outputs: bool, optional
        Returns the output of anlysis, by default False.
    download_files_first, bool, optional
        Downloads all the required input files first, by default True

    Returns
    -------
    tuple
        REMA DEMs and COP30 DEMs if `return_output` is set otherwise None.

    Raises
    ------
    e
        Error in multi-processing job fails.
    """

    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)

    lat_range = list(lat_range)
    lon_range = list(lon_range)

    if len(lat_range) < 2:
        lat_range = [lat_range[0], lat_range[0] + 1]
    if len(lon_range) < 2:
        lon_range = [lon_range[0], lon_range[0] + 1]

    lat_range = buffer_range(lat_range, range_index_buffer)
    lon_range = buffer_range(lon_range, range_index_buffer)

    lat_pairs = get_pairs(lat_range, 2)
    lon_pairs = get_pairs(lon_range, 2)
    combinations = list(product(lat_pairs, lon_pairs))

    bounds_list = [
        resize_bounds(BoundingBox(c[1][0], c[0][0], c[1][1], c[0][1]), 0.1)
        for c in combinations
    ]

    if download_files_first:
        cop_download_dir = temp_path / "COP_DEMS"
        if not cop_download_dir.exists():
            cop_download_dir.mkdir(parents=True, exist_ok=True)

        full_cop_dem_paths = []
        full_rema_dem_paths = []
        for bounds in bounds_list:
            cop_dem_paths = get_cop30_dem_for_bounds(
                bounds,
                save_path=temp_path / "TEMP_COP.tif",
                ellipsoid_heights=False,
                cop30_index_path=cop30_index_path,
                download_dir=temp_path / "COP_DEMS",
                return_paths=True,
                download_dem_tiles=True,
                num_tasks=-1,
            )
            full_cop_dem_paths.extend(cop_dem_paths)

            rema_dem_paths = get_rema_dem_for_bounds(
                resize_bounds(bounds, 10.0),
                save_path=temp_path / "TEMP_REMA.tif",
                rema_index_path=rema_index_path,
                resolution=rema_resolution,
                bounds_src_crs=4326,
                ellipsoid_heights=False,
                download_dir=temp_path / "REMA_DEMS",
                return_paths=True,
                num_tasks=-1,
            )
            full_rema_dem_paths.extend(rema_dem_paths)

        full_cop_dem_paths = np.unique(full_cop_dem_paths).tolist()
        to_download = [f for f in full_cop_dem_paths if not f.exists()]
        print(f"Num COP DEMs to download: {len(to_download)}")
        if len(to_download) > 0:
            download_cop_glo30_tiles(
                [Path(url.name) for url in to_download],
                save_folder=temp_path / "COP_DEMS",
                num_cpus=num_cpus,
                num_tasks=num_cpus,
            )

        rema_layer = f"REMA_Mosaic_Index_v2_{rema_resolution}m"
        rema_index_df = gpd.read_file(rema_index_path, layer=rema_layer)
        full_rema_dem_paths = list(filter(lambda x: x, full_rema_dem_paths))
        full_rema_dem_paths = np.unique(full_rema_dem_paths).tolist()
        to_download = [f for f in full_rema_dem_paths if not f.exists()]
        print(f"Num REMA DEMs to download: {len(to_download)}")
        rema_file_names = [f.name for f in to_download]
        rema_urls = [
            Path(url)
            for url in rema_index_df.s3url
            if Path(url.replace(".json", "_dem.tif")).name in rema_file_names
        ]
        download_rema_tiles(
            rema_urls,
            save_folder=temp_path / "REMA_DEMS",
            num_cpus=num_cpus,
            num_tasks=num_cpus,
        )

    if use_multiprocessing:
        with mp.Pool(processes=num_cpus) as p:
            try:
                outputs = p.starmap(
                    query_dems,
                    [
                        (
                            bounds,
                            temp_path,
                            rema_index_path,
                            cop30_index_path,
                            rema_resolution,
                            save_dir_path,
                            None,
                            None,
                            query_num_cpus,
                            query_num_tasks,
                            True,
                            return_outputs,
                            download_files_first,
                        )
                        for bounds in bounds_list
                    ],
                    chunksize=mp_chunk_size,
                )
                if return_outputs:
                    outputs = list(filter(lambda el: all(i for i in el), outputs))
                    rema_array_list = [o[0] for o in outputs]
                    cop_array_list = [o[1] for o in outputs]
            except Exception as e:
                raise e
    else:
        rema_array_list = []
        cop_array_list = []
        for bounds in bounds_list:
            outputs = query_dems(
                bounds,
                temp_path,
                rema_index_path,
                cop30_index_path,
                rema_resolution,
                save_dir_path,
                num_cpus=query_num_cpus,
                num_tasks=query_num_tasks,
                keep_temp_files=True,
                return_outputs=return_outputs,
                download_files_first=download_files_first,
            )
            if return_outputs and (outputs[0] is not None):
                rema_array_list.append(outputs[0])
                cop_array_list.append(outputs[1])

    if not keep_temp_files:
        shutil.rmtree(temp_path, ignore_errors=True)

    return (rema_array_list, cop_array_list) if return_outputs else None


def query_dems(
    bounds: BBox,
    temp_path: Path = Path("TMP"),
    rema_index_path: Path = Path("dem_comparison/data/REMA_Mosaic_Index_v2.gpkg"),
    cop30_index_path: Path = Path("dem_comparison/data/copdem_tindex_filename.gpkg"),
    rema_resolution: int = 10,
    save_dir_path: Path | None = None,
    rema_save_path: Path | None = None,
    cop_save_path: Path | None = None,
    num_cpus: int = 1,
    num_tasks: int | None = None,
    keep_temp_files: bool = False,
    return_outputs: bool = True,
    download_files_first: bool = True,
) -> tuple | None:
    """Finds the DEMs for a given bounds

    Parameters
    ----------
    bounds : BBox
        Query bounding box.
    temp_path : Path, optional
        Temporary path, by default Path("TMP")
    rema_index_path : Path, optional
        REMA index file, by default Path("dem_comparison/data/REMA_Mosaic_Index_v2.gpkg")
    cop30_index_path : Path, optional
        COP30 index file, by default Path("dem_comparison/data/copdem_tindex_filename.gpkg")
    rema_resolution : int, optional
        Desired REMA DEM resolution (2, 10, 32), by default 10
    save_dir_path : Path | None, optional
        Directory to save output data to, by default None
    num_cpus : int, optional
        Number of cpus for paralle downloads of REMA DEMs, by default 1
        If set to -1, all available cpus will be used.
    num_tasks: int | None, optional
        Number of tasks. Each task will run in async mode on each cpu, by default None, which turns off async mode.
        If set to -1, all DEMs will be downloaded in async mode in on single task on one cpu only.
    keep_temp_files: bool, optional,
        Flag to keep temporary files, by default False.
    return_outputs: bool, optional
        Returns the output of anlysis, by default True.
    download_files_first, bool, optional
        Downloads all the required input files first, by default True

    Returns
    -------
    tuple
       REMA DEMs and COP30 DEMs if `return_output` is set otherwise None.
    """

    original_bounds = resize_bounds(bounds, 10.0)
    east_west = "E" if original_bounds.xmin > 0 else "W"
    north_south = "N" if original_bounds.ymax > 0 else "S"
    top_left_str = f"{int(np.abs(np.round(original_bounds.ymin)))}{north_south}_{int(np.abs(np.round(original_bounds.xmin)))}{east_west}"
    temp_path = temp_path / top_left_str

    if save_dir_path:
        original_rema_metrics_path = (
            save_dir_path / f"original_rema_metrics/{top_left_str}.pkl"
        )
        output_dem_diff_path = save_dir_path / f"dem_diff/{top_left_str}.tif"
        output_dem_metrics_path = save_dir_path / f"dem_diff_metrics/{top_left_str}.pkl"
        if (
            (original_rema_metrics_path.exists())
            and (output_dem_diff_path.exists())
            and (output_dem_metrics_path.exists())
        ):
            print(
                f"All files for tile with top left of {top_left_str} already exist. Skipping..."
            )
            return tuple([None] * 4)

    if not temp_path.exists():
        temp_path.mkdir(parents=True, exist_ok=True)

    _, _, downloaded_rema_files = get_rema_dem_for_bounds(
        original_bounds,
        save_path=temp_path / "TEMP_REMA.tif",
        rema_index_path=rema_index_path,
        resolution=rema_resolution,
        bounds_src_crs=4326,
        ellipsoid_heights=False,
        num_cpus=num_cpus,
        num_tasks=num_tasks,
        local_dem_dir=(
            Path(temp_path.parts[0]) / "REMA_DEMS" if download_files_first else None
        ),
        return_paths=False,
    )
    if (downloaded_rema_files is None) or (len(downloaded_rema_files) == 0):
        return tuple([None] * 4)

    if len(downloaded_rema_files) > 1:
        required_rema_dem = temp_path / "TEMP_REMA.tif"
    else:
        required_rema_dem = downloaded_rema_files[0]
        print(f"Required REMA DEM: {required_rema_dem}")

    if save_dir_path:
        if not original_rema_metrics_path.parent.exists():
            original_rema_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        generate_metrics(
            required_rema_dem,
            save_path=original_rema_metrics_path,
        )

    _, _, downloaded_cop_files = get_cop30_dem_for_bounds(
        bounds,
        save_path=temp_path / "TEMP_COP.tif",
        ellipsoid_heights=False,
        cop30_index_path=cop30_index_path,
        cop30_folder_path=(
            Path(temp_path.parts[0]) / "COP_DEMS" if download_files_first else Path(".")
        ),
        download_dem_tiles=not download_files_first,
        return_paths=False,
        num_tasks=-1 if download_files_first else None,
    )
    if len(downloaded_cop_files) == 0:
        return tuple([None] * 4)

    required_cop_dem = temp_path / "TEMP_COP.tif"
    if len(downloaded_cop_files) == 1:
        required_cop_dem = downloaded_cop_files[0]
        print(f"Required COP DEM: {required_cop_dem}")

    temp_cop_dem_raster = rio.open(required_cop_dem)
    geoid_tif_path = temp_path / "geoid.tif"
    if not geoid_tif_path.exists():
        download_egm_08_geoid(geoid_tif_path, bounds=temp_cop_dem_raster.bounds)

    temp_cop_dem_profile = temp_cop_dem_raster.profile
    temp_cop_dem_array = temp_cop_dem_raster.read(1)
    if len(downloaded_cop_files) > 1:
        temp_cop_dem_profile.update({"nodata": np.nan})
        temp_cop_dem_array[temp_cop_dem_array == 0] = np.nan
    temp_cop_dem_raster.close()
    remove_geoid(
        dem_array=temp_cop_dem_array,
        dem_profile=temp_cop_dem_profile,
        geoid_path=geoid_tif_path,
        buffer_pixels=2,
        save_path=temp_path / "TEMP_COP_ELLIPSOID.tif",
    )
    required_cop_dem = temp_path / "TEMP_COP_ELLIPSOID.tif"

    cop_array, rema_array = reproject_match_tifs(
        required_cop_dem,
        required_rema_dem,
        target_crs=3031,
        verbose=True,
        convert_dB=False,
        save_path_1=cop_save_path if cop_save_path else "",
        save_path_2=rema_save_path if rema_save_path else "",
    )
    if save_dir_path:
        if not output_dem_diff_path.parent.exists():
            output_dem_diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_array = rema_array - cop_array
        diff_array.rio.write_nodata(np.nan, inplace=True)
        diff_array.rio.to_raster(output_dem_diff_path)
        if not output_dem_metrics_path.parent.exists():
            output_dem_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        generate_metrics(
            np.squeeze(diff_array.to_numpy()),
            save_path=output_dem_metrics_path,
        )
        diff_array.close()
        del diff_array

    if not keep_temp_files:
        shutil.rmtree(temp_path.parts[0])

    if return_outputs:
        rema_array = np.squeeze(rema_array.to_numpy())
        cop_array = np.squeeze(cop_array.to_numpy())
    else:
        rema_array.close()
        cop_array.close()

        del rema_array
        del cop_array

    return (rema_array, cop_array) if return_outputs else None


def generate_metrics(
    dataset1: np.ndarray | Path,
    dataset2: np.ndarray | Path | None = None,
    save_path: Path | None = None,
) -> tuple:
    """Generates statistical metrics for an error array, or a calculated error array if `array2` is provided.

    Parameters
    ----------
    dataset1 : np.ndarray | Path
        Input array
    dataset1 : np.ndarray | Path | None, optional
        Second array, if provided the operations will be carried out on the dofference between the first array and this one, by default None
    save_path : Path | None, optional
        Path to dump a pickle of the results, by default None

    Returns
    -------
    tuple
        Statistical metrics
        (Mean Absolute Error, Standard Deviation of Error, Mean Squared Error, Normalised Median Absolute Deviation of Error)
    """
    if type(dataset1) is not np.ndarray:
        with rio.open(dataset1, "r") as ds:
            array1 = ds.read(1)
    else:
        array1 = dataset1

    if dataset2:
        if type(dataset2) is not np.ndarray:
            with rio.open(dataset2, "r") as ds:
                array2 = ds.read(1)
        else:
            array2 = dataset2
        diff_array = array1 - array2
    else:
        diff_array = array1

    diff_array = diff_array[~np.isnan(diff_array)]

    me = diff_array.mean()
    std = diff_array.std()
    mse = np.square(diff_array).mean()
    median = np.median(diff_array)
    nmad = 1.4826 * np.median(np.abs(diff_array - median))

    metrics = me, std, mse, nmad
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(metrics, f)
    return metrics


def area_of_interest(
    bounds_polygon: Polygon | str,
    src_crs: int,
    aoi_name: str,
    rema_index_path: Path = Path("../dem_comparison/data/REMA_Mosaic_Index_v2.gpkg"),
    cop30_index_path: Path = Path("../dem_comparison/data/copdem_tindex_filename.gpkg"),
    num_cpu: int = 4,
    slope_maps: bool = True,
    return_outputs: bool = False,
    keep_temp_files: bool = False,
    desired_resolution: float | tuple = 30.0,
    s3_upload_dir: str | None = None,
    credentials_file: Path | None = None,
    save_path: Path = Path(""),
    temp_path: Path = Path(""),
) -> list[Path]:
    """Generates mosaics for a give area of interest

    Parameters
    ----------
    bounds_polygon : Polygon | str
        Shapely Polygon or a WKT compatible str
    src_crs : int
        CRS of the bounds polygon
    aoi_name : str
        Name of the area of the interest. It will be used in the generated directory and file names.
    rema_index_path : Path, optional
        Path to REMA index file, by default Path("../dem_comparison/data/REMA_Mosaic_Index_v2.gpkg")
    cop30_index_path : Path, optional
        Path to COP30 index file, by default Path("../dem_comparison/data/copdem_tindex_filename.gpkg")
    num_cpu : int, optional
        Number of CPUs for multi-processing, by default 4
    slope_maps : bool, optional
        Whether to generate slope maps or not, by default True
    return_outputs : bool, optional
        Only returns the output file paths, by default False
    keep_temp_files : bool, optional
        Keeping the intermediate files, by default False
    desired_resolution: float | tuple, optional,
        desired resolution for output mosaics, by default 30.0
    s3_upload_dir: str | None, optional
        If provided, the ouputs will be uploaded to S3 dir,
        pass `auto` to infer the S3 Dir from `aoi_name`, which puts the files inside a subfolder in `dem_comparison_results` directory,
        Bucket is hardcoded to `deant-data-public-dev` and the region is `ap-souteast-2`, by default None,
    credentials_file: Path | None, optional
        IF uploading to S3, a yaml file containing AWS credentials should be provided, by default None
    save_path: Path, optional
        Path to save the output files, by default Path("")
    temp_path: Path, optional
        Path to save the temporary files, by default Path("")

    Returns
    -------
    list[Path]
        List of output file paths.
    """

    diff_mos = Path(f"{save_path}/{aoi_name}_Outputs/{aoi_name}_mosaics/diff_mos.tif")
    matched_cop_mosaic = Path(
        f"{save_path}/{aoi_name}_Outputs/{aoi_name}_mosaics/matched_cop_mosaic.tif"
    )
    matched_rema_mosaic = Path(
        f"{save_path}/{aoi_name}_Outputs/{aoi_name}_mosaics/matched_rema_mosaic.tif"
    )

    slope_diff = Path(f"{save_path}/{aoi_name}_Outputs/{aoi_name}_mosaics/slope_diff.tif")
    matched_cop_slope = Path(
        f"{save_path}/{aoi_name}_Outputs/{aoi_name}_mosaics/matched_cop_slope.tif"
    )
    matched_rema_slope = Path(
        f"{save_path}/{aoi_name}_Outputs/{aoi_name}_mosaics/matched_rema_slope.tif"
    )

    outputs = [diff_mos, matched_rema_mosaic, matched_cop_mosaic]

    if slope_maps:
        outputs.extend([slope_diff, matched_rema_slope, matched_cop_slope])

    if return_outputs:
        return outputs

    AOI_CRS = 4326
    REMA_CRS = 3031

    if type(desired_resolution) is float:
        desired_resolution = (desired_resolution, desired_resolution)

    if type(bounds_polygon) is str:
        aoi_poly = from_wkt(bounds_polygon)
    else:
        aoi_poly = bounds_polygon

    if src_crs != AOI_CRS:
        aoi_bounds = transform_polygon(aoi_poly, src_crs, AOI_CRS).bounds
    else:
        aoi_bounds = aoi_poly.bounds

    extended_aoi_bounds = (
        *(np.floor(aoi_bounds[0:2]).astype("int")).tolist(),
        *(np.ceil(aoi_bounds[2:]).astype("int")).tolist(),
    )

    print(f"Analysing area of interest for boundary: {extended_aoi_bounds}")

    lat_range = range(extended_aoi_bounds[1], extended_aoi_bounds[3])
    lon_range = range(extended_aoi_bounds[0], extended_aoi_bounds[2])

    if temp_path == Path(""):
        temp_path = Path(f"TEMP_{aoi_name}")
    analyse_difference_for_interval(
        lat_range,
        lon_range,
        temp_path=temp_path,
        save_dir_path=Path(f"{save_path}/{aoi_name}_Outputs"),
        use_multiprocessing=True,
        query_num_tasks=None,
        keep_temp_files=True,
        return_outputs=False,
        num_cpus=num_cpu,
        rema_index_path=rema_index_path,
        cop30_index_path=cop30_index_path,
    )

    os.makedirs(f"{save_path}/{aoi_name}_Outputs/{aoi_name}_mosaics", exist_ok=True)

    if src_crs == REMA_CRS:
        rema_bounds = aoi_poly.bounds
    else:
        rema_bounds = transform_polygon(aoi_poly, src_crs, REMA_CRS).bounds

    cop_dems = glob.glob(f"{temp_path}/**/TEMP_COP_ELLIPSOID.tif")
    rema_dems = glob.glob(f"{temp_path}/**/TEMP_REMA.tif")

    cop_mosaic = Path(f"{save_path}/{aoi_name}_Outputs/cop_mosaic.tif")
    rema_mosaic = Path(f"{save_path}/{aoi_name}_Outputs/rema_mosaic.tif")

    simple_mosaic(
        cop_dems,
        cop_mosaic,
        force_bounds=aoi_bounds,
    )

    with rio.open(cop_mosaic, "r") as ds:
        profile = ds.profile
        img = ds.read(1)
        # This resampling mode gets smooth slopes and avoid aliasing stripes
        repr_img, repr_profile = reproject_arr_to_new_crs(
            img, profile, 3031, resampling="cubic"
        )
        repr_img = repr_img.squeeze()

    repr_cop_mosaic = Path(f"{save_path}/{aoi_name}_Outputs/repr_cop_mosaic.tif")

    with rio.open(repr_cop_mosaic, "w", **repr_profile) as ds:
        ds.write(repr_img, 1)

    simple_mosaic(
        rema_dems,
        rema_mosaic,
        force_bounds=rema_bounds,
        resolution=desired_resolution,
    )

    reproject_match_tifs(
        rema_mosaic,
        repr_cop_mosaic,
        target_crs=REMA_CRS,
        verbose=True,
        convert_dB=False,
        save_path_1=matched_rema_mosaic,
        save_path_2=matched_cop_mosaic,
    )

    with rio.open(matched_cop_mosaic, "r") as ds:
        cop_data = ds.read(1)
        cop_data_profile = ds.profile
        if slope_maps:
            cop_slope = gg.raster.calculate_slope(ds)

    with rio.open(matched_rema_mosaic, "r") as ds:
        rema_data = ds.read(1)
        rema_data_profile = ds.profile
        if slope_maps:
            rema_slope = gg.raster.calculate_slope(ds)

    diff_data = rema_data - cop_data
    with rio.open(diff_mos, "w", **rema_data_profile) as ds:
        ds.write(diff_data, 1)

    if slope_maps:
        with rio.open(matched_cop_slope, "w", **cop_data_profile) as ds:
            ds.write(cop_slope, 1)

        with rio.open(matched_rema_slope, "w", **rema_data_profile) as ds:
            ds.write(rema_slope, 1)

        slope_diff_array = rema_slope - cop_slope
        with rio.open(slope_diff, "w", **rema_data_profile) as ds:
            ds.write(slope_diff_array, 1)

    if not keep_temp_files:
        os.remove(cop_mosaic)
        os.remove(rema_mosaic)
        os.remove(repr_cop_mosaic)
        shutil.rmtree(Path(f"{temp_path}"))
        shutil.rmtree(Path(f"{save_path}/{aoi_name}_Outputs/dem_diff"))
        shutil.rmtree(Path(f"{save_path}/{aoi_name}_Outputs/dem_diff_metrics"))
        shutil.rmtree(Path(f"{save_path}/{aoi_name}_Outputs/original_rema_metrics"))

    if s3_upload_dir is not None:
        with open(credentials_file) as f:
            cred_dict = yaml.safe_load(f)
        AWS_ACCESS_KEY_ID = cred_dict["AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = cred_dict["AWS_SECRET_ACCESS_KEY"]
        AWS_DEFAULT_REGION = cred_dict["AWS_DEFAULT_REGION"]

        session = aioboto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION,
        )
        if s3_upload_dir == "auto":
            s3_dir = Path(f"dem_comparison_results/{aoi_name}_Outputs")
        local_dir = Path(f"{save_path}/{aoi_name}_Outputs")
        bulk_upload_files(
            s3_dir,
            local_dir,
            session=session,
            num_cpus=4,
        )

    return outputs
