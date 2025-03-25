from dem_handler.dem.rema import BBox
from dem_handler.utils.spatial import BoundingBox
from pathlib import Path
from dem_comparison.utils import *
from dem_handler.utils.spatial import resize_bounds
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
            rema_resolution,
            save_dir_path,
            use_multiprocessing=use_multiprocessing,
            num_cpus=num_cpus,
            mp_chunk_size=mp_chunk_size,
            query_num_cpus=query_num_cpus,
            query_num_tasks=query_num_tasks,
            download_files_first=download_files_first,
        )

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
            cop_download_dir.mkdir(exist_ok=True)

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

    if not temp_path.exists():
        temp_path.mkdir(exist_ok=True)

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
    if len(downloaded_rema_files) == 0:
        return tuple([None] * 4)

    if len(downloaded_rema_files) > 1:
        required_rema_dem = temp_path / "TEMP_REMA.tif"
    else:
        required_rema_dem = downloaded_rema_files[0]
        print(f"Required REMA DEM: {required_rema_dem}")

    if save_dir_path:
        if not save_dir_path.exists():
            save_dir_path.mkdir(exist_ok=True)
        generate_metrics(
            required_rema_dem,
            save_path=save_dir_path / f"original_rema_metrics_{top_left_str}.pkl",
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

    geoid_tif_path = temp_path / "geoid.tif"
    if not geoid_tif_path.exists():
        download_egm_08_geoid(geoid_tif_path, bounds=original_bounds.bounds)

    temp_cop_dem_raster = rio.open(required_cop_dem)
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
        diff_array_save_path = save_dir_path / f"dem_diff_{top_left_str}.tif"
        diff_array = rema_array - cop_array
        diff_array.rio.write_nodata(np.nan, inplace=True)
        diff_array.rio.to_raster(diff_array_save_path)
        generate_metrics(
            np.squeeze(diff_array.to_numpy()),
            save_path=save_dir_path / f"dem_diff_metrics_{top_left_str}.pkl",
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
