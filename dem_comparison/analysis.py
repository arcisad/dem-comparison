from dem_handler.dem.rema import BBox
from dem_handler.utils.spatial import BoundingBox
from pathlib import Path
from dem_comparison.utils import *
from dem_handler.utils.spatial import resize_bounds
from dem_handler.dem.rema import get_rema_dem_for_bounds
from dem_handler.dem.cop_glo30 import get_cop30_dem_for_bounds
import multiprocessing as mp
from itertools import product
import shutil


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
) -> tuple:
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

    Returns
    -------
    tuple
        REMA DEMs, COP30 DEMs, Valid indexes in the arrays, Difference array
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
    final_rema_valid_idx_list = []
    final_dem_diff_list = []
    for c in combinations:
        rema_array_list, cop_array_list, rema_valid_idx_list, dem_diff_list = (
            analyse_difference_for_interval(
                c[0],
                c[1],
                rema_resolution,
                save_dir_path,
                use_multiprocessing=use_multiprocessing,
                num_cpus=num_cpus,
                mp_chunk_size=mp_chunk_size,
                query_num_cpus=query_num_cpus,
                query_num_tasks=query_num_tasks,
            )
        )

        final_rema_array_list.append(rema_array_list)
        final_cop_array_list.append(cop_array_list)
        final_rema_valid_idx_list.append(rema_valid_idx_list)
        final_dem_diff_list.append(dem_diff_list)

    return (
        final_rema_array_list,
        final_cop_array_list,
        final_rema_valid_idx_list,
        final_dem_diff_list,
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
) -> tuple:
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

    Returns
    -------
    tuple
        REMA DEMs, COP30 DEMs, Valid indexes in the arrays, Difference array

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
                        )
                        for bounds in bounds_list
                    ],
                    chunksize=mp_chunk_size,
                )
                outputs = list(filter(lambda el: all(i for i in el), outputs))
                rema_array_list = [o[0] for o in outputs]
                cop_array_list = [o[1] for o in outputs]
                rema_valid_idx_list = [o[2] for o in outputs]
                dem_diff_list = [o[3] for o in outputs]
            except Exception as e:
                raise e
    else:
        rema_array_list = []
        cop_array_list = []
        rema_valid_idx_list = []
        dem_diff_list = []
        for bounds in bounds_list:
            rema_array, cop_array, rema_valid_idx, dem_diff = query_dems(
                bounds,
                temp_path,
                rema_index_path,
                cop30_index_path,
                rema_resolution,
                save_dir_path,
                num_cpus=query_num_cpus,
                num_tasks=query_num_tasks,
            )
            if rema_array is not None:
                rema_array_list.append(rema_array)
                cop_array_list.append(cop_array)
                rema_valid_idx_list.append(rema_valid_idx)
                dem_diff_list.append(dem_diff)

    return rema_array_list, cop_array_list, rema_valid_idx_list, dem_diff_list


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
) -> tuple:
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

    Returns
    -------
    tuple
       REMA DEMs, COP30 DEMs, Valid indexes in the arrays, Difference array
    """

    original_bounds = resize_bounds(bounds, 10.0)
    east_west = "E" if original_bounds.xmin > 0 else "W"
    north_south = "N" if original_bounds.ymax > 0 else "S"
    top_left_str = f"{int(np.abs(np.round(original_bounds.xmin)))}{east_west}_{int(np.abs(np.round(original_bounds.ymax)))}{north_south}"
    temp_path = temp_path / top_left_str

    if not temp_path.exists():
        temp_path.mkdir(exist_ok=True)

    rema_array, _ = get_rema_dem_for_bounds(
        bounds,
        save_path=temp_path / "TEMP_REMA.tif",
        rema_index_path=rema_index_path,
        resolution=rema_resolution,
        bounds_src_crs=4326,
        ellipsoid_heights=False,
        num_cpus=num_cpus,
        num_tasks=num_tasks,
    )
    if rema_array is None:
        return tuple([None] * 4)

    get_cop30_dem_for_bounds(
        bounds,
        save_path=temp_path / "TEMP_COP.tif",
        ellipsoid_heights=False,
        cop30_index_path=cop30_index_path,
        download_dem_tiles=True,
    )

    cop_array, rema_array = reproject_match_tifs(
        temp_path / "TEMP_COP.tif",
        temp_path / "TEMP_REMA.tif",
        target_crs=3031,
        verbose=True,
        convert_dB=False,
        save_path_1=cop_save_path if cop_save_path else "",
        save_path_2=rema_save_path if rema_save_path else "",
    )
    if save_dir_path:
        diff_array_save_path = save_dir_path / f"dem_diff_{top_left_str}.tif"
        (rema_array - cop_array).rio.to_raster(diff_array_save_path)

    if not keep_temp_files:
        shutil.rmtree(temp_path, ignore_errors=True)

    rema_array = np.squeeze(rema_array.to_numpy())
    cop_array = np.squeeze(cop_array.to_numpy())

    rema_array[rema_array == 0] = np.nan
    rema_valid_idx = np.where(~np.isnan(rema_array))
    rema_array_valid = rema_array[rema_valid_idx]
    cop_array_valid = cop_array[rema_valid_idx]

    dem_diff = rema_array_valid - cop_array_valid

    return rema_array, cop_array, rema_valid_idx, dem_diff
