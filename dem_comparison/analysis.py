from dem_handler.dem.rema import BBox
from dem_handler.utils.spatial import BoundingBox
from pathlib import Path
from utils import *
from dem_handler.utils.spatial import resize_bounds
from dem_handler.dem.rema import get_rema_dem_for_bounds
from dem_handler.dem.cop_glo30 import get_cop30_dem_for_bounds
import pickle
import multiprocessing as mp
from itertools import product


def analyse_difference(
    lat_range: range | list[float] = range(-90, -50),
    lon_range: range | list[float] = range(-180, 180),
    rema_resolution: int = 10,
    interval: int = 10,
    range_index_buffer: int = 1,
    save_dir_path: Path | None = None,
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
    for i, c in enumerate(combinations):
        rema_array_list, cop_array_list, rema_valid_idx_list, dem_diff_list = (
            analyse_difference_for_interval(
                c[0],
                c[1],
                rema_resolution,
            )
        )
        if save_dir_path:
            save_path = (
                save_dir_path / f"range_{c[0].start}_{c[1].start}_chunk_num_{i}.pkl"
            )
            pickle.dump(
                (rema_array_list, cop_array_list, rema_valid_idx_list, dem_diff_list),
                open(save_path, "wb"),
            )
        else:
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
    range_index_buffer: int = 1,
    temp_path: Path = Path("TMP"),
    use_multiprocessing: bool = True,
    num_cpu: int = mp.cpu_count(),
    mp_chunk_size: int = 20,
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
    range_index_buffer : int, optional
        Number of elements as buffer for the range, by default 1
    temp_path : Path, optional
        Temporary path, by default Path("TMP")
    use_multiprocessing : bool, optional
        Use multi-processing, by default True
    num_cpu : int, optional
        Number of cpus if using multi-processing, by default mp.cpu_count()
    mp_chunk_size : int, optional
        Chunk size for multi-processing, by default 20

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
        with mp.Pool(processes=num_cpu) as p:
            try:
                outputs = p.starmap(
                    get_output_arrays,
                    [
                        (
                            bounds,
                            temp_path,
                            rema_index_path,
                            cop30_index_path,
                            rema_resolution,
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
            rema_array, cop_array, rema_valid_idx, dem_diff = get_output_arrays(
                bounds, temp_path, rema_index_path, cop30_index_path, rema_resolution
            )
            if rema_array:
                rema_array_list.append(rema_array)
                cop_array_list.append(cop_array)
                rema_valid_idx_list.append(rema_valid_idx)
                dem_diff_list.append(dem_diff)

    return rema_array_list, cop_array_list, rema_valid_idx_list, dem_diff_list


def get_output_arrays(
    bounds: BBox,
    temp_path: Path = Path("TMP"),
    rema_index_path: Path = Path("dem_comparison/data/REMA_Mosaic_Index_v2.gpkg"),
    cop30_index_path: Path = Path("dem_comparison/data/copdem_tindex_filename.gpkg"),
    rema_resolution: int = 10,
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

    Returns
    -------
    tuple
       REMA DEMs, COP30 DEMs, Valid indexes in the arrays, Difference array
    """
    rema_array, _ = get_rema_dem_for_bounds(
        bounds,
        save_path=temp_path / Path("TEM_REMA.tif"),
        rema_index_path=rema_index_path,
        resolution=rema_resolution,
        bounds_src_crs=4326,
        ellipsoid_heights=False,
    )
    if not rema_array:
        return tuple([None] * 4)

    get_cop30_dem_for_bounds(
        bounds,
        save_path=temp_path / Path("TEM_COP.tif"),
        ellipsoid_heights=False,
        cop30_index_path=cop30_index_path,
        download_dem_tiles=True,
    )

    cop_array, rema_array = reproject_match_tifs(
        temp_path / "TEM_COP.tif",
        temp_path / "TEM_REMA.tif",
        target_crs=3031,
        verbose=True,
        convert_dB=False,
    )

    rema_array[rema_array == 0] = np.nan
    rema_valid_idx = np.where(~np.isnan(rema_array))
    rema_array_valid = rema_array[rema_valid_idx]
    cop_array_valid = cop_array[rema_valid_idx]

    dem_diff = rema_array_valid - cop_array_valid

    return rema_array, cop_array, rema_valid_idx, dem_diff
