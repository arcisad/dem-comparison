import sys
sys.path.insert(1, "../")

from dem_comparison.analysis import analyse_difference_for_interval
from pathlib import Path
import shapely
import numpy as np
import geopandas as gpd
from dem_handler.utils.spatial import transform_polygon
import logging
import os
from dem_comparison.utils import simple_mosaic, read_metrics
import glob

logging.basicConfig(level=logging.INFO)

location_dict = {
    "Casey": 0,
    "Amery": 1,
    "Fimbulheimen": 2,
    "Ross": 3,
    "McMurdo": 4,
    "Abbot": 5,
}

location_name = "Amery"  # "Abbot"

gdf = gpd.read_file("Antarctic_DEM_AOI.geojson")
geoms = gdf.geometry

polygons = []
for g in geoms:
    if type(g) == shapely.geometry.multipolygon.MultiPolygon:
        polygons.extend(list(g.geoms))
    else:
        polygons.append(g)

# bounds are in polar stereographic epsg:3031
bounds_list = [transform_polygon(p, 3031, 4326).bounds for p in polygons]
extended_bounds = []
for b in bounds_list:
    extended_bounds.append(
        (
            *(np.floor(b[0:2]).astype("int")).tolist(),
            *(np.ceil(b[2:]).astype("int")).tolist(),
        )
    )
lat_ranges = []
lon_ranges = []
for eb in extended_bounds:
    lat_ranges.append(range(eb[1], eb[3]))
    lon_ranges.append(range(eb[0], eb[2]))


analyse_difference_for_interval(
    lat_ranges[location_dict[location_name]],
    lon_ranges[location_dict[location_name]],
    temp_path=Path(f"TEMP_{location_name}"),
    save_dir_path=Path(f"{location_name}_Outputs"),
    use_multiprocessing=True,
    query_num_tasks=None,
    keep_temp_files=False,
    return_outputs=False,
    num_cpus=4,
    rema_index_path=Path("../dem_comparison/data/REMA_Mosaic_Index_v2.gpkg"),
    cop30_index_path=Path("../dem_comparison/data/copdem_tindex_filename.gpkg"),
)

os.makedirs(f"{location_name}_Outputs/{location_name}_mosaics", exist_ok=True)
diff_arrays = [Path(p) for p in glob.glob(f"{location_name}_Outputs/*.tif")]
diff_mos = Path(f"{location_name}_Outputs/{location_name}_mosaics/diff_mos.tif")
if diff_mos.exists():
    os.remove(diff_mos)
simple_mosaic(diff_arrays, diff_mos)


pkls = sorted(glob.glob(f"{location_name}_Outputs/dem_diff**.pkl"))
metrics, x, y = read_metrics(pkls)
dems = sorted(glob.glob(f"{location_name}_Outputs/dem_diff**.tif"))
metric_strs = ["me", "std", "mse", "nmad"]
simple_mosaic(
    dems,
    f"{location_name}_Outputs/{location_name}_mosaics/dem_mos_{metric_strs[0]}.tif",
    fill_value=metrics[0],
)