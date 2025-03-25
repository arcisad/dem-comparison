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
from dem_handler.download.aio_aws import bulk_upload_dem_tiles
from pathlib import Path
import aioboto3
import yaml

logging.basicConfig(level=logging.INFO)

yaml_file = input("Enter path to credential file:")
with open(yaml_file) as f:
    cred_dict = yaml.safe_load(f)

AWS_ACCESS_KEY_ID = cred_dict["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = cred_dict["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = cred_dict["AWS_DEFAULT_REGION"]


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
diff_arrays = [Path(p) for p in glob.glob(f"{location_name}_Outputs/dem_diff/*.tif")]
diff_mos = Path(f"{location_name}_Outputs/{location_name}_mosaics/diff_mos.tif")
if diff_mos.exists():
    os.remove(diff_mos)
simple_mosaic(diff_arrays, diff_mos)


pkls = sorted(glob.glob(f"{location_name}_Outputs/dem_diff_metrics/*.pkl"))
metrics, x, y = read_metrics(pkls)
metric_strs = ["me", "std", "mse", "nmad"]
for i, m in enumerate(metrics):
    dems = sorted(glob.glob(f"{location_name}_Outputs/dem_diff/*.tif"))
    simple_mosaic(
        dems,
        f"{location_name}_Outputs/{location_name}_mosaics/dem_mos_{metric_strs[i]}.tif",
        fill_value=m,
    )


session = aioboto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)
s3_dir = Path(f"dem_comparison_results/{location_name}_Outputs")
local_dir = Path(f"{location_name}_Outputs")
bulk_upload_dem_tiles(
    s3_dir,
    local_dir,
    session=session,
    num_cpus=4,
)
