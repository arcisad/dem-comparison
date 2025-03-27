import sys

sys.path.insert(1, "../")

import glob
from dem_comparison.utils import simple_mosaic
import numpy as np
from pathlib import Path
from dem_comparison.utils import get_pairs, buffer_range
from itertools import product

mosaic_dir = Path("../Antarctica_Dem_Comparison/mosaics")

diffs = glob.glob("../Antarctica_Dem_Comparison/dem_diff/*")
splits = [Path(f).stem.split("_") for f in diffs]
lats = [-int(el[0][:-1]) for el in splits]
lons = [int(el[1][:-1]) * (-1 if el[1][-1] == "W" else 1) for el in splits]

interval = 10
lat_range = range(-90, -50)
lon_range = range(-180, 180)

lat_range = list(range(lat_range.start, lat_range.stop, interval))
lon_range = list(range(lon_range.start, lon_range.stop, interval))

lat_range = buffer_range(lat_range, 1)
lon_range = buffer_range(lon_range, 1)

lat_pairs = [range(*l) for l in get_pairs(lat_range, 2)]
lon_pairs = [range(*l) for l in get_pairs(lon_range, 2)]
combinations = list(product(lat_pairs, lon_pairs))

for i, c in enumerate(combinations):
    lon_start = "W" if c[1].start < 0 else "E"
    lon_stop = "W" if c[1].stop < 0 else "E"
    combination_str = f"{abs(c[0].start)}S_{abs(c[1].start)}{lon_start}_{abs(c[0].stop)}S_{abs(c[1].stop)}{lon_stop}"
    lat_cond = np.logical_and(np.array(lats) >= c[0].start, np.array(lats) < c[0].stop)
    lon_cond = np.logical_and(np.array(lons) >= c[1].start, np.array(lons) < c[1].stop)
    chunk = np.array(diffs)[np.logical_and(lat_cond, lon_cond)].tolist()
    if len(chunk) == 0:
        print(f"No diff files found for {combination_str}. Skipping...")
        continue
    print(
        f"Mosaicing chunk {combination_str}. {i + 1} of {len(combinations)} combinations."
    )
    mosaic_name = f"{combination_str}.tif"
    simple_mosaic(chunk, mosaic_dir / mosaic_name)
