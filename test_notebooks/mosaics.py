import sys

sys.path.insert(1, "../")

from dem_comparison.utils import build_diff_mosaics
from pathlib import Path

mosaic_dir = Path("../Antarctica_Dem_Comparison/mosaics")
diffs = list(Path("Antarctica_Dem_Comparison/dem_diff/").glob("*.tif"))

build_diff_mosaics(diffs, mosaic_dir)
