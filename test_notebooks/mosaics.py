import sys

sys.path.insert(1, "../")

from dem_comparison.utils import build_diff_mosaics, resample_dataset, simple_mosaic
from pathlib import Path
from tqdm import tqdm

mosaic_dir = Path("../Antarctica_Dem_Comparison_Ocean_Corrected/mosaics")
mosaic_dir.mkdir(exist_ok=True, parents=True)
diffs = list(
    Path("../Antarctica_Dem_Comparison_Ocean_Corrected/dem_diff/").glob("*.tif")
)

build_diff_mosaics(diffs, mosaic_dir)


mosaics = list(
    Path("../Antarctica_Dem_Comparison_Ocean_Corrected/mosaics").glob("*.tif")
)
resampled_mosaic_dir = Path(
    "../Antarctica_Dem_Comparison_Ocean_Corrected/resampled_mosaics"
)
resampled_mosaic_dir.mkdir(exist_ok=True, parents=True)
loop = tqdm(mosaics, total=len(mosaics))
for mosaic in loop:
    loop.set_description(f"Processing {mosaic.name}")
    resample_dataset(mosaic, 0.195, resampled_mosaic_dir / f"resampled_{mosaic.name}")

simple_mosaic(
    list(
        Path("../Antarctica_Dem_Comparison_Ocean_Corrected/resampled_mosaics").glob(
            "*.tif"
        )
    ),
    "../Antarctica_Dem_Comparison_Ocean_Corrected/mosaic_160m.tif",
)
