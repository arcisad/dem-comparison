from dem_comparison.analysis import analyse_difference
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--lat-range", type=range, default=range(-90, -50))
parser.add_argument("--lon-range", type=range, default=range(-180, 180))
parser.add_argument("--rema-resolution", type=int, default=10)
parser.add_argument("--interval", type=int, default=10)
parser.add_argument("--use-multiprocessing", action="store_false")
parser.add_argument("--save-dir-path", type=str, default="Antarctica_Dem_Comparison")

args = parser.parse_args()


if __name__ == "__main__":
    analyse_difference(
        lat_range=args.lat_range,
        lon_range=args.lon_range,
        rema_resolution=args.rema_resolution,
        interval=args.interval,
        use_multiprocessing=args.use_multiprocessing,
        save_dir_path=Path(args.save_dir_path),
    )
