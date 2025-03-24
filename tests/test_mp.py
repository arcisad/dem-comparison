from dem_comparison.analysis import analyse_difference_for_interval
from pathlib import Path

if __name__ == "__main__":
    analyse_difference_for_interval(
        range(-72, -70),
        range(65, 67),
        temp_path=Path("TEMP_Range"),
        save_dir_path=Path("TEMP_Range_Ouputs"),
        use_multiprocessing=True,
        query_num_tasks=4,
        keep_temp_files=False,
        return_outputs=False,
        download_files_first=True,
        num_cpus=4,
    )
