import os
from .csv import full_nasc_data, write_nasc_to_csv


def compute_and_write_nasc(shoal_dataset, config, profiling_info=None):
    if not config["nasc"]["enabled"]:
        return None

    start_time = None
    start_cpu = None
    start_memory = None

    if config["profile"]:
        start_time, start_cpu, start_memory = start_profiling()

    nasc = full_nasc_data(shoal_dataset)

    if config["profile"]:
        profiling_info["nasc"] = end_profiling(start_time, start_cpu, start_memory)

    write_nasc_to_csv(
        nasc, os.path.join(config["output_folder"], config["raw_path"].stem + "_NASC.csv")
    )
