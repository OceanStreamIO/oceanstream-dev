import os
import xarray as xr
from typing import Any, Dict
from xarray import Dataset

from .shoal_detection_handler import attach_shoal_mask_to_ds
from .shoal_process import process_shoals, write_shoals_to_csv


def get_shoals_list(ds: xr.Dataset, config) -> tuple[Any, dict, Dataset]:
    parameters = config["shoals"]["parameters"]
    method = config["shoals"]["method"]

    shoal_dataset = attach_shoal_mask_to_ds(ds, parameters=parameters, method=method)
    shoal_list = process_shoals(shoal_dataset)

    return shoal_list, profiling_info, shoal_dataset


def write_csv(ds: xr.Dataset, config):
    shoal_list = None
    shoal_dataset = None

    if config["shoals"]["enabled"]:
        shoal_list, shoal_dataset = get_shoals_list(ds, config)

        if config["export_csv"]:
            write_shoals_to_csv(
                shoal_list,
                os.path.join(
                    config["output_folder"], config["raw_path"].stem + "_fish_schools.csv"
                ),
            )

    return shoal_list, shoal_dataset
