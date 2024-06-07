import concurrent.futures
import time
from collections.abc import Iterable
from typing import Dict

import xarray

# from echopype.clean.api import (
#     get_attenuation_mask_multichannel,
#     get_impulse_noise_mask_multichannel,
#     get_transient_noise_mask_multichannel,
# )
# from echopype.mask.api import get_seabed_mask_multichannel

from oceanstream.utils import add_metadata_to_mask, dict_to_formatted_list

from .types import DenoiseConfig

MASK_CREATION_FUNCTIONS = {
    # "transient": get_transient_noise_mask_multichannel,
    # "attenuation": get_attenuation_mask_multichannel,
    # "impulse": get_impulse_noise_mask_multichannel,
    # "false_seabed": get_seabed_mask_multichannel,
    # "seabed": get_seabed_mask_multichannel,
}


def create_masks(source_Sv: xarray.Dataset, profiling_info: Dict, config: DenoiseConfig):
    masks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_mask_type = {}
        future_to_start_time = {}

        for mask_type, mask_config in config.items():
            if is_scalar(mask_config):
                continue

            if mask_config["enabled"] and (mask_type in MASK_CREATION_FUNCTIONS):
                profiling_info[mask_type + " mask"] = {}
                # Map the mask type to the appropriate creation function
                create_mask_func = MASK_CREATION_FUNCTIONS.get(mask_type)

                if config["profile"]:
                    future_to_start_time[mask_type] = time.time()

                future = executor.submit(
                    create_and_add_metadata, create_mask_func, source_Sv, mask_type, mask_config
                )
                future_to_mask_type[future] = mask_type

        for future in concurrent.futures.as_completed(future_to_mask_type):
            mask_type = future_to_mask_type[future]

            try:
                mask = future.result()
                masks.append((mask_type, mask))
                if config["profile"]:
                    end_time = time.time()
                    start_time = future_to_start_time[mask_type]
                    profiling_info[mask_type + " mask"]["execution_time"] = end_time - start_time
            except Exception as exc:
                print(f"{future_to_mask_type[future]} mask generated an exception: {exc}")

    return tuple(masks), profiling_info


def create_and_add_metadata(create_mask_func, source_Sv, mask_type, config_item):
    mask = create_mask_func(
        source_Sv, parameters=config_item["parameters"], method=config_item["method"]
    )

    return add_metadata_to_mask(
        mask=mask,
        metadata={
            "mask_type": mask_type,
            "method": config_item["method"],
            "parameters": dict_to_formatted_list(config_item["parameters"]),
        },
    )


def is_scalar(value):
    if isinstance(value, Iterable) and not isinstance(value, str):
        return False

    return True
