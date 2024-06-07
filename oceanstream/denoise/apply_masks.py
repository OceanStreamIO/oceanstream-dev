import pathlib
import xarray as xr

from typing import Dict, Union
from oceanstream.echodata import interpolate_sv


from .applying_masks_handler import (
    apply_selected_noise_masks_and_or_noise_removal as apply_selected_masks,
)


def apply_noise_masks(sv_with_masks: Union[xr.Dataset, str, pathlib.Path], config) -> xr.Dataset:
    noise_masks = {}

    if config["impulse"]["enabled"]:
        noise_masks["mask_impulse"] = {"var_name": "Sv"}
    if config["attenuation"]["enabled"]:
        noise_masks["mask_attenuation"] = {"var_name": "Sv"}
    if config["transient"]["enabled"]:
        noise_masks["mask_transient"] = {"var_name": "Sv"}

    ds_processed = apply_selected_masks(sv_with_masks, noise_masks)

    return ds_processed


def apply_background_noise_removal(
    ds_processed: xr.Dataset, config
) -> xr.Dataset:
    ds_interpolated = interpolate_sv(ds_processed)
    params = config["remove_background_noise"]["parameters"]

    ds_clean = apply_selected_masks(
        ds_interpolated, processes_to_apply={"remove_background_noise": params}
    )

    return ds_clean


def apply_seabed_mask(ds: xr.Dataset, config) -> xr.Dataset:
    if config["seabed"]["enabled"] and config["false_seabed"]["enabled"]:
        params = {"mask_seabed": {"var_name": "Sv"}, "mask_false_seabed": {"var_name": "Sv"}}

        ds_processed = apply_selected_masks(ds, params)

        return ds_processed

    return ds
