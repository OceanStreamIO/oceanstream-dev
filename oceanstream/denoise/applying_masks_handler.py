"""
    applying_masks_handler.py
    -------------------------
    Description: This module provides functionalities for applying a sequence of
    selected masks and background noise removal to echosounder datasets based on
    specific methodologies.

    Masks/Processes supported in this module include:
    **Noise-related masks/processes, namely:**
    - 'mask_false_seabed'
    - 'mask_seabed'
    - 'mask_impulse'
    - 'mask_attenuation'
    - 'mask_transient'
    - 'remove_background_noise'

    **Organism-related masks, namely:**
    - 'mask_krill'
    - 'mask_gas_bearing_organisms'
    - 'mask_fluid_like_organisms'
    - 'mask_shoal'

    Note: If other masks that aren't supported by this module's functions are
    attached as variables to the dataset, they can be applied using
    `echopype.mask.apply_mask`. For details, consult the echopype documentation.

    Key Functions:
    - `apply_selected_noise_masks_and_or_noise_removal`: For applying selected
    noise masks and/or noise removal.

    - `apply_mask_organisms_in_order`: For applying organism-related masks in
    a user-defined order.
"""


import pathlib
from typing import Union

import echopype as ep
import xarray as xr

from .background_noise_remover import apply_remove_background_noise


def apply_selected_noise_masks_and_or_noise_removal(
    ds: Union[xr.Dataset, str, pathlib.Path],
    processes_to_apply: dict,
) -> xr.Dataset:
    """
    Apply a sequence of selected masks and background noise removal (if provided in the processes_to_apply)
    to the given dataset based on the specifications in the article:
    > Haris, K., et al. "Sounding out life in the deep using acoustic data from ships of opportunity."
    > Scientific Data 8.1 (2021): 23.

    Order of cleaning process:
    1. Impulse noise mask (if provided)
    2. Attenuated signal mask (if provided)
    3. Transient noise mask (if provided)
    4. Background noise mask (if provided)
    5. False seabed mask (if provided)
    6. Seabed mask (if provided)

    Valid processes/masks:
    - 'mask_impulse'
    - 'mask_attenuation'
    - 'mask_transient'
    - 'remove_background_noise'
    - 'mask_false_seabed'
    - 'mask_seabed'

    Parameters:
    - `ds`: (`xr.Dataset`, `str`, or `pathlib.Path`)
      Points to a Dataset that contains the variable the mask should be applied to
      and must have the masks as attached variables.
    - `processes_to_apply` (`dict`):
      A dictionary where keys are mask names or 'remove_background_noise' and values are specific parameters
      (sub-dictionaries).

    For more details on parameters associated with applying a mask and the `background noise removal` process,
    those parameters found within the specific sub-dictionaries, please refer to the documentation for:
    - `echopype.mask.apply_mask`
    - `background_noise_remover.apply_remove_background_noise`

    Returns:
    - `xr.Dataset`: Processed dataset.

    """

    valid_processes = [
        "mask_impulse",
        "mask_attenuation",
        "mask_transient",
        "remove_background_noise",
        "mask_false_seabed",
        "mask_seabed",
    ]

    # Check for unexpected masks/processes
    for process in processes_to_apply:
        if process not in valid_processes:
            raise ValueError(
                "Unexpected mask/process. Please refer to the function documentation for valid masks/processes."
            )

    for process in valid_processes:
        if process in processes_to_apply:
            params = processes_to_apply[process]
            if process in [
                "mask_impulse",
                "mask_attenuation",
                "mask_transient",
                "mask_false_seabed",
                "mask_seabed",
            ]:
                mask_data = ds[process]
                ds = ep.mask.apply_mask(ds, mask_data, **params)
            elif process == "remove_background_noise":
                ds = apply_remove_background_noise(ds, **params)

    return ds


def apply_mask_organisms_in_order(ds, processes_to_apply):
    """
    Apply a sequence of selected masks to the given dataset in the order that
    the masks were provided in the input dictionary.

    Valid masks:
        'mask_krill',
        'mask_gas_bearing_organisms',
        'mask_fluid_like_organisms',
        'mask_shoal'

    Parameters:
    - ds (xr.Dataset, str, or pathlib.Path):
        Points to a Dataset that contains the variable the mask should be applied to
        and must have the masks as attached variables.
    - processes_to_apply (dict):
        A dictionary where keys are mask names and values are specific parameters (sub-dictionaries).

    For more details on parameters associated with applying a mask,
    those parameters found within the specific sub-dictionaries, please refer to the documentation for:
    - `echopype.mask.apply_mask`

    Returns:
    - xr.Dataset: Processed dataset.
    """

    recognized_masks = [
        "mask_krill",
        "mask_gas_bearing_organisms",
        "mask_fluid_like_organisms",
        "mask_shoal",
    ]

    for mask_name in processes_to_apply.keys():
        if mask_name not in recognized_masks:
            raise ValueError(
                "Unrecognized mask name. Please refer to the function documentation for valid masks"
            )

        mask_params = processes_to_apply[mask_name]

        if mask_name == "mask_krill":
            ds = ep.mask.apply_mask(ds, ds[mask_name], **mask_params)
        elif mask_name == "mask_gas_bearing_organisms":
            ds = ep.mask.apply_mask(ds, ds[mask_name], **mask_params)
        elif mask_name == "mask_fluid_like_organisms":
            ds = ep.mask.apply_mask(ds, ds[mask_name], **mask_params)
        elif mask_name == "mask_shoal":
            ds = ep.mask.apply_mask(ds, ds[mask_name], **mask_params)

    return ds
