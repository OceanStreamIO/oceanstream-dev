"""
sv_dataset_extension.py
-----------------------

Module for enriching the volume backscattering strength (Sv) dataset by adding depth, location,
and split-beam angle information.

Functions:
- `enrich_sv_dataset`: Main function to enhance the Sv dataset by adding depth, location,
and split-beam angle information.

Usage:

To enrich an Sv dataset `sv` for a given EchoData object, `ed`, simply call:
`enriched_sv = enrich_sv_dataset(sv, ed, **kwargs)`

Note:
The specific keyword arguments (`**kwargs`) that can be passed to the function are dependent on
the `add_depth()`, `add_location()`, and `add_splitbeam_angle()` functions from the echopype module.
Refer to their respective documentation for details.

"""
import warnings
import numpy as np
import xarray as xr

from echopype.consolidate import add_location, add_splitbeam_angle
from echopype.echodata import EchoData


def enrich_sv_dataset(sv: xr.Dataset, echodata: EchoData, **kwargs) -> xr.Dataset:
    """
    Enhances the input `sv` dataset by adding depth, location, and split-beam angle information.

    Parameters:
    - sv (xr.Dataset): Volume backscattering strength (Sv) from the given echodata.
    - echodata (EchoData): An EchoData object holding the raw data.
    - **kwargs: Keyword arguments specific to `add_depth()`, `add_location()`, and `add_splitbeam_angle()`.
        Note: These functions are implemented in the echopype module.

    Returns:
    xr.Dataset:
        An enhanced dataset with depth, location, and split-beam angle.
    """

    depth_keys = ["depth_offset", "tilt", "downward"]
    depth_args = {k: kwargs[k] for k in depth_keys if k in kwargs}

    location_keys = ["nmea_sentence"]
    location_args = {k: kwargs[k] for k in location_keys if k in kwargs}


    splitbeam_keys = [
        "waveform_mode",
        "encode_mode",
        "pulse_compression",
        "storage_options",
        "return_dataset",
    ]
    splitbeam_args = {k: kwargs[k] for k in splitbeam_keys if k in kwargs}

    try:
        add_depth(sv, **depth_args)
    except Exception as e:
        warnings.warn(f"Failed to add depth due to error: {str(e)}")

    try:
        sv = add_location(sv, echodata, **location_args)
    except Exception as e:
        warnings.warn(f"Failed to add location due to error: {str(e)}")

    try:
        add_splitbeam_angle(sv, echodata, **splitbeam_args)
    except Exception as e:
        warnings.warn(f"Failed to add split-beam angle due to error: {str(e)}")

    return sv


def add_depth(Sv: xr.Dataset, depth_offset: float = 0, tilt: float = 0, downward: bool = True):
    """
    Given an existing Sv dataset, it adds a data variable called depth containing the depth of
    each ping.

    Parameters:
    - Sv (xr.Dataset): Volume backscattering strength (Sv) from the given echodata.
    - depth_offset (float): Depth offset to be added to the depth values.
    - tilt (float): Tilt angle of the transducer in degrees.
    - downward (bool): Flag indicating whether the depth is measured downward (True) or upward (False).
    """

    mult = 1 if downward else -1

    first_channel = Sv["channel"].values[0]
    first_ping_time = Sv["ping_time"].values[0]

    # Slice the echo_range to get the desired range of values
    selected_echo_range = Sv["echo_range"].sel(channel=first_channel, ping_time=first_ping_time)
    selected_echo_range = selected_echo_range.values.tolist()
    selected_echo_range = [mult * value * np.cos(tilt / 180 * np.pi) + depth_offset for value in selected_echo_range]
    Sv = Sv.assign_coords(range_sample=selected_echo_range)
    min_val = np.nanmin(selected_echo_range)
    max_val = np.nanmax(selected_echo_range)
    Sv = Sv.sel(range_sample=slice(min_val, max_val))

    return Sv


def add_seabed_depth(Sv: xr.Dataset):
    """
    Given an existing Sv dataset with a seabed mask attached, it adds a
    data variable called seabed depth containing the location of the seabed on
    each ping

    Parameters:
    - sv (xr.Dataset): Volume backscattering strength (Sv) from the given echodata.

    Returns:
    xr.Dataset:
        An enhanced dataset with seabed depth
    """
    seabed_mask = ~Sv["mask_seabed"]
    seabed_level = seabed_mask.argmax(dim="range_sample")
    res = Sv.assign(seabed_level=seabed_level)
    return res
