"""
shoal_detection_handler.py
--------------------------
This module provides functionalities for shoal detection in echosounder data.

Functions:

- `create_shoal_mask_multichannel(Sv, parameters, method)`:
    Generates two multichannel masks (`mask` and `mask_`) used for shoal detection based on the given Sv dataset.
- `combine_shoal_masks_multichannel(mask, mask_)`:
    Combines the multichannel masks (`mask` and `mask_`) to produce a final mask.
    The resulting mask contains `True` values only where both input masks are `True` for each channel.
 - `attach_shoal_mask_to_ds(ds, parameters, method)`:
    Attaches a shoal mask to the given dataset.

Usage:

To create the two necessary multichannel shoal masks for a given Sv dataset, ds:
    >>> create_shoal_mask_multichannel(Sv, parameters, method)
To combine two multichannel masks, `mask` and `mask_`:
    >>> combine_shoal_masks_multichannel(mask, mask_)
To attach a shoal mask to the given dataset, ds.:
    >>> attach_shoal_mask_to_ds(ds, parameters, method)

Notes:

- Ensure the echopype library is properly installed and imported when using this module.
- Sv represents the volume backscattering strength.
"""


from typing import Tuple

import xarray as xr
from echopype.mask.api import get_shoal_mask_multichannel
from oceanstream.utils import add_metadata_to_mask, attach_mask_to_dataset

WEILL_DEFAULT_PARAMETERS = {
    "thr": -70,
    "maxvgap": 5,
    "maxhgap": 5,
    "minvlen": 5,
    "minhlen": 5,
    "dask_chunking": {"ping_time": 1000, "range_sample": 1000},
}


def create_shoal_mask_multichannel(
    Sv: xr.Dataset,
    parameters: dict = WEILL_DEFAULT_PARAMETERS,
    method: str = "will",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Invokes echopype's `get_shoal_mask_multichannel` to create two multichannel masks used for shoal detection.

    Parameters:

    Sv : xr.Dataset
        The dataset for which the shoal masks will be created. This dataset should have the
        coordinate `channel` and variables `frequency_nominal` and `Sv`.
    parameters: dict
        Method parameters
    method: str
        Specifying the algorithm to use

    Returns:

    Tuple[xr.DataArray, xr.DataArray]
        A tuple containing two DataArrays:
        1. A multichannel mask (`mask`) for the Sv data. Regions that meet the thresholding criteria
           for shoal identification are marked as True.
        2. A mask (`mask_`) indicating the valid samples for the `mask`. Edge regions are filled
           with 'False', whereas the portion in which shoals could be detected is 'True'.

    Example:

    >>> mask = create_shoal_mask_multichannel(Sv, parameters, method)
    """
    mask = get_shoal_mask_multichannel(Sv, parameters, method)
    mask.attrs["shoal detection mask type"] = method
    return mask


def attach_shoal_mask_to_ds(
    ds: xr.Dataset, parameters: dict = WEILL_DEFAULT_PARAMETERS, method: str = "will"
) -> xr.Dataset:
    """
    Attaches a shoal mask to the given dataset.

    This function first creates two multichannel masks used for shoal detection
    using the `create_shoal_mask_multichannel` function.
    It then combines the masks using the `combine_shoal_masks_multichannel` function.
    Metadata indicating the mask type is added to the combined mask before attaching it to the dataset.

    Parameters:
    - ds (xr.Dataset):
        The dataset to which the shoal mask will be attached.
    - parameters (dist) and method (str):
        Arguments passed to the `create_shoal_mask_multichannel` function.

    Returns:
    - xr.Dataset: The dataset with the shoal mask attached.

    Example:
        >>> ds_with_shoal_mask = attach_shoal_mask_to_ds(ds, parameters, method)
    """
    shoal_mask = create_shoal_mask_multichannel(ds, parameters, method)
    shoal_mask = add_metadata_to_mask(mask=shoal_mask, metadata={"mask_type": "shoal"})
    return attach_mask_to_dataset(ds, shoal_mask)
