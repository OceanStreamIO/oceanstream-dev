"""
background_noise_remover.py
----------------------------

This module provides functionalities to enhance the quality of echosounder data by removing background noise.

Functions:

- `apply_remove_background_noise`:
Main function that applies the `remove_noise` function from echopype to a Sv dataset.
Computes default parameters based on dataset characteristics when not provided.
The dataset initially passed to the function, processed in the following manner:
`Sv_corrected` and `Sv_noise` variables are generated by the echopype `remove_noise` function.
`Sv` contains the values originally in `Sv_corrected`.
`Sv_with_background_noise` stores the original values of `Sv`.

Usage:

To remove background noise for a given Sv dataset, `ds_Sv`, simply call:
`apply_remove_background_noise(ds_Sv)`

Note:
- Ensure that the echopype library is properly installed and imported when using this module.
- Sv = the volume backscattering strength

"""
import echopype as ep
import numpy as np
import xarray as xr


def apply_remove_background_noise(
    ds_Sv: xr.Dataset,
    ping_num: int = 40,
    range_sample_num: int = None,
    noise_max: float = -125,
    SNR_threshold: float = 3,
) -> xr.Dataset:
    """
    Applies the `remove_noise` function from the echopype library to the `ds_Sv` dataset.
    If `range_sample_num` is not provided, the function computes it based on a 10-meter vertical bin
    for each channel, and then uses the minimum value across all channels as the default.

    For detailed information on `remove_noise`, consult the documentation
    within the `clean` module of the echopype library.

    For default values and their context, refer to:
    De Robertis & Higginbottom (2007). "A post-processing technique to estimate
    the signal-to-noise ratio and remove echosounder background noise."
    ICES Journal of Marine Sciences, 64(6), 1282–1291.

    Parameters:
    - ds_Sv: xr.Dataset
        Dataset containing Sv and echo_range [m].
    - ping_num: int, default=40
        Number of pings to obtain noise estimates.
    - range_sample_num: int, optional
        Number of samples along the range_sample dimension to obtain noise estimates. If not provided,
        it's computed based on a 10-meter vertical bin for each channel and the minimum value is used.
    - noise_max: float, default=-125
        The upper limit for background noise expected under the operating conditions (default=-125 dB).
    - SNR_threshold: float, default=3 [dB]
        Acceptable signal-to-noise ratio.

    Returns:
    - ds_Sv: xr.Dataset

    Notes:
    Returns - initial xr.Dataset:
    The dataset initially passed to the function, processed in the following manner:
    - `Sv_corrected` and `Sv_noise` variables are generated by the echopype `remove_noise` function.
    - `Sv` contains the values originally in `Sv_corrected`.
    - `Sv_with_background_noise` stores the original values of `Sv`.
    """

    # Compute range_sample_num if not provided
    if range_sample_num is None:
        range_sample_nums = []
        for ch in range(ds_Sv.sizes["channel"]):
            mean_diff = np.nanmean(np.diff(ds_Sv["echo_range"].isel(channel=ch).values[0, :]))
            if np.isnan(mean_diff):
                raise ValueError(
                    "The default computed value for the range_sample_num is nan, please add the range_sample_num as input parameter"
                )
            range_sample_nums.append(int(10 / mean_diff))

        range_sample_num = min(range_sample_nums)
    # Remove noise
    ds_Sv_processed = ep.clean.remove_background_noise(
        ds_Sv,
        ping_num=ping_num,
        range_sample_num=range_sample_num,
        noise_max=noise_max,
        SNR_threshold=SNR_threshold,
    )

    return ds_Sv_processed
