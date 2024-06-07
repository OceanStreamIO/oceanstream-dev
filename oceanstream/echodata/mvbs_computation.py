import xarray as xr

from typing import Literal, Optional
from echopype.commongrid.api import compute_MVBS, compute_MVBS_index_binning


def compute_mvbs(
    ds_Sv: xr.Dataset,
    method: Literal["index_binning", "physical_units"] = "physical_units",
    range_sample_num: Optional[int] = None,
    ping_num: Optional[int] = None,
    range_var: Optional[Literal["echo_range", "depth"]] = None,
    range_bin: Optional[str] = None,
    ping_time_bin: Optional[str] = None,
) -> xr.Dataset:
    """
    Wrapper function around echopype's `echopype.commongrid.api.compute_MVBS`
    and `echopype.commongrid.api.compute_MVBS_index_binning`.
    - `echopype.commongrid.api.compute_MVBS`:
        Compute Mean Volume Backscattering Strength (MVBS)
        based on intervals of range (`echo_range`) or depth (`depth`)
        and `ping_time` specified in physical units.
    - `echopype.commongrid.api.compute_MVBS_index_binning`:
        Compute Mean Volume Backscattering Strength (MVBS)
        based on intervals of range sample number(`range_sample_num`)
        and ping number (`ping_num`) specified in index number.

    Parameters:

    - ds_Sv : xr.Dataset
        Input dataset containing Sv
    - method : str
        Choose method type. Either "index_binning" or "physical_units"
    - range_sample_num, ping_num
        Parameters for "index_binning" method. Defaults are 100.
    - range_var: {'echo_range', 'depth'}, default 'echo_range'
        Parameter for "physical_units" method.
        The variable to use for range binning.
        Must be one of `echo_range` or `depth`.
        Note that `depth` is only available if the input dataset contains
        `depth` as a data variable.
    - range_bin, ping_time_bin
        Parameters for "physical_units" method. Defaults are "20m", and "20S" respectively.

    Returns:

    - A dataset containing bin-averaged Sv.
    """

    methods = {
        "index_binning": compute_MVBS_index_binning,
        "physical_units": compute_MVBS,
    }

    if method == "index_binning":
        if any([range_var, range_bin, ping_time_bin]):
            raise ValueError(
                "For 'index_binning' method, only range_sample_num and ping_num should be provided."
            )
        range_sample_num = range_sample_num if range_sample_num is not None else 100
        ping_num = ping_num if ping_num is not None else 100
        return methods[method](ds_Sv, range_sample_num, ping_num)

    elif method == "physical_units":
        if any([range_sample_num, ping_num]):
            raise ValueError(
                "For 'physical_units' method, only range_var, range_bin, and ping_time_bin should be provided."
            )
        range_var = range_var if range_var is not None else "echo_range"
        range_bin = range_bin if range_bin is not None else "20m"
        ping_time_bin = ping_time_bin if ping_time_bin is not None else "20S"
        return methods[method](
            ds_Sv, range_var=range_var, range_bin=range_bin, ping_time_bin=ping_time_bin
        )

    else:
        raise ValueError("Invalid method. Choose either 'index_binning' or 'physical_units'.")
