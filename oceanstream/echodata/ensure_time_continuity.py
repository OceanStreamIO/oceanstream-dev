"""
ensure_time_continuity.py
-----------------------
Module for ensuring temporal consistency of EK60/EK80-generated netcdf files

This module provides functionalities to:
- Check if a given netcdf file contains time reversals on a specific \
beam group and time dimension
- Fix time reversals in a given netcdf file on a specific list of beam groups \
and time dimensions associated.
"""

import xarray as xr
from echopype.qc.api import coerce_increasing_time, exist_reversed_time

DEFAULT_TIME_DICT = {"Sonar/Beam_group1": "ping_time"}
DEFAULT_DIMENSION = list(DEFAULT_TIME_DICT.keys())[0]
DEFAULT_TIME_NAME = DEFAULT_TIME_DICT.get(DEFAULT_DIMENSION)


def check_reversed_time(
    ed: xr.Dataset,
    dimension: str = DEFAULT_DIMENSION,
    time_name: str = DEFAULT_TIME_NAME,
):
    """
    Verifies if there are any issues with reversed timestamps in an unprocessed EK60/80 data file

    Parameters:
    - ds (xr.Dataset): Unprocessed netcdf file to check for time reversals
    - dimension (str): name of the sonar beam group, etc to examine
    - time_name (str): name of the time coordinate to test

    Returns:
    - has_reversal (bool): True if there are reversals, False otherwise

    Example:
    >>> check_reversed_time(ed, "Sonar/Beam_group1", "ping_time"})
    Expected Output
    False
    """
    has_reversal = exist_reversed_time(ds=ed[dimension], time_name=time_name)
    return has_reversal




def fix_time_reversions(er: xr.Dataset, time_dict=None, win_len: int = 100):
    """
    Fixes reversed timestamps in an unprocessed EK60/80 data file

    Parameters:
    - ds (xr.Dataset): Unprocessed netcdf file to check for time reversals
    - time_dict (dict): a dictionary where the keys are various sonar beam
                        groups and the values are the time dimension
    - win_len (int): length of the local window before the reversed timestamp
                        within which the median pinging interval
                        is used to infer the next ping time

    Returns:
    - ds (xr.Dataset): the input dataset but with specific time coordinates
                        coerced to flow forward

    Example:
    >>> fix_time_reversions(er, {"Sonar/Beam_group1": "ping_time"})
    Expected Output
    a fixed er dataset
    """
    if time_dict is None:
        time_dict = DEFAULT_TIME_DICT
    for dim, time in time_dict.items():
        if check_reversed_time(er, dim, time):
            coerce_increasing_time(er[dim], time, win_len)
    return er

