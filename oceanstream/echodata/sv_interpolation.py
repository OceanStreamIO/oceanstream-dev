"""
sv_interpolation.py
-------------------
Description: Module dedicated to the interpolation of acoustic backscatter data in echosounder datasets.
It provides utilities for converting between decibel and linear scales, applying various masks to the data,
and performing interpolation to fill missing or NaN values. The module supports both direct `xr.Dataset`
inputs and file path inputs, ensuring flexibility in data processing workflows. Additionally,
it offers edge filling capabilities to enhance the quality of interpolated echograms.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Hashable, Tuple, Union


def db_to_linear(db: xr.DataArray) -> xr.DataArray:
    """Convert decibel to linear scale, handling NaN values."""
    linear = xr.where(db.isnull(), np.nan, 10 ** (db / 10))
    return linear


def linear_to_db(linear: xr.DataArray) -> xr.DataArray:
    """Convert linear to decibel scale, handling NaN values."""
    db = xr.where(linear.isnull(), np.nan, 10 * np.log10(linear))
    return db


def interpolate_sv(
    ds_Sv: xr.Dataset, method: str = "linear", with_edge_fill: bool = False
) -> xr.Dataset:
    """
    Apply masks to the Sv DataArray in the dataset and interpolate over the resulting NaN values.

    Parameters:
    - Sv (Union[xr.Dataset, str, Path]): Dataset or path to the netCDF or zarr file.
    - method (str): Interpolation method.
    - with_edge_fill (bool): Flag to allow filling the edges of echograms

    Returns:
    - xr.Dataset: Dataset with the masked and interpolated Sv DataArray.

    Example:
    >> interpolate_sv(Sv, method)
    Expected Output
    """
    # Initialize an empty list to store the processed channels
    processed_channels = []

    # Loop over each channel
    for channel in ds_Sv.Sv["channel"]:
        channel_data = ds_Sv.Sv.sel(channel=channel)

        # Convert from dB to linear scale
        channel_data_linear = db_to_linear(channel_data)

        # Perform interpolation to fill NaN values in linear scale using Xarray's interpolate_na
        interpolated_channel_data_linear = channel_data_linear.interpolate_na(
            dim="ping_time",
            method=method,
            use_coordinate=True,
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

        if with_edge_fill:
            interpolated_channel_data_linear = interpolated_channel_data_linear.ffill(dim="ping_time")
            interpolated_channel_data_linear = interpolated_channel_data_linear.bfill(dim="ping_time")

        # Convert back to dB scale
        interpolated_channel_data = linear_to_db(interpolated_channel_data_linear)

        # Append the processed channel data to the list
        processed_channels.append(interpolated_channel_data)

        # Combine the processed channels back into a single DataArray
    interpolated_sv = xr.concat(processed_channels, dim="channel")

    # Update the Sv DataArray in the dataset with the interpolated values
    ds_Sv["Sv"] = interpolated_sv

    return ds_Sv


def find_impacted_variables(
    dataset: xr.Dataset, target_dim: str = "range_sample"
) -> list[Hashable]:
    """
    Finds and lists the variable names in a given xarray Dataset that are associated with a specified dimension.

    Parameters:
    - dataset (xr.Dataset): The xarray Dataset to search for variables.
    - target_dim (str, optional): The name of the target dimension to check in each variable. Defaults to "range_sample".

    Returns:
    - list[Hashable]: A list of variable names (hashable items) that have the target dimension.

    Example:
    >> impacted_vars = find_impacted_variables(dataset)
    Expected Output:
    A list of variable names from the dataset which have 'range_sample' as one of their dimensions.

    >> impacted_vars = find_impacted_variables(dataset, target_dim='time')
    Expected Output:
    A list of variable names from the dataset which have 'time' as one of their dimensions.
    """
    variable_list = []
    for var_name, data_array in dataset.variables.items():
        if target_dim in data_array.dims and var_name != target_dim:
            variable_list.append(var_name)
    return variable_list


def find_lowest_resolution_channel(dataset: xr.Dataset) -> Tuple[int, int]:
    """
    Finds the channel with the lowest resolution in a given xarray Dataset based on echo range data.

    This function iterates over all channels in the dataset, identified by 'frequency_nominal',
    and determines the channel that reaches the smallest maximum depth across all pings.
    The depth is determined based on 'echo_range' values in the dataset.

    Parameters:
    - dataset (xr.Dataset): The xarray Dataset containing echo data and channel information.

    Returns:
    - Tuple[int, int]: A tuple containing the index of the channel with the lowest resolution
      and the maximum depth index of that channel. The first element is the channel index,
      and the second element is the maximum depth index within that channel.

    Example:
    >> lowest_res_channel, max_depth_index = find_lowest_resolution_channel(dataset)
    Expected Output:
    A tuple where the first element is the index of the channel with the lowest resolution,
    and the second element is the maximum depth index within that channel.
    """
    arg_max_dataset = len(dataset["range_sample"].values)
    return_channel = 0
    for ch in range(len(dataset["frequency_nominal"])):
        arg_max = np.max(np.nanargmax(dataset["echo_range"].values[ch, :, :], axis=1))
        if arg_max < arg_max_dataset:
            arg_max_dataset = arg_max
            return_channel = ch
    return return_channel, arg_max_dataset


def resample_xarray(da: xr.DataArray, old_depth_da, new_depth_da, new_range_sample) -> xr.DataArray:
    """
    Resamples an xarray DataArray to a new depth profile using linear interpolation.

    This function processes an xarray DataArray representing acoustic data, resampling it to
    match a new depth profile. The function handles both 'Sv' (volume backscattering strength)
    and other data types. For 'Sv' data, it first converts from decibel to linear scale before
    resampling, and then back to decibel scale after resampling.

    Parameters:
    - da (xr.DataArray): The xarray DataArray to resample. It should contain 'channel' and 'ping_time' dimensions.
    - old_depth_da (array-like): The original depth values corresponding to each point in 'da'.
    - new_depth_da (array-like): The new depth values to which 'da' will be resampled.
    - new_range_sample (array-like): The new range sample values that will form the new 'range_sample' coordinate.

    Returns:
    - xr.DataArray: The resampled xarray DataArray with the new depth profile.

    Example:
    >> resampled_da = resample_xarray(da, old_depth_da, new_depth_da, new_range_sample)
    Expected Output:
    An xarray DataArray resampled to the new depth profile specified by 'new_depth_da' and 'new_range_sample'.

    Note:
    - The function assumes 'da' has dimensions 'channel' and 'ping_time'.
    - For 'Sv' data, conversions between decibel and linear scales are performed.
    """
    from scipy.interpolate import interp1d

    interpolated_data_arrays = []
    channel_order_list = []
    time_coord = da["ping_time"]
    depth_values = old_depth_da
    for ch in range(da.sizes["channel"]):
        depth_idx = len(depth_values[ch])
        if da.name == "Sv":
            data_array = db_to_linear(da[ch, :, :depth_idx]).values
        else:
            data_array = da[ch, :, :depth_idx].values
        new_data_array = np.zeros((data_array.shape[0], len(new_range_sample)))
        for i in range(len(data_array)):  # Loop over the time dimension
            # Assign new depth values as coordinates
            # Create an interpolation function for the current slice
            interp_func = interp1d(
                depth_values[ch],
                data_array[i],
                kind="nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            # Interpolate the data to the new depth values
            new_data_array[i, :] = interp_func(new_depth_da[:])
        # Create a DataArray for the interpolated data of this channel
        channel_data = xr.DataArray(
            new_data_array,
            coords={"ping_time": time_coord, "range_sample": new_range_sample},
            dims=["ping_time", "range_sample"],
        )
        channel_order_list.append(da["channel"][ch].values)
        interpolated_data_arrays.append(channel_data)
    # Combine interpolated data from all channels
    combined_data = xr.concat(interpolated_data_arrays, dim="channel")
    combined_data = combined_data.assign_coords(channel=("channel", channel_order_list))
    if da.name == "Sv":
        combined_data = linear_to_db(combined_data)
    return combined_data


def regrid_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Regrids an xarray Dataset to a new depth profile based on the channel with the lowest resolution.

    This function identifies the channel with the lowest resolution in the provided dataset and
    uses its depth profile to regrid all relevant variables in the dataset to a new common depth
    profile. It resamples variables that include the 'range_sample' dimension and retains other
    coordinates and attributes unchanged.

    Parameters:
    - dataset (xr.Dataset): The xarray Dataset to be regridded. It should contain 'echo_range'
      and 'range_sample' among other coordinates and dimensions.

    Returns:
    - xr.Dataset: A new xarray Dataset with variables resampled to the new common depth profile.

    Example:
    >> regridded_dataset = regrid_dataset(dataset)
    Expected Output:
    A new xarray Dataset with variables that have been resampled to match the depth profile of
    the channel with the lowest resolution in the original dataset.

    Note:
    - The function determines the new common depth profile based on the channel with the lowest resolution.
    - Variables with the 'range_sample' dimension are resampled; other variables and coordinates
      are copied as is.
    - The attributes of the original dataset are preserved in the new dataset.
    """
    new_dataset = xr.Dataset()
    channel, max_depth_idx = find_lowest_resolution_channel(dataset)

    per_ping_depth = np.nanargmax(dataset["echo_range"].values[channel, :, :], axis=1)
    new_range_sample = np.arange(np.max(per_ping_depth) + 1)
    ping_with_max_depth = np.nanargmax(per_ping_depth)
    bin_depths = dataset["echo_range"].values[:, ping_with_max_depth, :]
    new_bin_depths = dataset["echo_range"].values[channel, ping_with_max_depth, : max_depth_idx + 1]
    for coord_name in dataset.coords:
        if coord_name == "range_sample":
            new_dataset.coords[coord_name] = ("range_sample", new_range_sample)
        else:
            new_dataset.coords[coord_name] = dataset.coords[coord_name]
    impacted_vars = find_impacted_variables(dataset, target_dim="range_sample")
    for var_name, data_array in dataset.data_vars.items():
        if var_name in impacted_vars:
            data_array = resample_xarray(data_array, bin_depths, new_bin_depths, new_range_sample)
        new_dataset[var_name] = data_array
    new_dataset.attrs = dataset.attrs.copy()
    return new_dataset
