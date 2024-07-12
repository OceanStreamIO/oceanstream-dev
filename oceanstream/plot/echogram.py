import asyncio
import os
import tempfile
import xarray as xr
import numpy as np
import logging

from matplotlib.colors import LinearSegmentedColormap, Colormap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union


def plot_individual_channel(ds_Sv, channel, output_path, file_base_name,
                            cmap='viridis',
                            regions2d=None,
                            region_ids=None, region_class=None):
    """Plot and save echogram for a single channel with optional regions and enhancements."""
    import pandas as pd
    plot_params = {
        "vmin": -70,
        "vmax": -30,
        "cmap": cmap
    }

    full_channel_name = ds_Sv.channel.values[channel]
    channel_name = "_".join(full_channel_name.split()[:3])

    plt.figure(figsize=(20, 10))
    colors = ['black', 'yellow', 'red', 'orange']

    idx = 0
    labels_added = set()

    echogram_output_path = os.path.join(output_path, f"{file_base_name}_{channel_name}.png")
    ax = plt.subplot(1, 1, 1)

    echogram_data = np.flipud(ds_Sv.Sv.isel(channel=channel).T.values)
    plt.pcolormesh(ds_Sv.ping_time, ds_Sv.range_sample, echogram_data, **plot_params)
    cbar = plt.colorbar(ax=ax, orientation="vertical")
    cbar.set_label("Volume backscattering strength (dB re 1m$^{-1}$)")

    if region_ids:
        idx = _plot_region_ids(colors, ds_Sv, idx, labels_added, region_ids, regions2d)

    if region_class:
        _plot_region_classes(channel_name, colors, ds_Sv, idx, labels_added, region_class, regions2d)

    if regions2d:
        plt.legend()

    ax.invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Format the x-axis to show time (HH:mm)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Add horizontal lines at specific depth intervals
    depth_intervals = np.arange(0, ds_Sv.range_sample.max(), 10)  # Example interval of 10 units
    for depth in depth_intervals:
        ax.axhline(depth, color='gray', linestyle='--', linewidth=0.5)

    # Convert the first ping_time to a Python datetime object
    date_str = pd.to_datetime(ds_Sv.ping_time.values[0]).strftime('%Y-%m-%d')

    # Add date label on the right side
    plt.xlabel('Time (HH:MM)', fontsize=14)
    plt.ylabel('Depth', fontsize=14)
    plt.title(f'Echogram for Channel {channel_name}', fontsize=16, fontweight='bold')
    plt.text(1.02, 0.5, date_str, transform=ax.transAxes, rotation=90, va='center', fontsize=12, fontweight='bold')

    # Add a secondary y-axis for fathoms (1 fathom = 1.8288 meters)
    secax = ax.secondary_yaxis('right', functions=(lambda x: x / 1.8288, lambda x: x * 1.8288))
    secax.set_ylabel('Depth (fathoms)', fontsize=14)

    plt.savefig(echogram_output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_channel_shaders(ds_Sv, channel, output_path, file_base_name, cmap='ocean_r'):
    """Plot and save echogram for a single channel with optional regions and enhancements."""
    import holoviews as hv
    from holoviews.operation.datashader import rasterize
    from datashader import reductions as ds
    hv.extension('bokeh', 'matplotlib')

    full_channel_name = ds_Sv.channel.values[channel]
    channel_name = "_".join(full_channel_name.split()[:3])

    echogram_output_path_html = os.path.join(output_path, f"{file_base_name}_{channel_name}.html")
    echogram_output_path_png = os.path.join(output_path, f"{file_base_name}_{channel_name}.png")

    filtered_ds = ds_Sv['Sv']
    if 'beam' in filtered_ds.dims:
        filtered_ds = filtered_ds.isel(beam=0).drop('beam')

    if 'channel' in filtered_ds.dims:
        filtered_ds = filtered_ds.assign_coords({'frequency': ds_Sv.frequency_nominal})
        filtered_ds = filtered_ds.swap_dims({'channel': 'frequency'})
        if filtered_ds.frequency.size == 1:
            filtered_ds = filtered_ds.isel(frequency=0)

    # Handle NaNs by dropping NaNs directly from ping_time and range_sample
    filtered_ds = filtered_ds.dropna(dim='ping_time', how='all')
    filtered_ds = filtered_ds.dropna(dim='range_sample', how='all')

    # Extract data for visualization
    sv_data = filtered_ds.isel(frequency=channel)
    ping_time = sv_data['ping_time']
    range_sample = sv_data['range_sample']

    sv_values = sv_data.values
    ds_array = xr.DataArray(sv_values, coords=[ping_time, range_sample], dims=['ping_time', 'range_sample'])
    hv_quadmesh = hv.QuadMesh(ds_array, kdims=['ping_time', 'range_sample'], vdims=['Sv'])
    rasterized_quadmesh = rasterize(hv_quadmesh, aggregator=ds.mean('Sv'))

    rasterized_quadmesh = rasterized_quadmesh.opts(
        cmap='viridis',
        colorbar=True,
        responsive=True,
        min_height=600,
        clim=(np.nanmin(sv_values), np.nanmax(sv_values)),
        tools=['hover', 'box_select'],
        active_tools=['wheel_zoom'],
        invert_yaxis=True,
        hooks=[lambda plot, element: plot.handles['colorbar'].title('Sv')]
    )

    # Save the plot as HTML
    hv.save(rasterized_quadmesh, echogram_output_path_html, fmt='html')

    # Save the plot as PNG
    # hv.save(rasterized_quadmesh, echogram_output_path_png, fmt='png')


def plot_individual_channel_enhanced(ds_Sv, channel, output_path, file_base_name,
                                     cmap=None,
                                     regions2d=None,
                                     region_ids=None, region_class=None,
                                     threshold=[-80, -50]):
    """Plot and save echogram for a single channel with optional regions and enhancements."""
    full_channel_name = ds_Sv.channel.values[channel]
    channel_name = "_".join(full_channel_name.split()[:3])

    num_pings = ds_Sv.dims['ping_time']
    num_samples = ds_Sv.dims['range_sample']
    aspect_ratio = num_pings / num_samples

    # Set base dimensions
    base_width = 10  # Base width in inches
    base_height = 10  # Base height in inches

    # Adjust width and height based on aspect ratio, ensuring practical limits
    if aspect_ratio > 1:
        fig_width = base_width * aspect_ratio
        fig_height = base_height
    else:
        fig_width = base_width
        fig_height = base_height / aspect_ratio

    fig_width = max(min(fig_width, 20), 10)  # Width between 10 and 20 inches
    fig_height = max(min(fig_height, 10), 5)  # Height between 5 and 10 inches

    plt.figure(figsize=(fig_width, fig_height))

    echogram_output_path = os.path.join(output_path, f"{file_base_name}_{channel_name}.png")

    # Apply the same preprocessing steps from _plot_echogram
    ds = ds_Sv
    filter_var = 'channel'
    filter_val = channel
    if 'backscatter_i' in ds.variables:
        filtered_ds = np.abs(ds.backscatter_r + 1j * ds.backscatter_i)
    else:
        filtered_ds = ds['Sv']
        if 'beam' in filtered_ds.dims:
            filtered_ds = filtered_ds.isel(beam=0).drop('beam')

    if 'channel' in filtered_ds.dims:
        filtered_ds = filtered_ds.assign_coords({'frequency': ds.frequency_nominal})
        filtered_ds = filtered_ds.swap_dims({'channel': 'frequency'})
        if filtered_ds.frequency.size == 1:
            filtered_ds = filtered_ds.isel(frequency=0)

    # Update axis labels
    filtered_ds['ping_time'].attrs = {
        'long_name': filtered_ds['ping_time'].attrs.get('long_name', 'Ping time'),
        'units': filtered_ds['ping_time'].attrs.get('units', '')
    }
    filtered_ds['range_sample'].attrs = {
        'long_name': filtered_ds['range_sample'].attrs.get('long_name', 'Depth'),
        'units': filtered_ds['range_sample'].attrs.get('units', '')
    }

    # Handle NaNs by dropping NaNs directly from ping_time and range_sample
    filtered_ds = filtered_ds.dropna(dim='ping_time', how='all')
    filtered_ds = filtered_ds.dropna(dim='range_sample', how='all')

    # Set the colormap
    if cmap is None:
        simrad_color_table = [
            (1, 1, 1), (0.6235, 0.6235, 0.6235), (0.3725, 0.3725, 0.3725),
            (0, 0, 1), (0, 0, 0.5), (0, 0.7490, 0), (0, 0.5, 0),
            (1, 1, 0), (1, 0.5, 0), (1, 0, 0.7490), (1, 0, 0),
            (0.6509, 0.3255, 0.2353), (0.4705, 0.2353, 0.1568)
        ]
        cmap = LinearSegmentedColormap.from_list('Simrad', simrad_color_table)
        cmap.set_bad(color='grey')

    # Custom tick formatter for datetime64 values as floats
    def format_datetime(x, pos=None):
        try:
            dt = x.astype('datetime64[ms]').astype('object')
            tick_label = dt.strftime("%H:%M:%S")
        except:
            tick_label = ''
        return tick_label

    # Create the echogram using xarray's plot method
    echogram_data = filtered_ds.isel(frequency=channel).T
    echogram_data.plot(
        x='ping_time',
        y='range_sample',
        yincrease=False,
        vmin=threshold[0],
        vmax=threshold[1],
        cmap=cmap,
        cbar_kwargs={'label': 'Volume backscattering strength (Sv re 1 m-1)'}
    )

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_datetime))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Ping time', fontsize=14)
    plt.ylabel('Depth', fontsize=14)
    plt.title(f'Echogram for Channel {channel_name}', fontsize=16, fontweight='bold')
    plt.savefig(echogram_output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_channel_image_only(ds_Sv, channel, output_path, file_base_name,
                                       cmap='viridis'):
    full_channel_name = ds_Sv.channel.values[channel]
    channel_name = "_".join(full_channel_name.split()[:3])

    plt.figure(figsize=(30, 18))
    echogram_output_path = os.path.join(output_path, f"{file_base_name}_{channel_name}.png")

    filtered_ds = ds_Sv['Sv']
    if 'beam' in filtered_ds.dims:
        filtered_ds = filtered_ds.isel(beam=0).drop('beam')

    if 'channel' in filtered_ds.dims:
        filtered_ds = filtered_ds.assign_coords({'frequency': ds_Sv.frequency_nominal})
        filtered_ds = filtered_ds.swap_dims({'channel': 'frequency'})
        if filtered_ds.frequency.size == 1:
            filtered_ds = filtered_ds.isel(frequency=0)

    # Update axis labels
    filtered_ds['ping_time'].attrs = {
        'long_name': filtered_ds['ping_time'].attrs.get('long_name', 'Ping time'),
        'units': filtered_ds['ping_time'].attrs.get('units', '')
    }
    filtered_ds['range_sample'].attrs = {
        'long_name': filtered_ds['range_sample'].attrs.get('long_name', 'Depth'),
        'units': filtered_ds['range_sample'].attrs.get('units', '')
    }

    # Handle NaNs by dropping NaNs directly from ping_time and range_sample
    filtered_ds = filtered_ds.dropna(dim='ping_time', how='all')
    filtered_ds = filtered_ds.dropna(dim='range_sample', how='all')

    # Create the echogram using xarray's plot method
    filtered_ds.isel(frequency=channel).T.plot(
        x='ping_time',
        y='range_sample',
        yincrease=False,
        vmin=-80,
        vmax=-50,
        cmap='ocean_r',
        cbar_kwargs={},
        add_colorbar=False
    )

    plt.title('')
    plt.axis('off')
    plt.savefig(echogram_output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_channel_simplified(ds_Sv, channel, output_path, file_base_name, echogram_path=None,
                                       config_data=None, cmap='ocean_r'):
    """Plot and save echogram for a single channel with optional regions and enhancements."""
    full_channel_name = ds_Sv.channel.values[channel]
    channel_name = "_".join(full_channel_name.split()[:3])
    ds = ds_Sv

    filtered_ds = ds['Sv']
    if 'beam' in filtered_ds.dims:
        filtered_ds = filtered_ds.isel(beam=0).drop('beam')

    if 'channel' in filtered_ds.dims:
        filtered_ds = filtered_ds.assign_coords({'frequency': ds.frequency_nominal})
        try:
            filtered_ds = filtered_ds.swap_dims({'channel': 'frequency'})
            if filtered_ds.frequency.size == 1:
                filtered_ds = filtered_ds.isel(frequency=0)
        except Exception as e:
            print(f"Error in swapping dims while plotting echogram: {e}")

    # Update axis labels
    filtered_ds['ping_time'].attrs = {
        'long_name': filtered_ds['ping_time'].attrs.get('long_name', 'Ping time'),
        'units': filtered_ds['ping_time'].attrs.get('units', '')
    }
    filtered_ds['range_sample'].attrs = {
        'long_name': filtered_ds['range_sample'].attrs.get('long_name', 'Depth'),
        'units': filtered_ds['range_sample'].attrs.get('units', '')
    }

    # Handle NaNs by dropping NaNs directly from ping_time and range_sample
    filtered_ds = filtered_ds.dropna(dim='ping_time', how='all')
    filtered_ds = filtered_ds.dropna(dim='range_sample', how='all')

    plt.figure(figsize=(30, 18))
    filtered_ds.isel(frequency=channel).T.plot(
        x='ping_time',
        y='range_sample',
        yincrease=False,
        vmin=-80,
        vmax=-50,
        cmap=cmap,
        cbar_kwargs={'label': 'Volume backscattering strength (Sv re 1 m-1)'}
    )

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Ping time', fontsize=14)
    plt.ylabel('Depth', fontsize=14)
    plt.title(f'Echogram for Channel {channel_name}', fontsize=16, fontweight='bold')

    echogram_file_name = f"{file_base_name}_{channel_name}.png"

    if config_data and 'cloud_storage' in config_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            echogram_output_path = os.path.join(echogram_path, echogram_file_name)
            upload_to_cloud_storage(temp_file.name, echogram_output_path, config_data['cloud_storage'])
            os.remove(temp_file.name)
    else:
        echogram_output_path = os.path.join(output_path, echogram_file_name)
        plt.savefig(echogram_output_path, dpi=300, bbox_inches='tight')
        plt.close()

    return echogram_output_path


def plot_sv_data_parallel(ds_Sv, file_base_name=None, output_path=None, cmap=None, client=None):
    """Plot the echogram data and the regions."""
    from dask.distributed import wait

    if not plt.isinteractive():
        plt.switch_backend('Agg')

    futures = []

    for channel in range(ds_Sv.dims['channel']):
        future = client.submit(plot_individual_channel_simplified, ds_Sv, channel, output_path, file_base_name, cmap)
        futures.append(future)

    wait(futures)


def plot_sv_data(ds_Sv, file_base_name=None, output_path=None, echogram_path=None, config_data=None, cmap=None):
    """Plot the echogram data and the regions."""
    if not plt.isinteractive():
        plt.switch_backend('Agg')

    echogram_files = []
    for channel in range(ds_Sv.dims['channel']):
        echogram_file_path = plot_individual_channel_simplified(ds_Sv, channel, output_path, file_base_name,
                                                                echogram_path=echogram_path,
                                                                config_data=config_data,
                                                                cmap='ocean_r')
        echogram_files.append(echogram_file_path)
        # plot_individual_channel_image_only(ds_Sv, channel, output_path, file_base_name, cmap)
        # plot_individual_channel_shaders(ds_Sv=ds_Sv, channel=channel, output_path=output_path,
        #                                 file_base_name=file_base_name, cmap='ocean_r')
    return echogram_files

async def plot_sv_data_with_progress(ds_Sv, file_base_name=None, output_path=None, progress=None, plot_task=None,
                                     cmap='viridis',
                                     regions2d=None, region_ids=None, region_class=None):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        total_steps = 100
        progress_step = 95 / total_steps

        plot_task_future = loop.run_in_executor(pool, plot_sv_data, ds_Sv, file_base_name, output_path, cmap,
                                                regions2d,
                                                region_ids,
                                                region_class)

        for step in range(total_steps):
            if plot_task_future.done():
                break
            await asyncio.sleep(0.1)
            progress.update(plot_task, advance=progress_step)

        await plot_task_future


def plot_all_channels(
        dataset1: xr.Dataset,
        dataset2: Optional[xr.Dataset] = None,
        variable_name: str = "Sv",
        name: str = "",
        save_path: Optional[Union[str, Path]] = "",
        **kwargs,
):
    """
    Plots echograms for all channels from one or two xarray Datasets.

    This function iterates over channels in the specified variable of the given dataset(s) and creates echogram plots.
    Each channel's data is plotted in a separate figure. When two datasets are provided, their respective echograms
    for each channel are plotted side by side for comparison.

    Parameters:
    - dataset1 (xr.Dataset): The first xarray Dataset to plot.
    - dataset2 (xr.Dataset, optional): The second xarray Dataset to plot alongside the first. Defaults to None.
    - variable_name (str, optional): The name of the variable to plot from the dataset. Defaults to "Sv".
    - name (str, optional): Base name for the output plot files. Defaults to empty string"".
    - save_path ((str, Path) optional): Path where to save the images default is current working dir.
    - **kwargs: Arbitrary keyword arguments. Commonly used for plot customization like `vmin`, `vmax`, and `cmap`.

    Saves:
    - PNG files for each channel's echogram, named using the variable name, the `name` parameter and channel name.

    Example:
    >> plot_all_channels(dataset1, dataset2, variable_name="Sv", name="echogram", vmin=-70, vmax=-30, cmap='inferno')
    This will create and save echogram plots comparing dataset1 and dataset2 for each channel, using specified plot settings.

    Note:
    - If only one dataset is provided, echograms for that dataset alone will be plotted.
    - The function handles plotting parameters such as color range (`vmin` and `vmax`) and colormap (`cmap`) via kwargs.
    """
    for ch in dataset1[variable_name].channel.values:
        plt.figure(figsize=(20, 10))

        # Configure plotting parameters
        plot_params = {
            "vmin": kwargs.get("vmin", -80),
            "vmax": kwargs.get("vmax", -50),
            "cmap": kwargs.get("cmap", "ocean_r"),
        }

        if dataset2:
            # First subplot for dataset1
            ax1 = plt.subplot(1, 2, 1)
            mappable1 = ax1.pcolormesh(
                np.rot90(dataset1[variable_name].sel(channel=ch).values), **plot_params
            )
            plt.title(f"Original Data {ch}")

            # Second subplot for dataset2
            ax2 = plt.subplot(1, 2, 2)
            ax2.pcolormesh(np.rot90(dataset2[variable_name].sel(channel=ch).values), **plot_params)
            plt.title(f"Downsampled Data {ch}")

            # Create a common colorbar
            plt.colorbar(mappable1, ax=[ax1, ax2], orientation="vertical")

        else:
            ax = plt.subplot(1, 1, 1)
            plt.pcolormesh(np.rot90(dataset1[variable_name].sel(channel=ch).values), **plot_params)
            plt.title(f"{variable_name} Data {ch}")

            # Create a colorbar
            plt.colorbar(ax=ax, orientation="vertical")

        # Save the figure
        if save_path:
            used_path = Path(save_path)
            used_path = used_path / f"{name}_{variable_name}_channel_{ch}.png"
        else:
            used_path = f"{name}_{variable_name}_channel_{ch}.png"
        plt.savefig(used_path)
        plt.close()


def plot_regions_only(regions2d=None, output_path=None, region_ids=None, region_classes=None, evr_base_name=None):
    """Plot the echogram data and the regions."""
    plt.figure(figsize=(20, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    idx = 0
    labels_added = set()

    regions_output_path = os.path.join(output_path, f"{evr_base_name}_regions.png")

    if region_ids:
        for region_id in region_ids:
            regions_closed = regions2d.data[regions2d.data['region_id'] == region_id]
            color = colors[idx % len(colors)]
            label = f"Region ID: {region_id}"
            for _, point in regions_closed.iterrows():
                plt.plot(point["time"], point["depth"], fillstyle='full', markersize=1, color=color,
                         label=label if label not in labels_added else "")
                labels_added.add(label)
            idx += 1

    if region_classes:
        for region_class in region_classes:
            regions_closed = regions2d.close_region(region_class=region_class)
            color = colors[idx % len(colors)]
            label = f"Region Class: {region_class}"
            for _, point in regions_closed.iterrows():
                plt.plot(point["time"], point["depth"], fillstyle='full', markersize=2, color=color,
                         label=label if label not in labels_added else "")
                labels_added.add(label)
            idx += 1

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Depth')

    plt.title('Regions with Labels')
    plt.savefig(regions_output_path)
    plt.show()


def _plot_region_classes(channel_name, colors, ds_Sv, idx, labels_added, region_class, regions2d):
    for region in region_class:
        regions_closed = regions2d.close_region(region_class=region)
        color = colors[idx % len(colors)]
        label = f"Region Class: {region}"
        for _, point in regions_closed.iterrows():
            depth = ds_Sv.range_sample.max().values - point["depth"]
            plt.plot(point["time"], depth, fillstyle='full', markersize=5, color=color,
                     label=label if label not in labels_added else "")
            labels_added.add(label)
        idx += 1


def _plot_region_ids(colors, ds_Sv, idx, labels_added, region_ids, regions2d):
    for region_id in region_ids:
        regions_closed = regions2d.close_region(region_id=region_id)
        color = colors[idx % len(colors)]
        label = f"Region ID: {region_id}"
        for _, point in regions_closed.iterrows():
            depth = ds_Sv.range_sample.max().values - point["depth"]
            plt.plot(point["time"], depth, fillstyle='full', markersize=5, color=color,
                     label=label if label not in labels_added else "")
            labels_added.add(label)
        idx += 1
    return idx


def upload_to_cloud_storage(local_path, remote_path, cloud_storage_config):
    storage_type = cloud_storage_config['storage_type']
    container_name = cloud_storage_config['container_name']
    storage_options = cloud_storage_config['storage_options']

    if storage_type == 'azure':
        upload_to_azure_blob(local_path, remote_path, container_name, storage_options)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


def upload_to_azure_blob(local_path, remote_path, container_name, storage_options):
    from azure.storage.blob import BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(storage_options['connection_string'])
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=remote_path)

    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    logging.info(f"Uploaded {local_path} to Azure Blob Storage as {remote_path}")
