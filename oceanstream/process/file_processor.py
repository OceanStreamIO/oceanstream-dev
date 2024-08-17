import json
import logging
import os
import sys
import time
import traceback
import shutil
from asyncio import CancelledError
import echopype as ep

import xarray as xr
from pathlib import Path
from rich import print
from rich.traceback import install, Traceback

from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

from oceanstream.plot import plot_sv_data_with_progress, plot_sv_data
from oceanstream.echodata import get_campaign_metadata, read_file
from oceanstream.denoise import apply_background_noise_removal
from oceanstream.exports import export_location_json

from .utils import save_output_data, append_gps_data
from .process import compute_sv, process_file_with_progress, read_file_with_progress

install(show_locals=False, width=120)


def compute_single_file(config_data, **kwargs):
    file_path = config_data["raw_path"]
    start_time = time.time()
    chunks = kwargs.get("chunks")
    echodata = ep.open_converted(file_path, chunks=chunks)

    try:
        output_path = compute_Sv_to_zarr(echodata, config_data, **kwargs)
        print(f"[blue]✅ Computed Sv and saved to: {output_path}[/blue]")
    except Exception as e:
        logging.error(f"Error computing Sv for {file_path}")
        print(Traceback())
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken: {total_time:.2f} seconds")


def compute_and_export_single_file(config_data, **kwargs):
    file_path = config_data["raw_path"]
    chunks = kwargs.get("chunks")
    echodata = ep.open_converted(file_path, chunks=chunks)

    try:
        file_data = []
        output_path, ds_processed, echogram_files = compute_Sv_to_zarr(echodata, config_data, **kwargs)
        # gps_data = export_location_json(ds_processed)

        output_message = {
            "filename": str(output_path),  # Convert PosixPath to string
            "file_npings": len(ds_processed["ping_time"].values),
            "file_nsamples": len(ds_processed["range_sample"].values),
            "file_start_time": str(ds_processed["ping_time"].values[0]),
            "file_end_time": str(ds_processed["ping_time"].values[-1]),
            "file_freqs": ",".join(map(str, ds_processed["frequency_nominal"].values)),
            "file_start_depth": str(ds_processed["range_sample"].values[0]),
            "file_end_depth": str(ds_processed["range_sample"].values[-1]),
            "file_start_lat": echodata["Platform"]["latitude"].values[0],
            "file_start_lon": echodata["Platform"]["longitude"].values[0],
            "file_end_lat": echodata["Platform"]["latitude"].values[-1],
            "file_end_lon": echodata["Platform"]["longitude"].values[-1],
            "echogram_files": echogram_files
        }
        file_data.append(output_message)

        json_file_path = Path(config_data["output_folder"]) / "output_data.json"
        gps_json_file_path = Path(config_data["output_folder"]) / "gps_data.json"

        save_output_data(output_message, json_file_path)
        # append_gps_data(gps_data, gps_json_file_path, str(output_path))

        echogram_folder = Path(config_data["output_folder"]) / "echograms"
        echogram_folder.mkdir(parents=True, exist_ok=True)

        for png_file in Path(output_path).glob("*.png"):
            shutil.copy(png_file, echogram_folder / png_file.name)
    except CancelledError as e:
        logging.info("CancelledError received, terminating processes...")
    except (ValueError, RuntimeError) as e:
        logging.error(f"Error computing Sv for {file_path}")
        print(Traceback())


def export_location_from_Sv_dataset(store, config_data, chunks=None):
    dataset = xr.open_zarr(store, chunks=chunks, consolidated=True)
    file_name = Path(store).stem
    file_name = file_name.replace("_Sv", "")
    gps_data = export_location_json(dataset)

    gps_json_file_path = Path(config_data["output_folder"]) / f"gps_data_{file_name}.json"
    append_gps_data(gps_data, gps_json_file_path, file_name)


def compute_Sv_to_zarr(echodata, config_data, base_path=None, chunks=None, plot_echogram=False, **kwargs):
    """
    Compute Sv from echodata and save to zarr file.

    Args:
        echodata:
        config_data:
        base_path:
        chunks:
        plot_echogram:
        **kwargs:

    Returns:
        str: Path to the zarr file.
    """
    waveform_mode = kwargs.get("waveform_mode", "CW")
    if encode_mode := kwargs.pop("encode_mode", None):
        encode_mode = encode_mode.lower()
    else:
        encode_mode = waveform_mode == "CW" and "power" or "complex"

    echodata["Vendor_specific"] = echodata["Vendor_specific"].isel(channel=slice(1))

    Sv = compute_sv(echodata, encode_mode=encode_mode, **kwargs)

    ds_processed = Sv
    # ds_processed = apply_background_noise_removal(Sv, config=config_data)

    output_path = None
    file_base_name = None
    file_path = None

    if config_data.get('raw_path') is not None:
        file_path = config_data["raw_path"]
        file_base_name = file_path.stem
        if base_path:
            relative_path = file_path.relative_to(base_path)

            if relative_path.parent != ".":
                zarr_path = Path(relative_path.parent) / file_base_name
            else:
                zarr_path = relative_path.stem
        else:
            zarr_path = file_base_name

        output_path = Path(config_data["output_folder"]) / zarr_path
        output_path.mkdir(parents=True, exist_ok=True)

        zarr_file_name = f"{file_base_name}_Sv.zarr"
    else:
        zarr_path = base_path
        zarr_file_name = f"{zarr_path}_Sv.zarr"

    echogram_path = zarr_path
    if chunks is not None:
        for var in Sv.data_vars:
            var_chunk_sizes = _get_chunk_sizes(Sv[var].dims, chunks)
            Sv[var] = Sv[var].chunk(var_chunk_sizes)
            # Remove chunk encoding to avoid conflicts
            if 'chunks' in Sv[var].encoding:
                del Sv[var].encoding['chunks']
    print(f"\nWriting zarr file to: {output_path}/{zarr_file_name}")
    write_zarr_file(zarr_path, zarr_file_name, ds_processed, config_data, output_path)

    echogram_files = None
    if plot_echogram:
        try:
            echogram_files = plot_sv_data(ds_processed, file_base_name=file_base_name, output_path=output_path,
                                          echogram_path=echogram_path, config_data=config_data)
        except Exception as e:
            logging.error(f"Error plotting echogram for {file_path}: {e}")

    return output_path, ds_processed, echogram_files


####################################################################################################

async def process_raw_file_with_progress(config_data, plot_echogram, waveform_mode="CW", depth_offset=0):
    try:
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
        ) as progress:
            print(f"[green] Processing file: {config_data['raw_path']}[/green]")
            read_task = progress.add_task("[cyan]Reading raw file data...", total=100)

            campaign_id, date, sonar_model, metadata, _ = get_campaign_metadata(config_data['raw_path'])
            if config_data['sonar_model'] is None:
                config_data['sonar_model'] = sonar_model

            echodata, encode_mode = await read_file_with_progress(config_data, progress, read_task)
            echodata.to_zarr(save_path=config_data["output_folder"], overwrite=True, parallel=False)
            progress.update(read_task, advance=100 - progress.tasks[read_task].completed)

            if plot_echogram:
                zarr_file_name = config_data['raw_path'].stem

                if waveform_mode == "BB":
                    encode_mode = "complex"

                compute_task = progress.add_task(
                    f"[cyan]Computing Sv with waveform_mode={waveform_mode} and encode_mode={encode_mode}...",
                    total=100)

                sv_dataset = await process_file_with_progress(progress, compute_task, echodata,
                                                              encode_mode=encode_mode,
                                                              waveform_mode=waveform_mode,
                                                              depth_offset=depth_offset)
                progress.update(compute_task, advance=100 - progress.tasks[compute_task].completed)
                print(f"[blue]📝 Computed Sv and wrote zarr file to: {config_data['output_folder']}[/blue]")

                print(f"[green]✅ Plotting echogram for: {config_data['raw_path']}[/green]")
                plot_task = progress.add_task("[cyan]Plotting echogram...", total=100)
                await plot_sv_data_with_progress(sv_dataset, output_path=config_data["output_folder"],
                                                 progress=progress, file_base_name=zarr_file_name, plot_task=plot_task)
                progress.update(plot_task, advance=100 - progress.tasks[plot_task].completed)
                print(f"[blue]📊 Plotted echogram for the data in: {config_data['output_folder']}[/blue]")
    except Exception as e:
        logging.exception(f"Error processing file {config_data['raw_path']}: {e}")


def write_zarr_file(zarr_path, zarr_file_name, ds_processed, config_data=None, output_path=None):
    if 'cloud_storage' in config_data:
        from cloud import get_chunk_store

        store = get_chunk_store(config_data['cloud_storage'], Path(zarr_path) / zarr_file_name)
    else:
        store = os.path.join(output_path, zarr_file_name)

    ds_processed.to_zarr(store, mode='w')


def _get_chunk_sizes(var_dims, chunk_sizes):
    return {dim: chunk_sizes[dim] for dim in var_dims if dim in chunk_sizes}
