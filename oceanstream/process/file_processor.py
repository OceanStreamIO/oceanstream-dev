import logging
import os
import sys
import time
import traceback
import echopype as ep

from pathlib import Path
from rich import print
from rich.traceback import install, Traceback

from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

from oceanstream.plot import plot_sv_data_with_progress, plot_sv_data
from oceanstream.echodata import get_campaign_metadata
from .process import compute_sv, process_file_with_progress, read_file_with_progress


install(show_locals=True, width=120)


def get_chunk_sizes(var_dims, chunk_sizes):
    return {dim: chunk_sizes[dim] for dim in var_dims if dim in chunk_sizes}


def compute_Sv_to_zarr(echodata, config_data, chunks=None, plot_echogram=False, **kwargs):
    """
    Compute Sv from echodata and save to zarr file.

    Args:
        echodata:
        config_data:
        chunks:
        plot_echogram:
        **kwargs:

    Returns:
        str: Path to the zarr file.
    """
    file_path = config_data["raw_path"]
    waveform_mode = kwargs.get("waveform_mode", "CW")
    encode_mode = waveform_mode == "CW" and "power" or "complex"
    Sv = compute_sv(echodata, encode_mode=encode_mode, **kwargs)

    parent_folder = os.path.join(Path(config_data["output_folder"]), file_path.stem)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    output_path = os.path.join(parent_folder, f"{file_path.stem}_Sv.zarr")

    if chunks is not None:
        for var in Sv.data_vars:
            var_chunk_sizes = get_chunk_sizes(Sv[var].dims, chunks)
            Sv[var] = Sv[var].chunk(var_chunk_sizes)
            # Remove chunk encoding to avoid conflicts
            if 'chunks' in Sv[var].encoding:
                del Sv[var].encoding['chunks']

    print("Removing background noise...")
    ds_processed = Sv
    # ds_processed = apply_background_noise_removal(Sv, config=config_data)
    ds_processed.to_zarr(output_path)

    if plot_echogram:
        try:
            plot_sv_data(ds_processed, file_base_name=file_path.stem, output_path=parent_folder)
        except Exception as e:
            logging.exception(f"Error plotting echogram for {file_path}:")
            raise e

    return output_path


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


def convert_raw_file(file_path, config_data, progress_queue=None):
    logging.debug("Starting processing of file: %s", file_path)
    from oceanstream.echodata import read_file

    try:
        file_config_data = {**config_data, 'raw_path': Path(file_path)}
        echodata, encode_mode = read_file(file_config_data, use_swap=True, skip_integrity_check=True)
        echodata.to_zarr(save_path=config_data["output_folder"], overwrite=True, parallel=False)

        if progress_queue:
            progress_queue.put(file_path)
    except Exception as e:
        logging.error("Error processing file %s", file_path)
        print(Traceback())


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
