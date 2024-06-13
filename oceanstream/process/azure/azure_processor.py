import re

from asyncio import CancelledError
from rich import print
from rich.traceback import install, Traceback
from tqdm.auto import tqdm
from dask import delayed
from .blob_storage import open_zarr_store

install(show_locals=False, width=120)


def process_survey_data_with_progress(files_by_day, azfs, container_name, client, config_data, process_func):
    try:
        total_files = sum(len(files) for files in files_by_day.values())
        progress_bar = tqdm(total=total_files, desc="Processing Files", unit="file", ncols=100)

        def update_progress(*args):
            progress_bar.update()

        futures = process_survey_data(files_by_day, azfs, container_name, client, config_data, process_func)

        for future in futures:
            future.add_done_callback(update_progress)

        client.gather(futures)  # Ensure all tasks complete

        progress_bar.close()
    except KeyboardInterrupt:
        print("Closing down.")
    except CancelledError:
        print("Closing down.")
    except Exception as e:
        print(f"[bold red]An error occurred:[/bold red] {e}")
        print(f"{Traceback()}\n")


def process_survey_data(files_by_day, azfs, container_name, dask_client, config_data, process_func):
    """Process survey data from S3."""

    tasks = []

    for day, files in files_by_day.items():
        for file in files:
            task = delayed(_process_zarr_file)(file['Key'], azfs, container_name, config_data, process_func)
            tasks.append(task)

    # Execute all tasks in parallel
    futures = dask_client.compute(tasks)

    return futures


def _process_zarr_file(file, azfs, container_name, config_data, process_func=None):
    """Process a single Zarr file."""
    base_path = file.replace('.zarr', '')
    pattern = rf"^{re.escape(container_name)}/"
    base_path = re.sub(pattern, '', base_path)

    echodata = open_zarr_store(azfs, file, chunks=config_data['chunks'])

    process_func(echodata, config_data, base_path=base_path,
                 chunks=config_data.get('chunks'),
                 plot_echogram=config_data.get('plot_echogram', False),
                 waveform_mode=config_data.get('waveform_mode', "CW"),
                 depth_offset=config_data.get('depth_offset', 0.0))
