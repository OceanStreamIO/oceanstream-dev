import json
import os
import logging
import time
import re
import signal
import sys
import traceback
import inspect
import warnings
from queue import Queue
from threading import Thread

from dask import delayed, compute
from distributed import Semaphore
from pathlib import Path
from datetime import datetime
from rich import print
from functools import partial
from multiprocessing import Pool, Manager, Process
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from oceanstream.plot import plot_sv_data
from oceanstream.echodata import get_campaign_metadata, read_file
from tqdm.auto import tqdm

from .combine_zarr import read_zarr_files
from .process import compute_sv
from .processed_data_io import write_processed
from .file_processor import compute_single_file, compute_and_export_single_file, \
    export_location_from_Sv_dataset

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", module="echopype")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pool = None


def configure_logging(log_level=logging.ERROR):
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


def create_progress_bar():
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.completed]{task.completed} of {task.total} files"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    )


def populate_metadata(ed, raw_fname):
    """
    Manually populate into the "ed" EchoData object
    additional metadata about the dataset and the platform
    """

    # -- SONAR-netCDF4 Top-level Group attributes
    survey_name = (
        "2017 Joint U.S.-Canada Integrated Ecosystem and "
        "Pacific Hake Acoustic Trawl Survey ('Pacific Hake Survey')"
    )
    ed['Top-level'].attrs['title'] = f"{survey_name}, file {raw_fname}"
    ed['Top-level'].attrs['summary'] = (
        f"EK60 raw file {raw_fname} from the {survey_name}, converted to a SONAR-netCDF4 file using echopype."
        "Information about the survey program is available at "
        "https://www.fisheries.noaa.gov/west-coast/science-data/"
        "joint-us-canada-integrated-ecosystem-and-pacific-hake-acoustic-trawl-survey"
    )

    # -- SONAR-netCDF4 Platform Group attributes
    # Per SONAR-netCDF4, for platform_type see https://vocab.ices.dk/?ref=311
    ed['Platform'].attrs['platform_type'] = "Research vessel"
    ed['Platform'].attrs['platform_name'] = "Bell M. Shimada"  # A NOAA ship
    ed['Platform'].attrs['platform_code_ICES'] = "315"


def update_progress(progress_queue, total_files, log_level):
    configure_logging(log_level)
    logging.debug("Initializing progress updater with total files: %d", total_files)
    with create_progress_bar() as progress:
        task = progress.add_task("[cyan]Processing files...", total=total_files)
        completed_files = 0

        while completed_files < total_files:
            try:
                message = progress_queue.get(timeout=1)
                if message is None:
                    break
                completed_files += 1
                progress.update(task, advance=1)
                logging.debug("Files completed: %d / %d", completed_files, total_files)
            except Exception:
                pass


def process_single_file(file_path, config_data, progress_queue,
                        compute_sv_data=False,
                        use_distributed_dask=False,
                        plot_echogram=False, **kwargs):
    log_level = config_data.get('log_level', logging.ERROR)
    configure_logging(log_level)
    logging.debug("Starting processing of file: %s", file_path)
    try:
        logging.debug("Reading file: %s", file_path)

        file_config_data = {**config_data, 'raw_path': Path(file_path)}
        echodata, encode_mode = read_file(file_config_data, use_swap=True, skip_integrity_check=True)
        echodata.to_zarr(save_path=config_data["output_folder"], overwrite=True, parallel=False)

        if compute_sv_data and use_distributed_dask:
            sv_dataset = compute_sv(echodata, encode_mode, **kwargs)
            zarr_file_name = file_config_data['raw_path'].stem
            write_processed(sv_dataset, config_data["output_folder"], zarr_file_name, "zarr")

        if plot_echogram:
            sv_dataset = compute_sv(echodata, encode_mode, **kwargs)
            zarr_file_name = file_config_data['raw_path'].stem
            plot_sv_data(sv_dataset, output_path=config_data["output_folder"], file_base_name=zarr_file_name)

        progress_queue.put(file_path)
    except Exception as e:
        logging.exception("Error processing file %s: %s", file_path, e)


def print_call_stack():
    stack = inspect.stack()
    for frame in stack:
        print(f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.function}")


def find_raw_files(base_dir):
    raw_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.raw'):
                raw_files.append(os.path.join(root, file))
    return raw_files


def update_convert_raw(pgr_queue, pb):
    while True:
        item = pgr_queue.get()
        if item is None:
            break
        pb.update(item)


def convert_raw_files(config_data, workers_count=os.cpu_count()):
    global pool

    from oceanstream.convert import convert_raw_file

    try:
        dir_path = config_data['raw_path']
        log_level = config_data.get('log_level', logging.ERROR)
        configure_logging(log_level)

        print(f"Starting to convert folder: {dir_path} to Zarr using {workers_count} parallel processes...")
        raw_files = find_raw_files(dir_path)
        file_info = []

        if not raw_files:
            logging.error("No raw files found in directory: %s", dir_path)
            return

        print(f"Found {len(raw_files)} raw files in directory.")

        for file in raw_files:
            creation_time = from_filename(file)
            file_info.append((file, creation_time))
            logging.debug("File: %s, creation time: %s", file, creation_time)

        if not file_info:
            logging.error("No valid raw files with creation time found in directory: %s", dir_path)
            return

        filtered_file_info = [item for item in file_info if item[1] is not None]
        filtered_file_info.sort(key=lambda x: x[1])

        sorted_files = [file for file, _ in filtered_file_info]

        campaign_id, date, sonar_model, metadata, _ = get_campaign_metadata(sorted_files[0])
        if sonar_model is None and config_data['sonar_model'] is None:
            config_data['sonar_model'] = sonar_model

        progress_bar = tqdm(
            total=len(sorted_files),
            desc="Processing Files",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files [per file: {elapsed}, {rate_fmt}]",
            leave=True
        )

        with Manager() as manager:
            progress_counter = manager.Value('i', 0)
            counter_lock = manager.Lock()
            progress_queue = manager.Queue()
            pool = Pool(processes=workers_count)

            # Partial function with config_data, progress_queue and other arguments
            process_func = partial(convert_raw_file, config_data=config_data, base_path=dir_path,
                                   progress_counter=progress_counter, counter_lock=counter_lock)

            for file in sorted_files:
                pool.apply_async(process_func, args=(file,))
                # print("Processing file: ", file)
                logging.debug("Started async processing for file: %s", file)

            pool.close()

            while True:
                with counter_lock:
                    progress_value = progress_counter.value
                progress_bar.n = progress_value
                progress_bar.refresh()

                # Check if all files have been processed
                if progress_value >= len(sorted_files):
                    break

            pool.join()

            with counter_lock:
                progress_bar.n = progress_counter.value
            progress_bar.refresh()
            print(f"[green]✅ All files have been converted.[/green]")

            # Wait for the progress updater to finish

            progress_bar.close()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, terminating processes...")
        if pool:
            pool.terminate()
            pool.join()
    except Exception as e:
        logging.exception("Error processing folder %s: %s", config_data['raw_path'], e)


def process_single_zarr_file(file_path, config_data, base_path=None, chunks=None, plot_echogram=False,
                             waveform_mode="CW",
                             depth_offset=0):
    file_config_data = {**config_data, 'raw_path': Path(file_path)}
    print(f"Processing file: {file_path}")
    # print_call_stack()

    compute_and_export_single_file(file_config_data, base_path=base_path, chunks=chunks, plot_echogram=plot_echogram,
                                   waveform_mode=waveform_mode, depth_offset=depth_offset)
    print(f"Finished processing file: {file_path}")


def process_zarr_files(config_data, client, workers_count=None, chunks=None, plot_echogram=False, waveform_mode="CW", depth_offset=0):
    dir_path = config_data['raw_path']
    zarr_files = read_zarr_files(dir_path)

    if not zarr_files:
        logging.error("No valid .zarr files with creation time found in directory: %s", dir_path)
        return

    max_concurrent_tasks = workers_count or os.cpu_count()
    total_files = len(zarr_files)
    print(f"Found {total_files} Zarr files in directory: {dir_path}\n")
    progress_bar = tqdm(total=total_files, desc="Processing Files", unit="file", ncols=100)
    task_queue = Queue()
    results = []

    def process_file_task(file_path):
        @delayed
        def process_task():
            process_single_zarr_file(file_path, config_data, chunks=chunks, base_path=dir_path,
                                     plot_echogram=plot_echogram, waveform_mode=waveform_mode,
                                     depth_offset=depth_offset)
            return file_path

        return process_task()

    def worker():
        while True:
            file_path = task_queue.get()
            if file_path is None:
                break
            try:
                task = process_file_task(file_path)
                result = task.compute()
                results.append(result)
                progress_bar.update()
                print(f"Task completed: {file_path}")
            except Exception as e:
                print(f"Error processing file: {file_path}, error: {e}")
            finally:
                task_queue.task_done()

    # Enqueue initial tasks
    for file_path in zarr_files:
        task_queue.put(file_path)

    # Create and start worker threads
    threads = []
    for _ in range(max_concurrent_tasks):
        thread = Thread(target=worker)
        thread.start()
        threads.append(thread)

    # Block until all tasks are done
    task_queue.join()

    # Stop workers
    for _ in range(max_concurrent_tasks):
        task_queue.put(None)
    for thread in threads:
        thread.join()

    progress_bar.close()
    print("✅ All files have been processed")


def export_location_from_zarr_files(config_data, client=None, workers_count=os.cpu_count(), chunks=None):
    semaphore = Semaphore(max_leases=workers_count)

    tasks = []

    dir_path = config_data['raw_path']
    zarr_files = read_zarr_files(dir_path)
    progress_bar = tqdm(total=len(zarr_files), desc="Processing Files", unit="file", ncols=100)

    def update_progress_fn(*args):
        progress_bar.update()

    if not zarr_files:
        logging.error("No valid .zarr files with creation time found in directory: %s", dir_path)
        return

    logging.info(f"Found {len(zarr_files)} Zarr files in directory: {dir_path}\n")

    for file_path in zarr_files:
        task = delayed(export_location_from_Sv_dataset)(file_path, config_data, chunks=chunks)
        tasks.append(task)

    # Execute all tasks in parallel
    futures = client.compute(tasks)

    for future in futures:
        future.add_done_callback(update_progress_fn)

    client.gather(futures)
    progress_bar.close()
    logging.info("✅ All files have been processed")
    merge_json_files(dir_path)


def merge_json_files(output_folder):
    json_files = list(Path(output_folder).glob("gps_data_*.json"))
    all_data = []

    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

    merged_json_file_path = Path(output_folder) / "gps_data.json"
    with open(merged_json_file_path, 'w') as merged_file:
        json.dump(all_data, merged_file, indent=4)

    logging.info(f"✅ Merged {len(json_files)} JSON files into {merged_json_file_path}")


def from_filename(file_name):
    """Extract creation time from the file name if it follows a specific pattern."""
    # pattern = r'(\d{4}[A-Z])?-D(\d{8})-T(\d{6})\.raw'
    pattern = r'.*-D(\d{8})-T(\d{6})(-\d+)?\.raw'
    match = re.search(pattern, file_name)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        creation_time = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        return creation_time

    return None


# def signal_handler(sig, frame):
#     global pool
#     print('Terminating processes...')
#     if pool:
#         pool.terminate()
#         pool.join()
#     sys.exit(0)
#
#
# signal.signal(signal.SIGINT, signal_handler)
