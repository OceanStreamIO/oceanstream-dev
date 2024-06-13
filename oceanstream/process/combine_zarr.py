import logging
import echopype as ep
import re
import time
from datetime import datetime
from rich import print
from pathlib import Path
from oceanstream.echodata import check_reversed_time, fix_time_reversions


logging.basicConfig(level="DEBUG", format='%(asctime)s - %(levelname)s - %(message)s')

def configure_logging(log_level=logging.ERROR):
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


def read_zarr_files(input_folder):
    input_path = Path(input_folder)
    if not input_path.is_dir():
        logging.error("Input folder does not exist: %s", input_folder)
        return

    zarr_files = list(input_path.rglob("*.zarr"))
    if not zarr_files:
        logging.error("No .zarr files found in directory: %s", input_folder)
        return

    file_info = []
    for file in zarr_files:
        creation_time = from_filename(file.stem)
        file_info.append((file, creation_time))

    filtered_file_info = [item for item in file_info if item[1] is not None]
    filtered_file_info.sort(key=lambda x: x[1])

    sorted_files = [file for file, _ in filtered_file_info]

    return sorted_files


def fix_time(ed):
    if check_reversed_time(ed, "Sonar/Beam_group1", "ping_time"):
        ed = fix_time_reversions(ed, {"Sonar/Beam_group1": "ping_time"})
    if check_reversed_time(ed, "Environment", "time1"):
        ed = fix_time_reversions(ed, {"Environment": "time1"})
    return ed


def combine_zarr_files(input_folder, zarr_output_file=None, chunks=None):
    start_time = time.time()

    logging.debug("Starting to combine Zarr files from folder: %s", input_folder)

    output_path = Path(zarr_output_file)
    sorted_files = read_zarr_files(input_folder)
    logging.debug("Found %d .zarr files in directory: %s", len(sorted_files), input_folder)

    if not sorted_files:
        logging.error("No valid .zarr files with creation time found in directory: %s", input_folder)
        return

    ed_list = []

    for zarr_file in sorted_files:
        ed = ep.open_converted(zarr_file, chunks=chunks)
        ed = fix_time(ed)

        ed_list.append(ed)

    combined_ed = ep.combine_echodata(ed_list)
    combined_ed.to_zarr(output_path)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal time taken: {total_time:.2f} seconds")


def from_filename(file_name):
    """Extract creation time from the file name if it follows a specific pattern."""
    pattern = r'(\d{4}[A-Z])?-D(\d{8})-T(\d{6})'
    match = re.search(pattern, file_name)
    if match:
        date_str = match.group(2)
        time_str = match.group(3)
        creation_time = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        return creation_time

    return None
