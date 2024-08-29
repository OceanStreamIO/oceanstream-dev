import logging
import sys

from pathlib import Path
from oceanstream.echodata import read_file

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('oceanstream')


def convert_raw_file(file_path, config_data, base_path=None, progress_counter=None, counter_lock=None):
    logger.info("Starting processing of file: %s", file_path)

    file_path_obj = Path(file_path)
    file_config_data = {**config_data, 'raw_path': file_path_obj}

    if base_path:
        relative_path = file_path_obj.relative_to(base_path)
        relative_path = relative_path.parent
    else:
        relative_path = None

    echodata, encode_mode = read_file(file_config_data, use_swap=True, skip_integrity_check=True)
    file_name = file_path_obj.stem + ".zarr"

    if 'cloud_storage' in config_data:
        from oceanstream.process.cloud import get_chunk_store

        if relative_path:
            file_location = Path(relative_path) / file_name
        else:
            file_location = file_name
        store = get_chunk_store(config_data['cloud_storage'], file_location)
        echodata.to_zarr(save_path=store, overwrite=True, parallel=False)

        output_dest = config_data['cloud_storage']['container_name'] + "/" + file_location
    else:
        if relative_path:
            output_path = Path(config_data["output_folder"]) / relative_path
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(config_data["output_folder"])

        output_dest = output_path / file_name
        echodata.to_zarr(save_path=output_dest, overwrite=True, parallel=False)

    if counter_lock is not None:
        with counter_lock:
            progress_counter.value += 1

    logger.info("Finished converting file: %s", file_path)

    return output_dest