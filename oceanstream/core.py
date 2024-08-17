import asyncio
import os
import logging
import sys

from pathlib import Path
from oceanstream.settings import load_config
from oceanstream.process import convert_raw_files, compute_single_file

DEFAULT_OUTPUT_FOLDER = "output"
DEFAULT_SONAR_MODEL = "EK60"

logging.basicConfig(level="ERROR", format='%(asctime)s - %(levelname)s - %(message)s')


def initialize(settings, file_path=None, log_level=None, chunks=None):
    logging.debug(f"Initializing with settings: {settings}, file path: {file_path}, log level: {log_level}")
    if "config" not in settings:
        settings["config"] = ""

    config_data = load_config(settings["config"])

    if chunks:
        config_data['chunks'] = chunks
    else:
        config_data['chunks'] = config_data.get('base_chunk_sizes', None)

    if file_path is not None:
        config_data["raw_path"] = file_path

    if log_level is not None:
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
        config_data["log_level"] = log_level

    if 'sonar_model' in settings and settings["sonar_model"] is not None:
        config_data["sonar_model"] = settings["sonar_model"]

    if settings.get('plot_echogram', None) is not None:
        config_data["plot_echogram"] = settings["plot_echogram"]

    if settings.get('waveform_mode', None) is not None:
        config_data["waveform_mode"] = settings["waveform_mode"]

    if settings.get('depth_offset', None) is not None:
        config_data["depth_offset"] = settings["depth_offset"]

    if settings.get("output_folder", None) is not None:
        config_data["output_folder"] = settings["output_folder"]

    if settings.get('cloud_storage', None) is not None:
        config_data['cloud_storage'] = settings['cloud_storage']

    return config_data


def process_raw_file(source, output=None, sonar_model=None, plot_echogram=False, depth_offset=0.0, waveform_mode="CW",
                     config=None, log_level="WARNING", chunks=None):
    logging.debug("Starting process_raw_file function")
    settings = {
        "config": config,
        "sonar_model": sonar_model,
        "output_folder": output or DEFAULT_OUTPUT_FOLDER
    }

    file_path = Path(source)
    config_data = initialize(settings, file_path, log_level=log_level)

    if chunks:
        config_data['chunks'] = chunks

    if file_path.is_file():
        logging.debug(f"Processing raw file: {file_path}")
        from oceanstream.process import process_raw_file_with_progress
        asyncio.run(
            process_raw_file_with_progress(config_data, plot_echogram, waveform_mode=waveform_mode,
                                           depth_offset=depth_offset)
        )
    else:
        logging.error(f"The provided path '{source}' is not a valid raw file.")
        sys.exit(1)


# def convert(source, output=None, base_path=None, workers_count=None, config=None, log_level="WARNING", chunks=None):
#     logging.debug("Starting convert function")
#     settings = {
#         "output_folder": output or DEFAULT_OUTPUT_FOLDER
#     }
#
#     if config is not None:
#         settings.update(config)
#
#     file_path = Path(source)
#     config_data = initialize(settings, file_path, log_level=log_level)
#
#     if chunks:
#         config_data['chunks'] = chunks
#     else:
#         config_data['chunks'] = config_data.get('base_chunk_sizes', None)
#
#     if file_path.is_file():
#         logging.debug(f"Converting raw file: {file_path}")
#         convert_raw_file(file_path, config_data, base_path=base_path)
#         logging.info(f"Converted raw file {source} to Zarr and wrote output to: {config_data['output_folder']}")
#     elif file_path.is_dir():
#         logging.debug(f"Converting raw files in directory: {file_path}")
#         convert_raw_files(config_data, workers_count=workers_count)
#     else:
#         logging.error(f"The provided path '{source}' is not a valid file/folder.")


def combine(source, output=None, config=None, log_level="WARNING", chunks=None):
    logging.debug("Starting combine function")
    settings = {
        "config": config,
        "output_folder": output or DEFAULT_OUTPUT_FOLDER
    }

    dir_path = Path(source)
    config_data = initialize(settings, dir_path, log_level=log_level)

    if chunks:
        config_data['chunks'] = chunks

    if dir_path.is_dir():
        logging.debug(f"Combining Zarr files in directory: {dir_path}")

        from oceanstream.process import combine_zarr_files

        file_name = f"{Path(dir_path).stem}-combined.zarr"
        zarr_output_file = os.path.join(config_data['output_folder'], file_name)
        logging.info(f"Combining Zarr files to {zarr_output_file}")

        combine_zarr_files(dir_path, zarr_output_file=zarr_output_file, chunks=chunks)
        logging.info("Zarr files have been combined successfully.")
    else:
        logging.error(f"The provided path '{source}' is not a valid folder.")


def compute_sv(source, output=None, workers_count=None, sonar_model=DEFAULT_SONAR_MODEL, plot_echogram=False,
               depth_offset=0.0, waveform_mode="CW", log_level="WARNING", chunks=None, config=None,
               processed_count_var=None):
    settings_dict = {
        "sonar_model": sonar_model,
        "output_folder": output or DEFAULT_OUTPUT_FOLDER
    }

    if config is not None:
        settings_dict.update(config)

    file_path = Path(source)
    config_data = initialize(settings_dict, file_path, log_level=log_level)

    if chunks:
        config_data['chunks'] = chunks
    else:
        config_data['chunks'] = config_data.get('base_chunk_sizes', None)

    if file_path.is_dir() and source.endswith(".zarr"):
        logging.debug(f"Computing Sv for Zarr root file: {file_path}")
        compute_single_file(config_data, chunks=chunks, plot_echogram=plot_echogram, waveform_mode=waveform_mode,
                            depth_offset=depth_offset)
    elif file_path.is_dir():
        logging.debug(f"Processing Zarr files in directory: {file_path}")
        from oceanstream.process import process_zarr_files

        process_zarr_files(config_data, workers_count=workers_count, chunks=chunks,
                           processed_count_var=processed_count_var, plot_echogram=plot_echogram,
                           waveform_mode=waveform_mode, depth_offset=depth_offset)
    else:
        logging.error(f"The provided path '{source}' is not a valid Zarr root.")


def export():
    logging.info("Export data...")
