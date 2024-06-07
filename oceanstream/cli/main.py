import traceback

import typer
import asyncio
import os
import logging
import sys
import warnings

from pathlib import Path
from rich import print
from rich.traceback import install, Traceback
from oceanstream.settings import load_config
from dask.distributed import LocalCluster, Client
from rich.console import Console


install(show_locals=False, width=120)

DEFAULT_OUTPUT_FOLDER = "output"
DEFAULT_SONAR_MODEL = "EK60"

BANNER = """
                                   _                            
                                  | |                           
   ___   ___ ___  __ _ _ __    ___| |_ _ __ ___  __ _ _ __ ___  
  / _ \ / __/ _ \/ _` | '_ \  / __| __| '__/ _ \/ _` | '_ ` _ \ 
 | (_) | (_|  __/ (_| | | | | \__ \ |_| | |  __/ (_| | | | | | |
  \___/ \___\___|\__,_|_| |_| |___/\__|_|  \___|\__,_|_| |_| |_|

"""

logging.basicConfig(level="ERROR", format='%(asctime)s - %(levelname)s - %(message)s')


def initialize(settings, file_path, log_level=None):
    config_data = load_config(settings["config"])
    config_data["raw_path"] = file_path

    if log_level is not None:
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
        config_data["log_level"] = log_level

    if 'sonar_model' in settings and settings["sonar_model"] is not None:
        config_data["sonar_model"] = settings["sonar_model"]

    if settings["output_folder"] is not None:
        config_data["output_folder"] = settings["output_folder"]

    return config_data


app = typer.Typer(help="OceanStream CLI")


@app.command()
def process(
        source: str = typer.Option(..., help="Path to a raw data file"),
        output: str = typer.Option(None,
                                   help="Destination path for saving processed data. Defaults to a predefined "
                                        "directory if not specified."),
        sonar_model: str = typer.Option(None, help="Sonar model used to collect the data",
                                        show_choices=["AZFP", "EK60", "ES70", "EK80", "ES80", "EA640", "AD2CP"]),
        plot_echogram: bool = typer.Option(False, help="Plot the echogram after processing"),
        depth_offset: float = typer.Option(0.0, help="Depth offset for the echogram plot"),
        waveform_mode: str = typer.Option("CW", help="Waveform mode, can be either CW or BB",
                                          show_choices=["CW", "BB"]),
        config: str = typer.Option(None, help="Path to a configuration file"),
        log_level: str = typer.Option("WARNING", help="Set the logging level",
                                      show_choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
):
    """
    Process the input raw file and save the result to the specified output folder.
    """
    settings = {
        "config": config,
        "sonar_model": sonar_model,
        "output_folder": output or DEFAULT_OUTPUT_FOLDER
    }

    filePath = Path(source)
    configData = initialize(settings, filePath, log_level=log_level)

    if filePath.is_file():
        from oceanstream.process import process_raw_file_with_progress
        asyncio.run(
            process_raw_file_with_progress(configData, plot_echogram, waveform_mode=waveform_mode,
                                           depth_offset=depth_offset))
    else:
        print(f"[red]❌ The provided path '{source}' is not a valid raw file.[/red]")
        sys.exit(1)


@app.command()
def convert(
        source: str = typer.Option(..., help="Path to a raw data file/folder"),
        output: str = typer.Option(None,
                                   help="Destination path for saving Zarr converted data. Defaults to a predefined "
                                        "directory if not specified."),
        workers_count: int = typer.Option(os.cpu_count(), help="Number of CPU workers to use for processing"),
        config: str = typer.Option(None, help="Path to a configuration file"),
        log_level: str = typer.Option("WARNING", help="Set the logging level",
                                      show_choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
):
    """
    Convert the input raw data file(s) to Zarr and save the result to the specified output folder.
    """
    settings = {
        "config": config,
        "output_folder": output or DEFAULT_OUTPUT_FOLDER
    }

    filePath = Path(source)
    configData = initialize(settings, filePath, log_level=log_level)

    if filePath.is_file():
        from oceanstream.process import convert_raw_file
        convert_raw_file(filePath, configData)
        print(f"[blue]✅ Converted raw file {source} to Zarr and wrote output to: {configData['output_folder']} [/blue]")
    elif filePath.is_dir():
        from oceanstream.process import convert_raw_files
        convert_raw_files(configData, workers_count=workers_count)
    else:
        print(f"[red]❌ The provided path '{source}' is not a valid file/folder.[/red]")
        sys.exit(1)


@app.command()
def combine(
        source: str = typer.Option(..., help="Path to a source data folder where zarr files are located"),
        output: str = typer.Option(None,
                                   help="Destination path for saving combined zarr file. Defaults to a predefined "
                                        "directory if not specified."),
        workers_count: int = typer.Option(os.cpu_count(), help="Number of CPU workers to use for processing"),
        config: str = typer.Option(None, help="Path to a configuration file"),
        log_level: str = typer.Option("WARNING", help="Set the logging level",
                                      show_choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
):
    """
    Process the input zarr data files and save the combined zarr file to the specified output folder.
    """
    settings = {
        "config": config,
        "output_folder": output or DEFAULT_OUTPUT_FOLDER
    }

    dir_path = Path(source)
    configData = initialize(settings, dir_path, log_level=log_level)

    if dir_path.is_dir():
        console = Console()
        with console.status("Processing...", spinner="dots") as status:
            status.start()

            cluster = LocalCluster(n_workers=workers_count, threads_per_worker=1)
            client = Client(cluster)

            try:
                from oceanstream.process import combine_zarr_files

                file_name = f"{Path(dir_path).stem}-combined.zarr"
                zarr_output_file = os.path.join(configData['output_folder'], file_name)
                status.update(
                    f"Combining zarr files to {zarr_output_file}; navigate to http://localhost:8787/status for progress")

                combine_zarr_files(dir_path,
                                   zarr_output_file=zarr_output_file,
                                   chunks=configData.get('base_chunk_sizes'))
                print("\nZarr files have been combined successfully.")
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt received, terminating processes...")
            except Exception as e:
                logging.exception("Error processing folder %s", configData['raw_path'])
                traceback.print_exc()
            finally:
                status.stop()
                client.close()
                cluster.close()
    else:
        print(f"[red]❌ The provided path '{source}' is not a valid folder.[/red]")
        sys.exit(1)


@app.command()
def compute_sv(
        source: str = typer.Option(..., help="Path to a Zarr root file or a directory containing Zarr files"),
        output: str = typer.Option(None,
                                   help="Destination path for saving Sv data. Defaults to a predefined directory if not specified."),
        workers_count: int = typer.Option(os.cpu_count(), help="Number of CPU workers to use for Sv computation"),
        sonar_model: str = typer.Option(DEFAULT_SONAR_MODEL, help="Sonar model used to collect the data",
                                        show_choices=["AZFP", "EK60", "ES70", "EK80", "ES80", "EA640", "AD2CP"]),
        plot_echogram: bool = typer.Option(False, help="Plot the echogram after processing"),
        depth_offset: float = typer.Option(0.0, help="Depth offset for the echogram plot"),
        waveform_mode: str = typer.Option("CW", help="Waveform mode, can be either CW or BB",
                                          show_choices=["CW", "BB"]),
        config: str = typer.Option(None, help="Path to a configuration file"),
        log_level: str = typer.Option("WARNING", help="Set the logging level",
                                      show_choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
):
    """
    Compute the volume backscattering strength (Sv) from the Zarr file.
    """
    settings_dict = {
        "config": config,
        "sonar_model": sonar_model,
        "output_folder": output or DEFAULT_OUTPUT_FOLDER
    }
    file_path = Path(source)
    config_data = initialize(settings_dict, file_path, log_level=log_level)

    console = Console()
    with console.status("Processing...", spinner="dots") as status:
        status.start()
        cluster = LocalCluster(n_workers=workers_count, threads_per_worker=1)
        client = Client(cluster)

        try:
            if file_path.is_dir() and source.endswith(".zarr"):
                status.update(
                    f"[blue] Computing Sv for {file_path}...[/blue] – navigate to http://localhost:8787/status for progress")
                from oceanstream.process import compute_single_file

                compute_single_file(config_data,
                                    chunks=config_data.get('base_chunk_sizes'),
                                    plot_echogram=plot_echogram,
                                    waveform_mode=waveform_mode,
                                    depth_offset=depth_offset)
            elif file_path.is_dir():
                status.update(
                    f"[blue] Processing zarr files in {file_path}...[/blue] – navigate to "
                    f"http://localhost:8787/status for progress")
                from oceanstream.process import process_zarr_files

                process_zarr_files(config_data,
                                   workers_count=workers_count,
                                   status=status,
                                   chunks=config_data.get('base_chunk_sizes'),
                                   plot_echogram=plot_echogram,
                                   waveform_mode=waveform_mode,
                                   depth_offset=depth_offset)
            else:
                print(f"[red]❌ The provided path '{source}' is not a valid Zarr root.[/red]")
                sys.exit(1)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received, terminating processes...")
        except Exception as e:
            logging.exception("Error while processing %s", config_data['raw_path'])
            traceback.print_exc()
        finally:
            client.close()
            cluster.close()
            status.stop()


@app.command()
def export():
    typer.echo("Export data...")


def main():
    print(BANNER)
    warnings.filterwarnings("ignore", category=UserWarning)
    app(prog_name="oceanstream")


if __name__ == "__main__":
    main()
