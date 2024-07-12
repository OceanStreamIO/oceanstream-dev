import os
import tempfile
import botocore
import boto3
from pathlib import Path
from asyncio import CancelledError
from rich import print
from rich.traceback import install, Traceback
from tqdm.auto import tqdm
from dask import delayed
from botocore.config import Config


install(show_locals=False, width=120)


def process_survey_data_with_progress(files_by_day, bucket_name, client, config_data, process_func):
    try:
        total_files = sum(len(files) for files in files_by_day.values())
        progress_bar = tqdm(total=total_files, desc="Processing Files", unit="file", ncols=100)

        def update_progress(*args):
            progress_bar.update()

        temp_dir = tempfile.mkdtemp()
        futures, temp_dir = convert_survey_data_from_bucket(files_by_day, temp_dir, bucket_name, client, config_data,
                                                            process_func)

        for future in futures:
            future.add_done_callback(update_progress)

        client.gather(futures)  # Ensure all tasks complete

        progress_bar.close()
        os.rmdir(temp_dir)
    except KeyboardInterrupt:
        print("Closing down.")
    except CancelledError:
        print("Closing down.")
    except Exception as e:
        print(f"[bold red]An error occurred:[/bold red] {e}")
        print(f"{Traceback()}\n")


def convert_survey_data_from_bucket(files_by_day, temp_dir, bucket_name, dask_client, config_data, process_func):
    """Process survey data from S3."""
    tasks = []

    for day, files in files_by_day.items():
        for file in files:
            task = delayed(_process_raw_file)(file, day, temp_dir, bucket_name, config_data, process_func)
            tasks.append(task)

    # Execute all tasks in parallel
    futures = dask_client.compute(tasks)

    return futures, temp_dir


def list_raw_files_from_bucket(bucket_name, prefix):
    """List files in the S3 bucket along with their metadata."""
    s3_client = boto3.client('s3', config=Config(signature_version=botocore.UNSIGNED))
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    files = [
        {
            'Key': item['Key'],
            'Size': item['Size'],
            'LastModified': item['LastModified']
        }
        for item in response.get('Contents', []) if item['Key'].endswith('.raw')
    ]

    return files


def download_file_from_bucket(bucket_name, s3_key, local_dir):
    """Download a file from S3."""
    s3_client = boto3.client('s3', config=Config(signature_version=botocore.UNSIGNED))
    local_path = Path(local_dir) / os.path.basename(s3_key)
    s3_client.download_file(bucket_name, s3_key, str(local_path))

    return local_path


def _process_raw_file(file, day, temp_dir, bucket_name, config_data, process_func):
    """Process a single file."""
    day_dir = os.path.join(temp_dir, day)
    os.makedirs(day_dir, exist_ok=True)
    local_path = download_file_from_bucket(bucket_name, file['Key'], day_dir)

    process_func(local_path, config=config_data, base_path=temp_dir)
    os.remove(local_path)

