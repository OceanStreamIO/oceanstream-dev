import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
import pandas as pd
from IPython.display import display, HTML
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List


split_by_day_pattern = r'D(\d{8})-T(\d{6})'


def extract_timestamp_from_filename(file_name):
    """Extract creation time from the file name if it follows a specific pattern."""
    pattern = r'(\d{4}[A-Z])?-D(\d{8})-T(\d{6})'
    match = re.search(pattern, file_name)
    if match:
        date_str = match.group(2)
        time_str = match.group(3)
        creation_time = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        return creation_time

    return None


def extract_timestamp_str(filename):
    """Extract timestamp from the filename."""
    match = re.search(r'D(\d{8})-T(\d{6})', filename)
    if match:
        date_str, time_str = match.groups()
        return f"{date_str}T{time_str}"
    return None


def parse_filename(filename):
    """Parse the date and time from the filename."""
    match = re.search(r'D(\d{8})-T(\d{6})', filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
    return None


def sort_raw_files(raw_files):
    return sorted(raw_files, key=lambda x: extract_timestamp_str(Path(x).stem))


def group_files_by_day(files, extract_filename_from_object=lambda x: x['Key']):
    """Group files by day based on the parsed date from the filename."""
    files_by_day = {}
    for file in files:
        date_str, _ = re.search(split_by_day_pattern, extract_filename_from_object(file)).groups()
        day = date_str
        if day not in files_by_day:
            files_by_day[day] = []
        files_by_day[day].append(file)

    for day in files_by_day:
        files_by_day[day] = sorted(files_by_day[day], key=lambda x: parse_filename(x['Key']))

    return files_by_day


def _print_survey_summary(files_by_day):
    """Print summary of files by day."""
    print("Summary of files by day:")
    for day, files in files_by_day.items():
        print(f"Day: {day}, Number of files: {len(files)}")
    print(f"Total number of days: {len(files_by_day)}")


def print_survey_summary(files_by_day, show_in_terminal=True, show_in_notebook=False, show_file_size=False):
    """Print summary of files by day in terminal and Jupyter notebook."""
    # Calculate statistics
    statistics = calculate_survey_statistics(files_by_day)

    if show_in_terminal:
        # Create a summary using rich for terminal display
        console = Console()
        table = Table(title="Survey Summary", box=box.ROUNDED, show_header=True, header_style="bold magenta")

        # Add columns to the table
        table.add_column("Day", justify="center")
        table.add_column("Number of Files", justify="center")
        table.add_column("Total Recorded Time (HH:MM:SS)", justify="center")
        if show_file_size:
            table.add_column("Average File Size (MB)", justify="center")

        # Add rows to the table
        for day, stats in statistics.items():
            row = [
                datetime.strptime(day, "%Y%m%d").strftime("%b %d, %Y"),
                str(stats["Number of Files"]),
                stats["Total Recorded Time"]
            ]
            if show_file_size:
                row.append(str(stats["Average File Size (MB)"]))
            table.add_row(*row)

        # Display the table in the terminal
        console.print(table)

    if show_in_notebook:
        # Create a summary using pandas for Jupyter notebook display
        df = pd.DataFrame.from_dict(statistics, orient='index')
        df.index = pd.to_datetime(df.index, format='%Y%m%d').strftime('%b %d, %Y')

        if not show_file_size:
            df = df.drop(columns=["Average File Size (MB)"])

        display(HTML("<h2>Survey Summary</h2>"))
        display(df.style.set_table_styles(
            [{
                'selector': 'thead th',
                'props': [('background-color', 'magenta'), ('color', 'white'), ('font-weight', 'bold')]
            }, {
                'selector': 'tbody td',
                'props': [('border', '1px solid black')]
            }, {
                'selector': 'table',
                'props': [('border-collapse', 'collapse'), ('width', '100%')]
            }]
        ).set_properties(**{'text-align': 'center'}))


def calculate_survey_statistics(files_by_day):
    """Calculate the total recorded time and average file size per day."""
    statistics = {}
    for day, files in files_by_day.items():
        total_time = timedelta()
        previous_time = None
        total_size = sum(file['Size'] for file in files)
        avg_size = total_size / len(files) if files else 0

        for file in files:
            current_time = parse_filename(file['Key'])
            if previous_time:
                time_diff = current_time - previous_time
                if time_diff > timedelta(minutes=10):  # More than 10 minutes indicates a stop
                    total_time += timedelta(minutes=10)
                else:
                    total_time += time_diff
            previous_time = current_time

        statistics[day] = {
            "Number of Files": len(files),
            "Total Recorded Time": format_duration(total_time),
            "Average File Size (MB)": round(avg_size / (1024 * 1024), 2)
        }
    return statistics


def format_duration(duration):
    """Format a timedelta duration as HH:MM:SS."""
    total_seconds = int(duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def append_gps_data(gps_data: List[Dict[str, Any]], gps_json_file_path: Path, filename: str):
    """
    Append GPS data to a JSON file.

    Args:
        gps_data (List[Dict[str, Any]]): The GPS data to append.
        gps_json_file_path (Path): The path to the JSON file where the GPS data will be saved.
        filename (str): The filename to associate with the GPS data.
    """
    gps_entry = {
        "filename": filename,
        "gps_data": gps_data
    }

    # Check if the file already exists
    if gps_json_file_path.exists():
        with open(gps_json_file_path, 'r+') as json_file:
            try:
                # Load existing data
                existing_data = json.load(json_file)
            except json.JSONDecodeError:
                existing_data = []

            # Ensure existing data is a list
            if not isinstance(existing_data, list):
                existing_data = [existing_data]

            # Append new data
            existing_data.append(gps_entry)

            # Move file pointer to the beginning
            json_file.seek(0)
            json_file.truncate()

            # Write updated data
            json.dump(existing_data, json_file, indent=4)
    else:
        # Create a new file with the GPS data
        with open(gps_json_file_path, 'w') as json_file:
            json.dump([gps_entry], json_file, indent=4)


def save_output_data(output_message: dict, output_data_path: Path):
    """
    Save the output message data to a JSON file.

    Args:
        output_message (dict): The output message data.
        output_data_path (Path): The path to the JSON file where the output data will be saved.
    """
    # Check if the JSON file already exists and read its content
    if output_data_path.exists():
        with open(output_data_path, "r") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []

    # Append the new data to the existing data
    existing_data.append(output_message)

    # Write the updated data back to the JSON file
    with open(output_data_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
