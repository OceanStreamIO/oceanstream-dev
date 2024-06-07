"""
raw_handler.py
--------------
Module for reading, verifying, and converting echo sounder raw data files.

This module provides functionalities to:

- Search for raw echo sounder files within specified directories or paths.
- Verify the integrity of these files, ensuring they are readable by echopype.
- Extract essential metadata from the files, such as campaign ID,\
date of measurement, and sonar model.
- Convert raw files to specified formats (netCDF or zarr) and\
save them to a desired location.
- Group similar files based on specific criteria, such as campaign ID,\
sonar model, and time difference.

The module supports various sonar models including
EK60, ES70, EK80, EA640, AZFP, and AD2CP.
It also defines a time threshold for determining
the similarity between two consecutive files.

"""

import os
import re
import echopype as ep
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Union, Tuple, Any
from echopype.convert.utils.ek_raw_io import RawSimradFile
from .ensure_time_continuity import check_reversed_time, fix_time_reversions

SUPPORTED_SONAR_MODELS = ["EK60", "ES70", "EK80", "EA640", "AZFP", "AD2CP"]
TIME_BETWEEN_FILES = 30  # time in minutes between two consecutive files


def read_file(config_data, use_swap='auto', skip_integrity_check=False, skip_time_reversion=False):
    filename = config_data["raw_path"]

    if not skip_integrity_check:
        check = file_integrity_checking(filename)
        if not check.get("file_integrity", False):
            return {"Processing Error": f"File {filename} could not usable!"}

        sonar_model = check.get("sonar_model", config_data["sonar_model"])
    else:
        sonar_model = config_data["sonar_model"]

    echodata = _read_file(filename, use_swap=use_swap, sonar_model=sonar_model)

    if sonar_model == "EK80":
        encode_mode = get_encode_mode(echodata)
    else:
        encode_mode = "power"

    # Fix time reversions if necessary
    if not skip_time_reversion and check_reversed_time(echodata, "Sonar/Beam_group1", "ping_time"):
        echodata = fix_time_reversions(echodata, {"Sonar/Beam_group1": "ping_time"})

    return echodata, encode_mode


def file_finder(paths: Union[str, List[str]], file_type: str = "raw") -> List[str]:  # noqa: E501
    """
    Finds and returns all files of a specified type from given paths.

    This function searches for files of a specified type (e.g., "raw")
    within the provided paths.
    It can search within a single directory or
    across multiple specified file paths.

    Parameters:

    - paths (str or list[str]): If a string is provided,\
    it should be the absolute path to a directory.\
    If a list is provided,\
    it should contain absolute paths to individual files.
    - file_type (str, optional): The type of files to search for.\
    Defaults to "raw".

    Returns:

    - list[str]: A sorted list of all found files of the specified type.

    Raises:

    - ValueError: If the provided paths input is neither a directory\
    nor a list of file paths.

    Example:

     file_finder("/path/to/directory")
    ['/path/to/directory/file1.raw', '/path/to/directory/file2.raw']

     file_finder(["/path/to/file1.raw", "/path/to/file2.raw"])
    ['/path/to/file1.raw', '/path/to/file2.raw']
    """
    if file_type == "zarr" or file_type == ".zarr":
        zarr_files = _find_zarr_root_directories(paths)
        if isinstance(paths, str):
            # Filter out subdirectory paths
            zarr_files = [f for f in zarr_files if os.path.dirname(f) == paths]
        return sorted(zarr_files)

    if isinstance(paths, str) and os.path.isdir(paths):
        ret_files = [
            os.path.join(paths, f_p)
            for f_p in os.listdir(paths)
            if os.path.isfile(os.path.join(paths, f_p))
        ]
        ret_files = file_finder(ret_files, file_type)
    elif isinstance(paths, list):
        ret_files = []
        for elem in paths:
            if "." + file_type in elem and os.path.isfile(elem):
                ret_files.append(elem)
    else:
        raise ValueError(
            "Invalid input. Provide either a directory\
             path or a list of file paths."
        )

    return sorted(ret_files)


def file_integrity_checking(
    file_path: str
) -> Dict[str, Union[str, datetime, bool]]:  # noqa: E501
    """
    Checks the integrity of a given echo sounder file.

    This function verifies if the provided echo sounder file is
    readable by echopype and extracts essential metadata such as the campaign ID, date of measurement, sonar model
    and `use_swap` option (relevant only for raw files)

    The function supports raw, netCDF, and zarr file formats.

    Parameters:
    - file_path (str): Absolute path to the echo sounder file.
    - use_swap (bool, optional): Parameter specific to the echopype library `open_raw` function. Defaults to False\
      If True, variables with a large memory footprint will be written to a temporary zarr store at \
      ``~/.echopype/temp_output/parsed2zarr_temp_files``\
      Relevant only for raw files.

    Returns:

    - dict: A dictionary containing the following keys:
        'file_path': Absolute path to the file.
        'campaign_id': Identifier for the measuring campaign,
                      extracted from the file name.
        'date': Date and time when the measurement started,
                extracted from the file name. Returns None if the date
                and time cannot be determined.
        'file_integrity': Boolean indicating if the file is of a supported type.

    Raises:
    - Exception: If the file type is not supported.

    Example:
    file_integrity_checking("/path/to/JR161-D20230509-T100645.raw")
    {
        'file_path': '/path/to/JR161-D20230509-T100645.raw',
        'campaign_id': 'JR161',
        'date': datetime.datetime(2023, 5, 9, 10, 6, 45),
        'file_integrity': True
    }
    """
    return_dict = {}

    # get file name from path
    file_path = os.path.abspath(file_path)
    _, file_name = os.path.split(file_path)
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension not in [".raw", ".nc", ".zarr"]:
        raise Exception("File type not supported for " + str(file_path))

    campaign_id, date, sonar_model, _, file_integrity = get_campaign_metadata(file_path)

    return_dict["file_path"] = file_path
    return_dict["campaign_id"] = campaign_id
    return_dict["date"] = date
    return_dict["file_integrity"] = file_integrity
    return_dict["sonar_model"] = sonar_model

    return return_dict


def get_campaign_metadata(file_path: str) -> tuple[
    str | None | Any, datetime | None | Any, str | None, bytes | None, bool
]:
    file_path = os.path.abspath(file_path)
    _, file_name = os.path.split(file_path)
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    metadata = None
    file_integrity = True
    sonar_model = None
    date = None
    campaign_id = None

    if ".raw" == file_extension:
        metadata = _parse_metadata(file_path)
        if metadata is not None:
            campaign_id = metadata.get("survey_name", None)
            date = metadata.get("timestamp", None)
            sonar_model = detect_sonar_model(file_path, metadata=metadata)

    if not metadata:
        campaign_id = file_name.split("-")[0]

        try:
            pattern_date = r"D(\d{4})(\d{2})(\d{2})"
            pattern_time = r"T(\d{2})(\d{2})(\d{2})"

            matches_date = re.findall(pattern_date, file_name)[0]
            matches_time = re.findall(pattern_time, file_name)[0]

            year, month, day = matches_date
            hour, minute, second = matches_time

            datetime_string = f"D{year}{month}{day}-T{hour}{minute}{second}"
            date = datetime.strptime(datetime_string, "D%Y%m%d-T%H%M%S")
        except Exception as e:
            e += "!"
            file_integrity = False
            date = None

    return campaign_id, date, sonar_model, metadata, file_integrity

def read_raw_files(
    file_dicts: List[Dict[str, Union[str, datetime, bool]]]
) -> List[ep.echodata.EchoData]:
    """
    Reads multiple raw echo sounder files and returns a list of Datasets.

    This function processes a list of file information dictionaries,
    opens each raw file
    using the specified sonar model,
    and returns the corresponding datasets.

    Parameters:

    - file_dicts (list of dict): List of dictionaries, \
    each containing file information \
    as provided by the file_integrity_checking function.

    Returns:

    - list: List of EchoData datasets corresponding to each raw file.

    """
    ret_list = []
    for f_i in file_dicts:
        opened_file = _read_file(file_path=f_i["file_path"])
        ret_list.append(opened_file)
    return ret_list


def convert_raw_files(
        file_dicts: List[Dict[str, Union[str, datetime, bool]]],
        save_path: str = "",
        save_file_type: str = "nc",
) -> List[str]:
    """
    Converts multiple raw echo sounder files to the
    specified file type and saves them.

    This function processes a list of file information dictionaries,
    converts each raw file
    to the specified file type (netCDF or zarr),
    and saves the converted files to the given path.

    Parameters:

    - file_dicts (list of dict): List of dictionaries,\
    each containing file information.
    - save_path (str): Directory path where\
    the converted files will be saved.
    - save_file_type (str): Desired file type\
    for saving the converted files.\
    Options are 'nc' or 'zarr'.

    Returns:

    - list: List of paths to the saved converted files.

    """
    ret_list = []
    for f_i in file_dicts:
        opened_file = _read_file(file_path=f_i["file_path"])
        _write_file(opened_file, save_path, save_file_type)
        file_name = os.path.split(f_i["file_path"])[-1]
        file_type = save_file_type
        new_file_name = file_name.replace("raw", file_type)
        ret_list.append(os.path.join(save_path, new_file_name))
    return ret_list


def split_files(
        file_dicts: List[Dict[str, Union[str, datetime, bool]]]
) -> List[List[Dict[str, Union[str, datetime, bool]]]]:
    """
    Splits a list of file information dictionaries into sublists based on their similarity.

    This function processes a list of file information dictionaries and groups them into
    sublists where each sublist contains files that are similar to each other based on
    specific criteria.

    Parameters:
    - file_dicts (list of dict): List of file information dictionaries.

    Returns:
    - list of lists: List containing sublists of file dictionaries\
    grouped by their similarity.

    """
    list_of_lists = []

    temp_list = []
    prev_elem = file_dicts[0]
    for elem in file_dicts:
        if _is_similar(elem, prev_elem):
            temp_list.append(elem)
        else:
            list_of_lists.append(temp_list)
            temp_list = [elem]
        prev_elem = elem
    list_of_lists.append(temp_list)
    return list_of_lists


def concatenate_files(
        file_dicts: List[Dict[str, Union[str, datetime, bool]]]
) -> ep.echodata.EchoData:
    list_of_datasets = []
    for file_info in file_dicts:
        list_of_datasets.append(_read_file(file_info["file_path"]))
    combined_dataset = ep.combine_echodata(list_of_datasets)
    return combined_dataset


def get_encode_mode(echo_data) -> str:
    """
    For EK80:
    If only complex data (can be BB or CW signals) exist,
    there exists only Beam_group1 and this group may
    contain CW or BB complex data, or a mixture of both. See example below.

    If only power/angle data (only valid for CW signals) exist,
    there exists only Beam_group1 and this group contains CW power and angle data.
    The structure is almost identical with EK60 data above.

    If both complex and power/angle data exist,
    there exist Beam_group1 (containing complex data)
    and Beam_group2 (containing power/angle data).
    """
    beam_group1, beam_group2 = _check_beam_groups(echo_data)
    encode_mode = "power"
    if beam_group1:
        imaginary_part = "backscatter_i" in echo_data.data_vars
        if not imaginary_part:
            return "power"
        else:
            return "complex"
    if beam_group2:
        return "complex"
    return encode_mode


def _parse_metadata(file_path):
    try:
        with RawSimradFile(file_path, "r", storage_options={}) as fid:
            config_datagram = fid.read(1)
            return config_datagram
    except Exception as e:
        print(f"Error parsing metadata from {file_path}. Error: {e}")
        return None


def _find_zarr_root_directories(paths: Union[str, List[str]]) -> List[str]:
    """
    Finds and returns paths to the root directories of zarr datasets within the given paths.

    Parameters:
    - paths (str or List[str]): A single directory path or a list of directory paths.

    Returns:
    - List[str]: A list of paths to the root directories of found zarr datasets.

    Raises:
    - ValueError: If the provided paths input is not valid.
    """
    zarr_roots = []

    def is_zarr_root(directory: str) -> bool:
        """
        Checks if a directory is the root of a zarr dataset.
        """
        return any(
            fname.endswith(".zarray") or fname.endswith(".zgroup")
            for fname in os.listdir(directory)
        )

    if isinstance(paths, str):
        if not os.path.isdir(paths):
            raise ValueError(f"Path {paths} is not a valid directory.")
        search_paths = [paths]
    elif isinstance(paths, list):
        search_paths = paths
    else:
        raise ValueError(
            "Invalid input. Provide either a directory path or a list of directory paths."
        )

    for path in search_paths:
        for root, dirs, _ in os.walk(path):
            if is_zarr_root(root):
                zarr_roots.append(root)
                dirs[:] = []  # Skip subdirectories to avoid nested zarr datasets

    return sorted(zarr_roots)


def _read_file(file_path: str, sonar_model: str = None, use_swap: Union[bool, str] = 'auto',
               max_chunk_size: str = None) -> ep.echodata.EchoData:
    """
    Reads an echo sounder file and returns the corresponding Dataset.

    This function determines the type of the file (raw, netCDF, or zarr) based on its extension and opens it using
    echopype. The sonar_model and use_swap parameters are relevant only for the raw files.

    Parameters:

    - file_path (str): Absolute path to the echo sounder file.
    - sonar_model (str, optional): Type of sonar model. Defaults to "EK60".\
      Relevant only for raw files.
    - use_swap (bool, optional): Parameter specific to the echopype library `open_raw` function. Defaults to False\
      If True, variables with a large memory footprint will be written to a temporary zarr store at \
      ``~/.echopype/temp_output/parsed2zarr_temp_files``\
      Relevant only for raw files.

    Returns:
    - EchoData: Dataset corresponding to the provided file.

    Raises:
    - Exception: If the file type is not supported by echopype.

    """
    file_name = os.path.split(file_path)[-1]
    if ".raw" in file_name:
        if sonar_model is None:
            sonar_model = detect_sonar_model(file_path)

        ed = ep.open_raw(file_path, sonar_model=sonar_model, use_swap=use_swap, max_chunk_size=max_chunk_size)  # type: ignore
    elif ".nc" in file_name or ".zarr" in file_name:
        ed = ep.open_converted(file_path)  # create an EchoData object
    else:
        raise Exception("File not supported by echopype.")
    return ed


def _write_file(
        ed: ep.echodata.EchoData,
        save_path: str,
        save_file_type: str = "nc",
        overwrite: bool = False,  # noqa: E501
) -> str:
    """
    Writes an echo sounder dataset to a specified file type and saves it.

    This function takes an EchoData dataset, converts it to the specified file type
    (netCDF or zarr), and saves the file to the provided path.

    Parameters:

    - ed (EchoData): echo sounder dataset to be saved.
    - save_path (str): Directory path where the dataset will be saved.
    - save_file_type (str, optional): Desired file type\
    for saving the dataset. Defaults to 'nc'.\
      Options are 'nc' or 'zarr'.
    - overwrite (bool, optional): If True, overwrites\
    the file if it already exists. Defaults to False.

    Returns:

    - str: Path to the saved file.

    Raises:

    - Exception: If the specified file type is not supported by echopype.

    """
    if save_file_type == "nc":
        ed.to_netcdf(save_path=save_path, overwrite=overwrite)
    elif save_file_type == "zarr":
        ed.to_zarr(save_path=save_path, overwrite=overwrite)
    else:
        raise Exception("File type not supported echopype.")
    return save_path


def _is_similar(
        file_dict1: Dict[str, Union[str, datetime, bool]],
        file_dict2: Dict[str, Union[str, datetime, bool]],
) -> bool:
    """
    Determines if two file information dictionaries are similar based on specific criteria.

    This function checks if two file dictionaries have the same campaign ID, sonar model,
    file integrity, and if their date difference is within a specified time range.

    Parameters:

    - file_dict1 (dict): First file information dictionary.
    - file_dict2 (dict): Second file information dictionary.

    Returns:

    - bool: True if the file dictionaries are similar\
    based on the criteria, False otherwise.

    """
    if file_dict1["campaign_id"] != file_dict2["campaign_id"]:
        return False
    if file_dict1["sonar_model"] != file_dict2["sonar_model"]:
        return False
    if file_dict1["file_integrity"] != file_dict2["file_integrity"]:
        return False
    date_diff = file_dict1["date"] - file_dict2["date"]
    if date_diff > timedelta(minutes=TIME_BETWEEN_FILES):
        return False
    return True


def _check_beam_groups(echo_data):
    # Check if the 'Sonar' group exists
    if "sonar" in echo_data.group_map:
        sonar_group = echo_data.group_map["sonar"]
        # Checking for Beam_group1 and Beam_group2
        has_beam_group1 = "Beam_group1" in sonar_group
        has_beam_group2 = "Beam_group2" in sonar_group

        return has_beam_group1, has_beam_group2
    else:
        # The 'Sonar' group does not exist in group_map
        return False, False


def detect_sonar_model(file_path: str, metadata=None) -> str:
    if metadata is None:
        metadata = _parse_metadata(file_path)

    if metadata is None:
        return None

    if "sounder_name" not in metadata:
        try:
            xml_string = metadata.get("xml", None)
            root = ET.fromstring(xml_string)
            header = root.find("Header")
            application_name = header.attrib.get("ApplicationName")

            if application_name == "EK80":
                return "EK80"

        except ET.ParseError:
            return None

        return None

    if metadata["sounder_name"] == "EK60" or metadata["sounder_name"] == "ER60":
        return "EK60"

    return metadata["sounder_name"]
