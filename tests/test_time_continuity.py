import os
import subprocess
import echopype as ep
import numpy as np
import pytest
import xarray as xr

from oceanstream.L0_unprocessed_data.ensure_time_continuity import check_reversed_time, fix_time_reversions
from pathlib import Path


current_directory = os.path.dirname(os.path.abspath(__file__))
FTP_MAIN = "ftp.bas.ac.uk"
FTP_PARTIAL_PATH = "rapidkrill/ek60/"
BASE_FOLDER = Path(__file__).parent.parent.absolute()
TEST_DATA_FOLDER = BASE_FOLDER / "test_data"
FILE_NAME = "JR230-D20091215-T121917.raw"


def _setup_file(file_name):
    test_data_path = os.path.join(TEST_DATA_FOLDER, file_name)
    if not os.path.exists(TEST_DATA_FOLDER):
        os.mkdir(TEST_DATA_FOLDER)
    if not os.path.exists(test_data_path):
        ftp_file_path = FTP_MAIN + FTP_PARTIAL_PATH + file_name
        subprocess.run(["wget", ftp_file_path, "-o", test_data_path])
    ed = ep.open_raw(test_data_path, sonar_model="EK60")
    return ed


@pytest.mark.parametrize(
    "file_name, dimension, time_name",
    [(FILE_NAME, "Sonar/Beam_group1", "ping_time")]
)
def test_check_reverse_time(file_name: str, dimension: str, time_name: str):
    dataset = _setup_file(file_name)
    has_reverse = check_reversed_time(dataset, dimension, time_name)
    assert not has_reverse
    dataset[dimension].coords[time_name].values[51] = "2009-12-15T12:20:55.3130629021"
    has_reverse_bad = check_reversed_time(dataset, dimension, time_name)
    assert has_reverse_bad


@pytest.mark.parametrize(
    "file_name, dimension, time_name",
    [(FILE_NAME, "Sonar/Beam_group1", "ping_time")]
)
def test_fix_reversal(file_name: str, dimension: str, time_name: str):
    dataset = _setup_file(file_name)
    dataset[dimension].coords[time_name].values[51] = "2009-12-15T12:20:55.3130629021"
    fixed_dataset = fix_time_reversions(dataset, dimension, time_name)
    has_reverse_bad = check_reversed_time(dataset, dimension, time_name)
    has_reverse_fixed = check_reversed_time(fixed_dataset, dimension, time_name)
    assert has_reverse_bad
    assert not has_reverse_fixed








"""
def test_check_reverse_time(dataset=dataset_jr230(), dimension="Sonar/Beam_group1", time_name="ping_time"):
    has_reversed = check_reversed_time(dataset, dimension, time_name)
    assert has_reversed is False
"""


def synthetic_dataset(date_time) -> xr.DataArray:
    da = xr.DataArray(
        np.random.rand(len(date_time)),
        coords=[np.array(date_time)],
        dims="ping_time"
    )
    sd = da
    return sd























