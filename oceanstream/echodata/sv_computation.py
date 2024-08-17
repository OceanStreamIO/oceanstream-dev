"""
sv_computation.py
-----------------

Module for computing the volume backscattering strength (Sv) from raw data.
Supported Sonar Models:
- EK60
- AZFP
- EK80
Functions and Classes:
- `SupportedSonarModelsForSv`: An Enum containing the sonar models supported
for Sv computation.
- `WaveformMode`: Enum specifying the waveform mode ("CW" or "BB").
- `EncodeMode`: Enum indicating the encoding mode ("complex" or "power").
- `ComputeSVParams`: Class to validate and structure the parameters passed
to the Sv computation function.
- `compute_sv`: Main function to calculate Sv given an EchoData object
and other optional parameters. This function is based on the `echopype.calibrate.compute_Sv()` function.
- `compute_sv_with_encode_mode`

Usage:

To compute Sv for a given EchoData object, `ed`, simply call:
`compute_sv(ed)`
"""
import echopype as ep
import xarray as xr

from enum import Enum
from typing import Any, Optional
from echopype.echodata.echodata import EchoData
from pydantic import BaseModel, ValidationError, field_validator


class SupportedSonarModelsForSv(str, Enum):
    EK60 = "EK60"
    AZFP = "AZFP"
    EK80 = "EK80"


class WaveformMode(str, Enum):
    CW = "CW"
    BB = "BB"


class EncodeMode(str, Enum):
    COMPLEX = "complex"
    POWER = "power"


class ComputeSVParams(BaseModel):
    echodata: Any
    env_params: Optional[dict] = None
    cal_params: Optional[dict] = None
    waveform_mode: Optional[WaveformMode] = None
    encode_mode: Optional[EncodeMode] = None

    @field_validator("echodata")
    def check_echodata_type(cls, value):
        if not isinstance(value, EchoData):
            raise ValueError("Invalid type for echodata. Expected an instance of EchoData.")
        return value


def compute_sv(echodata: EchoData, **kwargs) -> xr.Dataset:
    """
    Computes the volume backscattering strength (Sv) from the given echodata.

    Parameters:
    - echodata (EchoData): The EchoData object containing sonar data for computation.
    - **kwargs: Additional keyword arguments passed to the Sv computation.

    Returns:
    - xr.Dataset: A Dataset containing the computed Sv values.

    Example:
    >>> sv_results = compute_sv(echodata)
    >>> print(sv_results)

    Notes:
    This function:
    - Validates the `echodata`'s sonar model against supported models.
    - Uses the `ComputeSVParams` pydantic model to validate parameters.
    - Checks if the computed Sv is empty.
    - Returns Sv only if it is not empty.
    - Is based on the `echopype.calibrate.compute_Sv()` function.

    """
    # Validate parameters using the pydantic model
    try:
        ComputeSVParams(echodata=echodata, **kwargs)
    except ValidationError as e:
        raise ValueError(str(e))
    # Check if the sonar model is supported
    sonar_model = echodata.sonar_model

    try:
        SupportedSonarModelsForSv(sonar_model)
    except ValueError:
        raise ValueError(
            f"Sonar model '{sonar_model}'\
                          is not supported for Sv computation.\
                          Supported models are \
                            {list(SupportedSonarModelsForSv)}."
        )
    # Compute Sv
    Sv = ep.calibrate.compute_Sv(echodata, **kwargs)
    # Check if the computed Sv is empty
    if Sv["Sv"].values.size == 0:
        raise ValueError("Computed Sv is empty!")
    return Sv


def compute_sv_with_encode_mode(
    echodata: EchoData, waveform_mode: str = "CW", encode_mode: str = "power"
) -> xr.Dataset:
    try:
        sv_dataset = compute_sv(echodata, waveform_mode=waveform_mode, encode_mode=encode_mode)
    except Exception as e:
        print(f"\n--------Error computing Sv with encode mode: {e}--------\n")
        import traceback
        traceback.print_exc()
        raise e

    return sv_dataset
