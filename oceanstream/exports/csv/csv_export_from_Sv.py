import json
import os
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
from haversine import haversine
from scipy.signal import savgol_filter


def ramer_douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points

    def get_perpendicular_distance(point, line_start, line_end):
        if np.allclose(line_start, line_end):
            return np.linalg.norm(point - line_start)

        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        t = np.clip(t, 0, 1)
        nearest = line_start + t * line_vec
        return np.linalg.norm(point - nearest)

    max_distance = 0
    index = 0
    for i in range(1, len(points) - 1):
        distance = get_perpendicular_distance(points[i], points[0], points[-1])
        if distance > max_distance:
            index = i
            max_distance = distance

    if max_distance > epsilon:
        left_points = ramer_douglas_peucker(points[:index + 1], epsilon)
        right_points = ramer_douglas_peucker(points[index:], epsilon)
        return np.vstack((left_points[:-1], right_points))

    return np.vstack((points[0], points[-1]))


def create_location(data: xr.Dataset, epsilon=0.00001, min_distance=0.01) -> pd.DataFrame:
    """
    Given a processed Sv file, enriched with lat/lon, it returns location data:
        lat, lon, time, speed

    Parameters:
    - data: xr.Dataset
        The raw data to extract information from.
    - epsilon: float
        The epsilon parameter for the Ramer-Douglas-Peucker algorithm.
    - min_distance: float
        The minimum distance between points after thinning (in nautical miles).

    Returns:
    - pd.DataFrame
        The required metadata.
    """
    df = data.drop_vars(
        [v for v in data.variables if v not in ["latitude", "longitude"]]
    ).to_dataframe()
    df["dt"] = data.coords["ping_time"]
    df.columns = ["lat", "lon", "dt"]

    # Filter out rows with null lat/lon values and invalid lat/lon ranges
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90) & (df["lon"] >= -180) & (df["lon"] <= 180)]

    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "dt", "knt"])

    # Apply smoothing to lat/lon to reduce noise
    window_size = min(11, len(df))  # Window size for smoothing filter
    poly_order = 2  # Polynomial order for smoothing filter

    if len(df) > window_size:
        df["lat"] = savgol_filter(df["lat"], window_size, poly_order)
        df["lon"] = savgol_filter(df["lon"], window_size, poly_order)

    df["distance"] = [
        haversine(
            (df["lat"].iloc[i], df["lon"].iloc[i]),
            (df["lat"].iloc[i - 1], df["lon"].iloc[i - 1]),
            unit="nmi",
        )
        if i > 0
        else 0
        for i in range(len(df))
    ]
    df["time_interval"] = df["dt"] - df["dt"].shift()
    df["knt"] = (df["distance"] / df["time_interval"].dt.total_seconds()) * 3600
    df = df[["lat", "lon", "dt", "knt"]]

    # Additional check for outliers based on unrealistic speed
    df = df[df["knt"] < 100]  # Removing points with speeds higher than 100 knots

    # Apply Ramer-Douglas-Peucker algorithm for thinning coordinates
    points = df[["lat", "lon"]].values

    thinned_points = ramer_douglas_peucker(points, epsilon)

    # Create a thinned DataFrame
    thinned_df = pd.DataFrame(thinned_points, columns=["lat", "lon"])

    # Ensure the datetime and speed values are correctly associated
    thinned_df["dt"] = thinned_df.apply(
        lambda row: df.loc[(df["lat"] == row["lat"]) & (df["lon"] == row["lon"]), "dt"].values[0], axis=1)
    thinned_df["knt"] = thinned_df.apply(
        lambda row: df.loc[(df["lat"] == row["lat"]) & (df["lon"] == row["lon"]), "knt"].values[0], axis=1)

    # Further thin by minimum distance
    final_points = [thinned_df.iloc[0]]
    for i in range(1, len(thinned_df)):
        if haversine((final_points[-1]["lat"], final_points[-1]["lon"]),
                     (thinned_df.iloc[i]["lat"], thinned_df.iloc[i]["lon"]), unit="nmi") >= min_distance:
            final_points.append(thinned_df.iloc[i])

    final_df = pd.DataFrame(final_points)

    return final_df


def create_Sv(data: xr.Dataset, channel: str) -> pd.DataFrame:
    """
    Given a processed Sv file, enriched with lat/lon, it returns Sv data

    Parameters:
    - data: xr.Dataset
        The raw data to extract information from.
    - channel: str
        The channel to use

    Returns:
    - pd.DataFrame
        The required data.
    """
    data = data.copy(deep=True)
    data["ping_time"] = range(0, len(data.ping_time))
    data["range_sample"] = data["range_sample"] / 2
    df = data.sel(channel=channel)["Sv"].to_dataframe()
    df = df["Sv"].unstack(level="ping_time")
    return df


def export_Sv_csv(data: xr.Dataset, folder: str, root_name: str):
    """
    Given a Sv file, a folder to write the outputs into, and a name pattern for the files,
    it extracts and exports to CSV the location data and the Sv data

    Parameters:
    - data: xr.Dataset
        The Sv to extract this information from.
    - folder: str
        The folder name to use.
    - root_name: str
        The root name to be used in the file patterns.

    Returns:
    - None
    """
    location = create_location(data)
    Sv = create_Sv(data, data["channel"][1])
    location_filename = os.path.join(folder, root_name + "_GPS.csv")
    Sv_filename = os.path.join(folder, root_name + "_Sv_38000.0.csv")
    try:
        location.to_csv(location_filename, index=False)
        Sv.to_csv(Sv_filename)
    except Exception as e:
        raise ValueError(str(e))


def export_location_json(data: xr.Dataset):
    location = create_location(data)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        location.to_json(temp_file.name, orient="records")
        temp_file_path = temp_file.name
    with open(temp_file_path, 'r') as file:
        gps_data = json.load(file)
    os.remove(temp_file_path)
    return gps_data
