import rasterio
import numpy as np

"""
Utility functions for loading raster data used in fire spread modeling.
"""

def load_raster(name):
    """
    Loads a cropped raster and optionally sub-windows it.

    Parameters
    ----------
    name : str
        Base name of the raster (without “_cropped.tif”).
    x_interval : tuple of int (start, end), optional
        Column indices to keep (0-based, [start:end]).
    y_interval : tuple of int (start, end), optional
        Row indices to keep (0-based, [start:end]).

    Returns
    -------
    np.ndarray: A 2D array of the raster data (float32) with shape (rows, cols).
    """

    path = name
    y_interval = None
    x_interval = None
    with rasterio.open(path) as src:
        data = src.read(1)  # full 2D array, shape (rows, cols)

        rows, cols = data.shape
        y0, y1 = y_interval if y_interval is not None else (0, rows)
        x0, x1 = x_interval if x_interval is not None else (0, cols)
        data = data[y0:y1, x0:x1]

    # Flip so row 0 becomes bottom
    data = data
    return np.ascontiguousarray(data).astype(np.float32)


def normalize(data, datatype=None):
    """
    Converts input data into a 3D cube format for simulations.

    Parameters
    ----------
    data : np.ndarray
        The input data to be converted.
    time_steps : int
        The number of time steps for the cube.
    datatype : str, optional
        Specifies the type of data for normalization.

    Returns
    -------
    np.ndarray: A 3D cube of the input data.
    """

    # Normalize data based on datatype
    if datatype == "cbd":
        data = data / 100.0
    elif datatype == "cc":
        data = data / 100.0
    elif datatype == "fbfm":
        data = data.astype(np.int32)
    if datatype == "slp":
        data = np.tan(np.pi / 180 * data)
    if datatype == "ch":
        data = data / 10.0
    if datatype == "cbh":
        data = data / 10.0
    data = np.ascontiguousarray(data)
    return data


def load_all_rasters(filename, index, raster_dir = "cropped_raster", dim = 50):
    """
    Load all rasters for a given x, y coordinate and filename.

    Parameters
    ----------
    x : int
        X coordinate (column index).
    y : int
        Y coordinate (row index).
    filename : str
        Base name of the raster files.
    index : int
        Index of the raster tile.
    raster_dir : str, optional
        Directory where the raster files are stored.
    dim : int, optional
        Dimension of the raster tiles (default is 50).

    Returns
    -------
    dict: A dictionary with loaded rasters.
    """
    
    rasters = {}
    for suffix in ["slp", "asp", "dem", "cc", "cbd", "cbh", "ch", "fbfm"]:
        name = f"{filename}/{suffix}/{filename}_{index}_{suffix}.tif"
        rasters[suffix] = normalize(load_raster(name), suffix)
    return rasters
