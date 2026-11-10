"""Readers of the data: one chemostat run from its CSV file, and the lynx-hare table
shipped inside the package."""
from importlib.resources import files

import numpy as np


def load_run(path):
    """Read the time and algae density of a chemostat run from a CSV file.

    Parameters
    ----------
    path : str or path-like
        CSV file with one header row, time in the first column and algae
        density in the second. Further columns are ignored.

    Returns
    -------
    t : numpy.ndarray of float32
        Sample times in days.
    algae : numpy.ndarray of float32
        Algae density in millions of cells per millilitre, at the retained
        sample times. Rows with NaN algae values are removed from both arrays.
    """
    d = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
    t, algae = d[:, 0], d[:, 1]
    keep = ~np.isnan(algae)
    return t[keep], algae[keep]


def lynx_hare():
    """Read the lynx and hare pelt records distributed with the package.

    Returns
    -------
    year : numpy.ndarray
        Years 1900 to 1920, inclusive.
    lynx : numpy.ndarray
        Recorded lynx pelts in thousands, one value per year.
    hare : numpy.ndarray
        Recorded hare pelts in thousands, one value per year.
    """
    text = files("attractor").joinpath("data/lynx_hare.csv").read_text()
    d = np.genfromtxt(text.splitlines(), delimiter=",", skip_header=3)
    return d[:, 0], d[:, 1], d[:, 2]
