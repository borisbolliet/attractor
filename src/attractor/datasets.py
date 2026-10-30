"""Readers of the data: one chemostat run from its CSV file, and the lynx-hare table
shipped inside the package."""
from importlib.resources import files

import numpy as np


def load_run(path):
    """Time (days) and algae density (10^6 cells/ml) of one run, missing samples dropped."""
    d = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
    t, algae = d[:, 0], d[:, 1]
    keep = ~np.isnan(algae)
    return t[keep], algae[keep]


def lynx_hare():
    """Years 1900-1920, lynx and hare pelts (thousands), from the package."""
    text = files("attractor").joinpath("data/lynx_hare.csv").read_text()
    d = np.genfromtxt(text.splitlines(), delimiter=",", skip_header=3)
    return d[:, 0], d[:, 1], d[:, 2]
