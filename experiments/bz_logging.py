"""Log messages of the integrators, shown by configuring the logging module.

Run from the project root: python experiments/bz_logging.py
"""
import logging

import numpy as np

from attractor import RK4, Adaptive, Oregonator

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
np.seterr(over="ignore", invalid="ignore")   # the log reports the failure

RK4(0.01).run(Oregonator(), [1.0, 2.0, 3.0], 0.0, 30.0)
Adaptive("LSODA").run(Oregonator(), [1.0, 2.0, 3.0], 0.0, 30.0)
