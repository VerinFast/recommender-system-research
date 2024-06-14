"""Sets defaults used throughout (recommender_system/*)"""

import random

import numpy as np

from . import controls as const

random.seed(const.SEED)
np.random.seed(const.SEED)

np.set_printoptions(threshold = 10, edgeitems = 5, formatter = {"float": lambda x: ("" if x < 0 else " ") + f'{int(x)}' if x == x else f" {const.BLANK_REP}"})