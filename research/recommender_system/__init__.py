"""Sets defaults used throughout (recommender_system/*)"""

import random

import numpy as np

from . import controls as const

random.seed(const.SEED)
np.random.seed(const.SEED)

np.set_printoptions(threshold = 20, edgeitems = 10, formatter = {"float": lambda x: ("" if int(x) < 0 or int(x) >= 10 else " ") + f'{int(x)}' if x == x else f" {const.BLANK_REP}"})