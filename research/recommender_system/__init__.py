"""Sets defaults used throughout (recommender_system/*)"""

import random
from types import NoneType

import numpy as np

from . import controls as const

random.seed(const.SEED)
np.random.seed(const.SEED)

np.set_printoptions(threshold = 10, edgeitems = 5, formatter = {"object": lambda x: f" {const.BLANK_REP}" if isinstance(x, NoneType) else f'{x:2}'}, sign = " ")