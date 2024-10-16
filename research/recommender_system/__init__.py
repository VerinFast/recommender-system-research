#
# This file is part of Recommender System Research (https://github.com/VerinFast/recommender-system-research).
# Copyright (c) 2024  Cole Golding
#
# Recommender System Research is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Recommender System Research is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

###----------------------------------------------------###
"""Sets defaults used throughout (recommender_system/*)"""
###----------------------------------------------------###

import random

import numpy as np

from . import controls as const

random.seed(const.SEED)
np.random.seed(const.SEED)

np.set_printoptions(threshold=20, edgeitems=10, formatter={"float": lambda x: ("" if int(x) < 0 or int(x) >= 10 else " ") + f'{int(x)}' if x == x else f" {const.BLANK_REP}"})
