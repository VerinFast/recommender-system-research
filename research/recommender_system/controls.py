###--------------------------------------------------------------------------###
"""A list of constant variables that can be used to manipulate the experiment"""
###--------------------------------------------------------------------------###

MATRIX_SIZE: int = 50
"""The size (# of customers / goods) used to create a (n x n) matrix for the experiment"""


###------------------------------------------------------------------------------###
"""A list of constant variables used in (research/recommender_system/__init__.py)"""
###------------------------------------------------------------------------------###

BLANK_REP: str = "#"
"""The character used to represent an empty (NoneType object) space in matrix printouts"""

SEED: int = 10
"""A seed to generate random numbers from for test reproducibility"""


###------------------------------------------------------###
"""A list of constant variables used for testing purposes"""
###------------------------------------------------------###

NEG_PERCENT: float = 0.1
"""Percent (weighted chance) of randomly generated user reviews containing a negative review (-1)"""

ZERO_PERCENT: float = 0.2
"""Percent (weighted chance) of randomly generated user reviews containing a neutral / no review (0)"""

POS_PERCENT: float = 0.1
"""Percent (weighted chance) of randomly generated user reviews containing a positive review (1)"""

NONE_PERCENT: float = 0.6
"""Percent (weighted chance) of randomly generated user reviews that have not been used (None)"""
