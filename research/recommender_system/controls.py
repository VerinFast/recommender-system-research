###--------------------------------------------------------------------------###
"""A list of constant variables that can be used to manipulate the experiment"""
###--------------------------------------------------------------------------###

MATRIX_SIZE: int = 50
"""The size (# of users / goods) used to create a (n x n) matrix for the experiment"""

NUMBER_OF_TICKS: int = 100
"""The amount of time / number of loops that the experiment should run for"""

UTILITY_MEAN: int = 4
"""The mean used to generate the users' utility per good"""

UTILITY_STD: int = 2
"""The standard deviation used to generate the users' utility per good"""

USER_BUDGET: int = 10
"""The set "budget" (time, money, etc) a user has to consume goods per tick"""

CONSIDERATION_COST: int = 1
"""The "cost" to the user when looking for a good to be recommended to them"""

USAGE_COST: int = 1
"""The "cost" to the user when using the good that has been recommended to them"""

COUNT_NEGATIVE_REVIEWS: bool = False
"""Whether or not to count negative reviews when finding recommendations for a user"""


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

BLANK_PERCENT: float = 0.6
"""Percent (weighted chance) of randomly generated user reviews that have not been used (np.nan)"""
