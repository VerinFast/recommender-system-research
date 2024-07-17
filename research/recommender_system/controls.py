###--------------------------------------------------------------------------###
"""A list of constant variables that can be used to manipulate the experiment"""
###--------------------------------------------------------------------------###

MATRIX_SIZE: int = 20
"""The size (# of users / goods) used to create a (n x n) matrix for the experiment"""

NUMBER_OF_TICKS: int = 10
"""The amount of time / number of loops that the experiment should run for"""

UTILITY_MEAN: int = 4
"""The mean used to generate the users' utility per good"""

UTILITY_STD: int = 2
"""The standard deviation used to generate the users' utility per good"""

USER_BUDGET: int = 10
"""The set "budget" (time, money, etc) a user has to consume goods per tick"""

CONSIDERATION_COST: int = 1
"""The "cost" to the user when looking for a good to be recommended to them"""

USAGE_COST: int = 5
"""The "cost" to the user when using the good that has been recommended to them"""

COUNT_NEGATIVE_REVIEWS: bool = True
"""Whether or not to count negative reviews when finding recommendations for a user"""


###-----------------------------------------------------------###
"""A list of constant variables used during the final analysis"""
###-----------------------------------------------------------###

ANALYZE_N_GOODS: int = 10
"""The number of most and least popular goods that should be analyzed in each test"""

WELL_SERVED_PERCENT: float = 0.8
"""The percent of the optimal utility for a user to be used as a threshold to determine if a user was "well served" by the recommender system"""


###------------------------------------------------------------------------------###
"""A list of constant variables used in (research/recommender_system/__init__.py)"""
###------------------------------------------------------------------------------###

SEED: int = 20
"""A seed to generate random numbers from for test reproducibility"""

BLANK_REP: str = "·"
"""The character used to represent an empty (np.nan) space in matrix printouts"""


###------------------------------------------------------###
"""A list of constant variables used for testing purposes"""
###------------------------------------------------------###

NEG_PERCENT: float = 0.075
"""Weighted chance of randomly generated user reviews containing a negative review (-1)"""

ZERO_PERCENT: float = 0.35
"""Weighted chance of randomly generated user reviews containing a neutral / no review (0)"""

POS_PERCENT: float = 0.075
"""Weighted chance of randomly generated user reviews containing a positive review (1)"""

BLANK_PERCENT: float = 0.5
"""Weighted chance of randomly generated user reviews that have not been used (np.nan)"""
