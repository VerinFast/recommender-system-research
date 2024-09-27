###--------------------------------------------------------------------------###
"""A list of constant variables that can be used to manipulate the experiment"""
###--------------------------------------------------------------------------###

MATRIX_SIZE: int = 20
"""The size (# of users / goods) used to create a (n x n) matrix for the experiment"""

NUMBER_OF_TICKS: int = 10
"""The amount of time / number of loops that the experiment should run for"""

UTILITY_MEAN: float = 4
"""The mean used to generate the users' utility per good"""

UTILITY_STD: float = 2
"""The standard deviation used to generate the users' utility per good"""

USER_BUDGET: int = 10
"""The set "budget" (time, money, etc) a user has to consume goods per tick"""

CONSIDERATION_COST: int = 1
"""The "cost" to the user when looking for a good to be recommended to them"""

USAGE_COST: int = 5
"""The "cost" to the user when using the good that has been recommended to them"""


###------------------------------------------------------------------------------------###
"""A list of constant variables that can be used to manipulate the users' rating system"""
###------------------------------------------------------------------------------------###

RATING_SYSTEM_SCALE: int = 0
"""The rating system used by the users to rate the goods they consume:
   0) [-1, 0, 1]
   n) [1 to n] where n is the int provided
"""

RATING_SYSTEM_MEAN: float = -1
"""The mean of the rating system, if set to -1 the ratings are normally distributed around the mean, and if using the default rating scale this value is ignored"""

RATING_SYSTEM_STD: float = -1
"""The standard deviation of the rating system, if set to -1 the ratings are normally distributed around the mean, and if using the default rating scale this value is ignored"""


###-----------------------------------------------------------###
"""A list of constant variables used during the final analysis"""
###-----------------------------------------------------------###

ANALYZE_N_GOODS: int = 10
"""The number of most and least popular goods that should be analyzed during each test"""

WELL_SERVED_PERCENT: float = 0.8
"""The percent of the optimal utility for a user to be used as a threshold to determine if a user was "well served" by the recommender system"""


###------------------------------------------------------------------------------###
"""A list of constant variables used in (research/recommender_system/__init__.py)"""
###------------------------------------------------------------------------------###

SEED: int = 20
"""A seed to generate random numbers from for test reproducibility"""

BLANK_REP: str = "·"
"""The character used to represent an empty (np.nan) space in matrix printouts"""
