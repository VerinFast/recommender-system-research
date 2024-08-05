"""A list of static methods related to goods"""

import random

import numpy as np
from numpy.typing import NDArray

from . import controls as const
from . import matrix as mtx
from . import people as ppl


def find_most_popular(matrix: NDArray[np.float_], count_dislikes: bool = const.COUNT_NEGATIVE_REVIEWS) -> int:
	"""Find the location of the most popular good based on positive reviews or positive and negative reviews.

	Args:
			matrix (NDArray[np.float_]): A 2D numpy array containing the reviews
			count_dislikes (bool, optional): Whether to sum both negative and positive reviews

	Returns:
			int: The index of the most popular item. If multiple items receive the same popularity score, a random good is chosen from the sublist.
	"""

	# Count reviews to find each good's "popularity score"
	if count_dislikes:
		popularity_array = np.nansum(matrix, axis=0)
	else:
		popularity_array = np.nansum(matrix > 0, axis=0)

	# Find max value
	max_popularity = np.max(popularity_array)

	# Find all indexes of matching values
	popular_indexes = np.flatnonzero(popularity_array == max_popularity)

	# Randomly choose an index to be returned
	return random.choice(popular_indexes)

def recommend_good(user: ppl.Person, matrix: mtx.RevMatrix, previously_recommended: list[int] = []) -> int:
	"""Recommend a good for the user based on similar users found in the matrix.

	Args:
			user (ppl.Person): The Person to recommend a good for
			matrix (mtx.RevMatrix): The list of reviews to compare self to
			previously_recommended(list[int], optional): A list of indexes to ignore, fill with past recommendations to get new ones recommneded

	Returns:
			int: The index of the recommended good, if no available recommendations are possible (all goods have been used), the size of the matrix is returned instead.
	"""

	# Find the indexes of all user(s) unused (unreviewed) goods
	unused = np.argwhere(np.isnan(user.reviews)).flatten()

	# Exclude goods that have already been recommended, in case multiple recommendations are wanted
	unused_and_not_recommended = unused[np.isin(unused, previously_recommended, invert=True)]

	# If no recommendations are possible, return matrix size
	if len(unused_and_not_recommended) == 0:
		return const.MATRIX_SIZE

	# Find the most similar user(s)
	most_similar = matrix.find_all_most_similar(user)

	# Convert list of similar users into an NDArray[float_]
	similar_users = ppl.Population(most_similar).get_review_table()

	# Return the index of the most popular good from recommendations
	return unused_and_not_recommended[find_most_popular(similar_users[:, unused_and_not_recommended])]

def give_rating(utility: float) -> int:
	"""Rate the good recommended to the user based on its utility.

	Args:
			utility (float): The utility of the item given to the user

	Returns:
			int: The user's rating
	"""

	if utility > (const.UTILITY_MEAN + const.UTILITY_STD):
		# Rate movie positively
		return 1
	elif utility < (const.UTILITY_MEAN - const.UTILITY_STD):
		# Rate movie negatively
		return -1
	else:
		# Leave no / a neutral rating
		return 0
