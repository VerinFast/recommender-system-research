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

###-----------------------------------------###
"""A list of static methods related to goods"""
###-----------------------------------------###

import random

import numpy as np
from numpy.typing import NDArray

from . import controls as const
from . import matrix as mtx
from . import people as ppl


def find_most_popular_location(matrix: NDArray[np.float_]) -> int:
	"""Find the location of the most popular good based on positive and negative reviews.

	Args:
			matrix (NDArray[np.float_]): A 2D numpy array containing the reviews

	Returns:
			int: The index of the most popular item. If multiple items receive the same popularity score, a random good is chosen from the sublist.
	"""

	# Count reviews to find each good's "popularity score"
	popularity_array = np.nansum(matrix, axis=0)

	# Normalize values against the mean to avoid picking the most used item
	if const.RATING_SYSTEM_SCALE != 0:
		mean = ((const.RATING_SYSTEM_SCALE + 1) / 2) if (const.RATING_SYSTEM_MEAN < 0 or const.RATING_SYSTEM_MEAN > const.RATING_SYSTEM_SCALE) else const.RATING_SYSTEM_MEAN
		count_num_goods = (~np.isnan(matrix)).sum(0)
		popularity_array -= count_num_goods * mean

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
	return unused_and_not_recommended[find_most_popular_location(similar_users[:, unused_and_not_recommended])]

def give_rating(utility: float) -> int:
	"""Rate the good recommended to the user based on its utility.

	Args:
			utility (float): The utility of the item given to the user

	Returns:
			int: The user's rating
	"""

	def rating_threshold(rating: int) -> float:
		"""Generate the threshold for each rating by breaking the normal distribution into const.RATING_SYSTEM_SCALE "bins" for each number in the rating system.

		Args:
				rating (int): The rating to create the threshold for

		Returns:
				float: The max threshold of the "bin"
		"""

		# Assign the mean and standard deviation of the rating system
		# If none is given or the values are out of the possible range, use the average of the minimum and maximum ratings (1 an const.RATING_SYSTEM_SCALE)
		mean = ((const.RATING_SYSTEM_SCALE + 1) / 2) if (const.RATING_SYSTEM_MEAN < 0 or const.RATING_SYSTEM_MEAN > const.RATING_SYSTEM_SCALE) else const.RATING_SYSTEM_MEAN
		std  = ((const.RATING_SYSTEM_SCALE - 1) / 6) if (const.RATING_SYSTEM_STD < 0 or const.RATING_SYSTEM_STD > ((const.RATING_SYSTEM_SCALE + 1) / 2)) else const.RATING_SYSTEM_STD
	
		return ((((rating + 0.5) - mean) / std) * const.UTILITY_STD) + const.UTILITY_MEAN

	if const.RATING_SYSTEM_SCALE == 0: # Default [-1, 0, 1]
		if utility > (const.UTILITY_MEAN + const.UTILITY_STD):
			# Rate movie positively
			return 1
		elif utility < (const.UTILITY_MEAN - const.UTILITY_STD):
			# Rate movie negatively
			return -1
		else:
			# Leave no / a neutral rating
			return 0
	else: # For scales of [1 to n]
		# Create "bins" for each possible rating
		for threshold in range(1, const.RATING_SYSTEM_SCALE):
			if utility <= rating_threshold(threshold):
				return threshold
		# If the given utility does not fit into the lower bins, return the max rating
		return const.RATING_SYSTEM_SCALE
