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

###-----------------------------------------------------------------------------------------------------------------###
"""Functions corresponding to important questions that help test / prove the hypotheses stated in the research paper"""
###-----------------------------------------------------------------------------------------------------------------###

import heapq

import numpy as np
from numpy.typing import NDArray

from . import controls as const
from . import people as ppl


def is_well_served(person: ppl.Person, percent: float) -> bool:
	"""Determine whether a user was "well served" by the recommender system.

	Args:
			person (ppl.Person): The user to run the analysis on
			percent (float): The percent of total utility that needs to be passed in order for a user to be "well served"

	Returns:
			bool: Whether or not the user passed the above threshold and was therefore "well served"
	"""

	return person.generated_utility >= (find_optimal_utility(person) * percent)

def find_most_popular(matrix: NDArray[np.float_], n: int) -> list[int]:
	"""Find the n most popular goods from a matrix of reviews.

	Args:
			matrix (NDArray[np.float_]): The matrix filled with user reviews
			n (int): The number of goods to find

	Returns:
			list[int]: The indexes of the n most popular goods
	"""

	# Count reviews to find each good's "popularity score"
	popularity_array = np.nansum(matrix, axis=0)

	# Normalize values against the mean to avoid picking the most used item
	if const.RATING_SYSTEM_SCALE != 0:
		mean = ((const.RATING_SYSTEM_SCALE + 1) / 2) if (const.RATING_SYSTEM_MEAN < 0 or const.RATING_SYSTEM_MEAN > const.RATING_SYSTEM_SCALE) else const.RATING_SYSTEM_MEAN
		count_num_goods = (~np.isnan(matrix)).sum(0)
		popularity_array -= count_num_goods * mean

	# Get indexes of top n results
	return heapq.nlargest(n, range(const.MATRIX_SIZE), key=popularity_array.__getitem__)

def find_least_popular(matrix: NDArray[np.float_], n: int) -> list[int]:
	"""Find the n least popular goods from a matrix of reviews.

	Args:
			matrix (NDArray[np.float_]): The matrix filled with user reviews
			n (int): The number of goods to find

	Returns:
			list[int]: The indexes of the n least popular goods
	"""

	# Count reviews to find each good's "popularity score"
	popularity_array = np.nansum(matrix, axis=0)

	# Normalize values against the mean to avoid picking the most used item
	if const.RATING_SYSTEM_SCALE != 0:
		mean = ((const.RATING_SYSTEM_SCALE + 1) / 2) if (const.RATING_SYSTEM_MEAN < 0 or const.RATING_SYSTEM_MEAN > const.RATING_SYSTEM_SCALE) else const.RATING_SYSTEM_MEAN
		count_num_goods = (~np.isnan(matrix)).sum(0)
		popularity_array -= count_num_goods * mean

	# Get indexes of top n results
	return heapq.nsmallest(n, range(const.MATRIX_SIZE), key=popularity_array.__getitem__)

def num_most_popular_recommended(user: ppl.Person, matrix: NDArray[np.float_], top_results: int) -> int:
	"""Determine how many of the n most popular goods were used by the given user.

	Args:
			user (ppl.Person): The user to run the analysis on
			matrix (NDArray[np.float_]): The matrix filled with user reviews
			top_results (int): The number of goods to find

	Returns:
			int: The number of most popular goods that the user used
	"""

	# Get indexes of the most popular goods
	most_popular = find_most_popular(matrix, top_results)

	# Get indexes of all used goods
	used = np.argwhere(~np.isnan(user.reviews)).flatten()

	return sum(np.isin(used, most_popular))

def all_most_popular_recommended(user: ppl.Person, matrix: NDArray[np.float_], top_results: int) -> int:
	"""Determine if the given user used all n most popular goods.

	Args:
			user (ppl.Person): The user to run the analysis on
			matrix (NDArray[np.float_]): The matrix filled with user reviews
			top_results (int): The number of goods to check again

	Returns:
			int: Return 1 if the user used all n most popular goods, and 0 if they did not
	"""

	return 1 if num_most_popular_recommended(user, matrix, top_results) == top_results else 0

def likely_recommended(most: int, least: int, count: int = 1) -> str:
	"""Return how much more likely the most popular good(s) were recommended than the least popular good(s).

	Args:
			most (int): The number of times the most popular good(s) were recommended
			least (int): The number of times the least popular good(s) were recommended
			count (int, optional): How many of the most and least popular goods were counted, defaults to 1

	Returns:
			str: A printable string response to how many more goods were recommended
	"""

	if most == 0 and least == 0:
		return f"The {'' if count == 1 else f'{count} '}most popular good{'' if count == 1 else 's'} and {'' if count == 1 else f'{count} '}least popular good{'' if count == 1 else 's'} were not recommended"
	elif most == 0:
		return f"The {'' if count == 1 else f'{count} '}least popular good{f' \033[2m({least}x)\033[0m was' if count == 1 else f's \033[2m({least}x)\033[0m were'} recommended, but the {'' if count == 1 else f'{count} '}most popular good{' was' if count == 1 else 's were'} not"
	elif least == 0:
		return f"The {'' if count == 1 else f'{count} '}most popular good{f' \033[2m({most}x)\033[0m was' if count == 1 else f's \033[2m({most}x)\033[0m were'} recommended, but the {'' if count == 1 else f'{count} '}least popular good{' was' if count == 1 else 's were'} not"
	elif most == least:
		return f"The {'' if count == 1 else f'{count} '}most popular good{f' \033[2m({most}x)\033[0m was' if count == 1 else f's \033[2m({most}x)\033[0m were'} recommended the same amount of times as the {'' if count == 1 else f'{count} '}least popular good{f' \033[2m({least}x)\033[0m' if count == 1 else f's \033[2m({least}x)\033[0m'}"
	else:
		return f"The {'' if count == 1 else f'{count} '}most popular good{f' \033[2m({most}x)\033[0m was' if count == 1 else f's \033[2m({most}x)\033[0m were'} recommended \033[96m{round(most / least, 1)}x\033[0m more than the {'' if count == 1 else f'{count} '}least popular good{f' \033[2m({least}x)\033[0m' if count == 1 else f's \033[2m({least}x)\033[0m'}"

def find_optimal_utility(user: ppl.Person) -> float:
	"""Find the optimal utility for the user.

	Args:
			user (ppl.Person): The user to run the anlysis on

	Returns:
			float: The optimal utility for he user if they had picked the highest utility good each time
	"""

	# Find the number of goods the user reviewed
	num_goods = sum(~np.isnan(user.reviews))

	return sum(heapq.nlargest(num_goods, user.utility))

def find_popular_utility(user: ppl.Person, matrix: NDArray[np.float_]) -> float:
	"""Find the utility for the user if they had just chosen the most popular goods.

	Args:
			user (ppl.Person): The user to run the analysis on
			matrix (NDArray[np.float_]): The matrix filled with user reviews

	Returns:
			float: The utitility of the top_results most popular goods for the user
	"""

	popular_idxs = find_most_popular(matrix, sum(~np.isnan(user.reviews)))
	return sum(user.utility[popular_idxs])
