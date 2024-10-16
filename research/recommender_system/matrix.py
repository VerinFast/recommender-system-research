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

###------------------------------------------------------------------------------------------------------------------------###
"""The two matrices, the review matrix (RevMatrix) and the utility matrix (UtilMatrix), and various methods to measure them"""
###------------------------------------------------------------------------------------------------------------------------###

import heapq
import math
import random

import numpy as np
from numpy.typing import NDArray

from . import controls as const
from . import people as ppl


class UtilMatrix:
	"""A matrix (2D numpy array) containing a Person's gain/loss in utility per good."""

	def __init__(self, mean: float = const.UTILITY_MEAN, std: float = const.UTILITY_STD) -> None:
		"""A matrix containing the utility associated with each good in a RevMatrix.

		Args:
				mean (float, optional): The mean of the utility distribution. Defaults to const.UTILITY_MEAN.
				std (float, optional): The standard deviation of the utility distribution. Defaults to const.UTILITY_STD.
		"""
		self.mean = mean; self.std = std
		self.matrix = np.random.normal(mean, std, (const.MATRIX_SIZE, const.MATRIX_SIZE))

	def __str__(self) -> str:
		return f"{self.matrix}"

	def add_user(self, user: ppl.Person) -> None:
		"""Adds a new user to the utility matrix and generates a utility array for them.

		Args:
				user (ppl.Person): The user to be added to the utility matrix
		"""
		# Create utility array for new user
		user_utility_array = np.random.normal(self.mean, self.std, const.MATRIX_SIZE)
		user.utility = user_utility_array

		# Add new user to matrix
		self.matrix = np.vstack([self.matrix, user_utility_array])


class RevMatrix:
	"""A matrix (2D numpy array) containing user reviews."""

	def __init__(self, utility_matrix: NDArray[np.float_]) -> None:
		"""Creates an empty matrix for reviews from the population.

		Args:
				utility_matrix (NDArray[np.float_]): The utility matrix associated with each good in the review matrix
		"""
		self.pop = ppl.Population()
		self.matrix = self.pop.get_review_table()

		# Make sure reviews are always a representation of the overall matrix
		self.pop.soft_copy_matrix(self.matrix)

		# Assign users their possible utility
		for idx, person in enumerate(self.pop.people):
			person.utility = utility_matrix[idx]

	def __str__(self) -> str:
		return f"{self.matrix}"

	def count_shared_likes(self, row1: NDArray[np.float_], row2: NDArray[np.float_]) -> int:
		"""Count the number of matching positive reviews for the same goods between two users.

		Args:
				row1 (NDArray[np.float_]): A list of user1's reviews
				row2 (NDArray[np.float_]): A list of user2's reviews

		Returns:
				int: The sum of the number of positive reviews shared by both users
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Replace all non-positive reviews with 0 (including -1, np.nan, and 0)
		row1[row1 != 1] = 0
		row2[row2 != 1] = 0
	
		return np.logical_and(row1, row2).sum()

	def count_shared_dislikes(self, row1: NDArray[np.float_], row2: NDArray[np.float_]) -> int:
		"""Count the number of matching negative reviews for the same goods between two users.

		Args:
				row1 (NDArray[np.float_]): A list of user1's reviews
				row2 (NDArray[np.float_]): A list of user2's reviews

		Returns:
				int: The sum of the number of negative reviews shared by both users
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Replace all non-negative reviews with 0 (including 1, np.nan, and 0)
		row1[row1 != -1] = 0
		row2[row2 != -1] = 0

		return np.logical_and(row1, row2).sum()

	def count_shared_likes_and_dislikes(self, row1: NDArray[np.float_], row2: NDArray[np.float_]) -> float:
		"""Count the number of matching positive and negative reviews for the same goods between two users.

		Args:
				row1 (NDArray[np.float_]): A list of user1's reviews
				row2 (NDArray[np.float_]): A list of user2's reviews

		Returns:
				float: The sum of the number of positive and negative reviews shared by both users and a random decimal value < 1
			(Noise added so the first result with the same similarity score is not chosen every time)
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Combine both positive and negative counts
		shared_likes = self.count_shared_likes(row1, row2)
		shared_dislikes = self.count_shared_dislikes(row1, row2)

		return shared_likes + shared_dislikes + random.random()

	def count_shared_ratings(self, row1: NDArray[np.float_], row2: NDArray[np.float_]) -> float:
		"""Count the number of matching reviews for the same goods between two users.

		Args:
				row1 (NDArray[np.float_]): A list of user1's reviews
				row2 (NDArray[np.float_]): A list of user2's reviews

		Returns:
				float: The sum of the number of reviews shared by both users and a random decimal value < 1
			(Noise added so the first result with the same similarity score is not chosen every time)
		"""
		# Replace all non-positive reviews with 0 (including -1, np.nan, and 0)
	
		return (row1 == row2).sum() + random.random()

	def find_all_most_similar(self, user: ppl.Person) -> list[ppl.Person]:
		"""Finds the Person(s) who's reviews are most similar to the provided user's.

		Args:
				user (ppl.Person): The base user to find the most similar Person to

		Returns:
				ppl.Person: All Person objects containing the most similar reviews to the provided user
		"""

		# "comp_list" contains the total count of matching likes and dislikes shared between the two users
		if const.RATING_SYSTEM_SCALE == 0: # Default count
			comp_list = np.apply_along_axis(self.count_shared_likes_and_dislikes, 1, self.matrix, user.reviews)
		else: # Count for rating systems of [1 to n]
			comp_list = np.apply_along_axis(self.count_shared_ratings, 1, self.matrix, user.reviews)

		# Find the indexes of the top two most similar results (in case first result is user.review)
		idx_most_similar: list[int] = heapq.nlargest(2, range(len(comp_list)), key=comp_list.__getitem__)

		# Return the index of the non-matching Person object
		recommend_not_same: int = idx_most_similar[-1] if self.pop.people[idx_most_similar[0]] == user else idx_most_similar[0]

		# Get noiseless comparison value from comp_list
		best_comp: int = math.floor(comp_list[recommend_not_same])

		# Find list of indexes with matching noiseless comparison values
		best_comps: NDArray[np.int_] = np.flatnonzero(comp_list.astype(int) == best_comp)

		# Remove user index from list of best indexes
		best_comps = best_comps[best_comps != self.pop.people.index(user)]

		return [self.pop.people[i] for i in best_comps]

	def add_user(self, user: ppl.Person) -> None:
		"""Adds a new user to the review matrix.

		Args:
				user (ppl.Person): The user to be added to the review matrix
		"""
		# Add new user to Population
		self.pop.people.append(user)

		# Add new user to matrix
		self.matrix = np.vstack([self.matrix, user.reviews])
		user.reviews = self.matrix[-1,:]
