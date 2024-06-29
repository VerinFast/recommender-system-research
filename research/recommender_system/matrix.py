"""The two matrices, the review matrix (RevMatrix) and the utility matrix (UtilMatrix), and various methods to measure them."""

import heapq
import math
import random

import numpy as np
from numpy.typing import NDArray

from . import controls as const
from . import people as ppl


class UtilMatrix:
	"""A matrix (2D numpy array) containing a Person's gain/loss in utility per good."""

	def __init__(self, mean: int = const.UTILITY_MEAN, std: int = const.UTILITY_STD) -> None:
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

	def __init__(self, pop: ppl.Population = ppl.Population(), utility_matrix: NDArray[np.float_] = UtilMatrix().matrix) -> None:
		self.pop = pop
		self.matrix = pop.get_review_table()

		# Make sure reviews are always a representation of the overall matrix
		self.pop.soft_copy_matrix(self.matrix)

		# Assign users their possible utility
		for idx, person in enumerate(self.pop.people):
			person.utility = utility_matrix[idx]

	def __str__(self) -> str:
		return f"{self.matrix}"

	def count_shared_likes(self, row1: NDArray[np.float_], row2: NDArray[np.float_], rand: bool = True) -> float:
		"""Count the number of positive reviews for the same goods between two users.

		Args:
				row1 (NDArray[np.float_]): A list of user1's reviews
				row2 (NDArray[np.float_]): A list of user2's reviews
				rand (bool): Whether to add a random number to the returned sum, defaults to True

		Returns:
				float: The sum of the number of positive reviews shared by both users and a random decimal value < 1
			(Noise added so the first result with the same similarity score is not chosen every time)
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Replace all non-positive reviews with 0 (including -1, np.nan, and 0)
		row1[row1 != 1] = 0
		row2[row2 != 1] = 0
		comp = np.logical_and(row1, row2)

		return comp.sum() + (random.random() if rand else 0)

	def count_shared_dislikes(self, row1: NDArray[np.float_], row2: NDArray[np.float_], rand: bool = True) -> float:
		"""Count the number of negative reviews for the same goods between two users.

		Args:
				row1 (NDArray[np.float_]): A list of user1's reviews
				row2 (NDArray[np.float_]): A list of user2's reviews
				rand (bool): Whether to add a random number to the returned sum, defaults to True

		Returns:
				float: The sum of the number of negative reviews shared by both users and a random decimal value < 1
			(Noise added so the first result with the same similarity score is not chosen every time)
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Replace all non-negative reviews with 0 (including 1, np.nan, and 0)
		row1[row1 != -1] = 0
		row2[row2 != -1] = 0
		comp = np.logical_and(row1, row2)

		return comp.sum() + (random.random() if rand else 0)

	def count_shared_likes_and_dislikes(self, row1: NDArray[np.float_], row2: NDArray[np.float_], rand: bool = True) -> float:
		"""Count the number of positive and negative reviews for the same goods between two users.

		Args:
				row1 (NDArray[np.float_]): A list of user1's reviews
				row2 (NDArray[np.float_]): A list of user2's reviews
				rand (bool): Whether to add a random number to the returned sum, defaults to True

		Returns:
				float: The sum of the number of positive and negative reviews shared by both users and a random decimal value < 1
			(Noise added so the first result with the same similarity score is not chosen every time)
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Combine both positive and negative counts
		shared_likes = math.floor(self.count_shared_likes(row1, row2, False))
		shared_dislikes = math.floor(self.count_shared_dislikes(row1, row2, False))

		return shared_likes + shared_dislikes + (random.random() if rand else 0)

	def find_most_similar(self, user: ppl.Person, count_dislikes: bool = True, count_likes: bool = True) -> ppl.Person:
		"""Finds the Person who's reviews are most similar to the provided user using the criteria provided.

		Args:
				user (ppl.Person): The base user to find the most similar Person to
				count_dislikes (bool, optional): Whether to count dislikes (-1) ratings when finding similarities, defaults to True
				count_likes (bool, optional): Whether to count likes (+1) ratings when finding similarities, defaults to True
		* If both count_likes and count_dislikes are True / False, they will both be counted

		Returns:
				ppl.Person: The Person object containing the most similar reviews to the provided user
		"""

		# Find which comparison method to use, defaults to counting both likes and dislikes
		# "comp_list" contains the total count of matching likes and/or dislikes shared between the two users
		if count_likes and not count_dislikes:
			comp_list = np.apply_along_axis(self.count_shared_likes, 1, self.matrix, user.reviews)
		elif count_dislikes and not count_likes:
			comp_list = np.apply_along_axis(self.count_shared_dislikes, 1, self.matrix, user.reviews)
		else:
			comp_list = np.apply_along_axis(self.count_shared_likes_and_dislikes, 1, self.matrix, user.reviews)

		# Find the indexes of the top two most similar results (in case first result is user.review)
		idx_most_similar: list[int] = heapq.nlargest(2, range(len(comp_list)), key=comp_list.__getitem__)

		# Return the index of the non-matching Person object
		recommend_not_same: int = idx_most_similar[-1] if self.pop.people[idx_most_similar[0]] == user else idx_most_similar[0]
		
		return self.pop.people[recommend_not_same]

	def find_all_most_similar(self, user: ppl.Person, count_dislikes: bool = True, count_likes: bool = True) -> list[ppl.Person]:
		"""Finds the Person(s) who's reviews are most similar to the provided user using the criteria provided.

		Args:
				user (ppl.Person): The base user to find the most similar Person to
				count_dislikes (bool, optional): Whether to count dislikes (-1) ratings when finding similarities, defaults to True
				count_likes (bool, optional): Whether to count likes (+1) ratings when finding similarities, defaults to True
		* If both count_likes and count_dislikes are True / False, they will both be counted

		Returns:
				ppl.Person: All Person objects containing the most similar reviews to the provided user
		"""

		# Find which comparison method to use, defaults to counting both likes and dislikes
		# "comp_list" contains the total count of matching likes and/or dislikes shared between the two users
		if count_likes and not count_dislikes:
			comp_list = np.apply_along_axis(self.count_shared_likes, 1, self.matrix, user.reviews)
		elif count_dislikes and not count_likes:
			comp_list = np.apply_along_axis(self.count_shared_dislikes, 1, self.matrix, user.reviews)
		else:
			comp_list = np.apply_along_axis(self.count_shared_likes_and_dislikes, 1, self.matrix, user.reviews)

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
