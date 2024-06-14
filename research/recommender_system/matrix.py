"""The review matrix (RevMatrix) and various methods to measure it."""

import heapq
import math
import random

import numpy as np
import numpy.typing as npt

from . import people as ppl
from . import controls as const

class RevMatrix:
	"""A matrix (2D numpy array) containing user reviews."""

	def __init__(self, pop: ppl.Population = ppl.Population()) -> None:
		self.pop = pop
		self.matrix = pop.get_review_table()

	def __str__(self) -> str:
		return f"{self.matrix}"

	def count_shared_likes(self, row1: npt.NDArray[np.float_], row2: npt.NDArray[np.float_], rand: bool = True) -> float:
		"""Count the number of positive reviews for the same goods between two users.

		Args:
				row1 (npt.NDArray[np.float_]): A list of user1's reviews
				row2 (npt.NDArray[np.float_]): A list of user2's reviews
				rand (bool): Whether to add a random number to the returned sum, defaults to True

		Returns:
				float: The sum of the number of positive reviews shared by both users and a random decimal value < 1
    	(Randomness added so the first result with the same similarity score is not chosen every time)
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Replace all non-positive reviews with 0 (including -1, np.nan, and 0)
		row1[row1 != 1] = 0
		row2[row2 != 1] = 0
		comp = np.logical_and(row1, row2)

		return comp.sum() + (random.random() if rand else 0)

	def count_shared_dislikes(self, row1: npt.NDArray[np.float_], row2: npt.NDArray[np.float_], rand: bool = True) -> float:
		"""Count the number of negative reviews for the same goods between two users.

		Args:
				row1 (npt.NDArray[np.float_]): A list of user1's reviews
				row2 (npt.NDArray[np.float_]): A list of user2's reviews
				rand (bool): Whether to add a random number to the returned sum, defaults to True

		Returns:
				float: The sum of the number of negative reviews shared by both users and a random decimal value < 1
    	(Randomness added so the first result with the same similarity score is not chosen every time)
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Replace all non-negative reviews with 0 (including 1, np.nan, and 0)
		row1[row1 != -1] = 0
		row2[row2 != -1] = 0
		comp = np.logical_and(row1, row2)

		return comp.sum() + (random.random() if rand else 0)

	def count_shared_likes_and_dislikes(self, row1: npt.NDArray[np.float_], row2: npt.NDArray[np.float_], rand: bool = True) -> float:
		"""Count the number of positive and negative reviews for the same goods between two users.

		Args:
				row1 (npt.NDArray[np.float_]): A list of user1's reviews
				row2 (npt.NDArray[np.float_]): A list of user2's reviews
				rand (bool): Whether to add a random number to the returned sum, defaults to True

		Returns:
				float: The sum of the number of positive and negative reviews shared by both users and a random decimal value < 1
    	(Randomness added so the first result with the same similarity score is not chosen every time)
		"""
		row1 = row1.copy(); row2 = row2.copy()
		# Combine both positive and negative counts
		shared_likes = math.floor(self.count_shared_likes(row1, row2, False))
		shared_dislikes = math.floor(self.count_shared_dislikes(row1, row2, False))

		return shared_likes + shared_dislikes + (random.random() if rand else 0)

	def find_most_similar(self, user: ppl.Person, count_dislikes: bool = True,count_likes: bool = True,) -> ppl.Person:
		"""Finds the Person who's reviews are most similar to the provided user using the criteria provided.

		Args:
				user (ppl.Person): The base user to find the most similar Person to
				count_dislikes (bool, optional): Whether to count likes (+1) ratings when finding similarities, defaults to True
				count_likes (bool, optional): Whether to count dislikes (-1) ratings when finding similarities, defaults to True
		* If both count_likes and count_dislikes are True / False, they will both be counted

		Returns:
				ppl.Person: The Person object containing the most similar reviews to the provided user
		"""

		# Find which comparison method to use, defaults to counting like and dislikes
		if not count_dislikes:
			comp_list = np.apply_along_axis(self.count_shared_likes, 1, self.matrix, user.reviews)
		elif not count_likes:
			comp_list = np.apply_along_axis(self.count_shared_dislikes, 1, self.matrix, user.reviews)
		else:
			comp_list = np.apply_along_axis(self.count_shared_likes_and_dislikes, 1, self.matrix, user.reviews)

		# Find the top two most similar results (in case first result is user.review, in which case the other result is returned)
		idx_most_similar: list[int] = heapq.nlargest(2, range(len(comp_list)), key = comp_list.__getitem__)

  	# Return the non-matching result
		recommend_not_same: int = idx_most_similar[-1] if self.pop.people[idx_most_similar[0]] == user else idx_most_similar[0]
		
		return self.pop.people[recommend_not_same]



class UtilMatrix:
	"""A matrix (2D numpy array) containing a Person's gain/loss in utility per good."""

	def __init__(self, mean:int = const.UTILITY_MEAN, std:int = const.UTILITY_STD) -> None:
		self.matrix = np.random.normal(mean, std, (const.MATRIX_SIZE, const.MATRIX_SIZE))

	def __str__(self) -> str:
		return f"{self.matrix}"
