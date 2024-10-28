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

###--------------------------------------------------------------###
"""The set of classes representing individual and groups of users"""
###--------------------------------------------------------------###

import random
import string

import numpy as np
from numpy.typing import NDArray

from . import controls as const


class Person:
	"""The Person class represents the individual users of the goods, and contains their names, reviews, and true utility."""

	def __init__(self, name: str) -> None:
		"""Create a user.

		Args:
				name (str): The name of the new user
		"""
		self.name = name.title()
		self.budget = const.USER_BUDGET
		self.generated_utility: float = 0

		# Assign reviews to a Person
		self.reviews = np.full(const.MATRIX_SIZE, np.nan)

	 # Assign possible utility to a Person
		self.utility = np.random.normal(const.UTILITY_MEAN, const.UTILITY_STD, const.MATRIX_SIZE)

	def __str__(self) -> str:
			return f"\n{self.name}'s reviews: {self.reviews}"

	def reset_budget(self) -> None:
		"""Reset the user's budget to the constant defined default."""
		self.budget = const.USER_BUDGET

	@staticmethod
	def rand_name(len: int) -> str:
		"""Generate a random name using alpha characters.

		Args:
				len (int): The length of the name

		Returns:
				str: The randomly generated name in Title Case
		"""
		return ''.join(random.choices(string.ascii_lowercase, k=len)).title()



class Population:
	"""The Population class contains functions related to calling Person classes across the entire matrix."""

	def __init__(self, people: list[Person] = []) -> None:
		"""Create a population.

		Args:
				people (list[Person], optional): The list of type Person to be included in the population. If none is provided, a Population of size controls.MATRIX_SIZE is created.
		"""
		if not people:
			# If no list provided, generate a population
			self.people = self.generate_population()
		else:
			self.people = people

	def __str__(self) -> str:
		return "\n".join([str(ppl) for ppl in self.people])

	def generate_population(self) -> list[Person]:
		"""Create a const.MATRIX_SIZE length list of "people" containing randomly generated "names" and blank reviews.

		Returns:
				list[Person]: A list of Person objects with reviews
		"""
		pop: list[Person] = []
		
		for i in range(const.MATRIX_SIZE):
			# Randomly generate a name with 5 characters
			name = Person.rand_name(len=5)

			# Create blank list of reviews (filled with np.nan)
			pop.append(Person(name))

		return pop

	def get_review_table(self) -> NDArray[np.float_]:
		"""Combine all individual 1D numpy arrays stored in people -> list[Person] into a 2D numpy array

		Returns:
				NDArray[np.float]: A 2D numpy array of all reviews
		"""
		return np.vstack(list(map(lambda x : x.reviews, self.people)))

	def soft_copy_matrix(self, matrix: NDArray[np.float_]) -> None:
		"""Makes sure each Person-in-Population's reviews are a view of the matrix and not an independent copy.
		This allows them to be manipulated through the greater matrix or through Person.reviews so that a user's reviews stay consistent.

		Args:
				matrix (NDArray[np.float]): The matrix to be split up and copied
		"""
		for i, person in enumerate(self.people):
			person.reviews = matrix[i,:]

	def reset_budgets(self) -> None:
		"""Reset all of the users' budgets to the constant defined default."""
		for person in self.people:
			person.reset_budget()
