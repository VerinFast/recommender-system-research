"""The set of classes representing individual and groups of users."""

import random
import string

import numpy as np
import numpy.typing as npt

from . import controls as const


class Person:
	"""The Person class contains functions related to person including "using goods", "leaving reviews", and "getting recommended more goods"."""

	def __init__(self, name: str, reviews: npt.NDArray[np.int_] = np.array([])) -> None:
		self.name = name.title()

		if not reviews.any():
			self.reviews = self.generate_reviews()
		else:
			self.reviews = reviews

	def __str__(self) -> str:
			return f"{self.name}'s reviews:\n{self.reviews}"

	def generate_reviews(self) -> npt.NDArray[np.int_]:
		"""Generate sample user reviews for quick testing.

		Returns:
				npt.NDArray[np.int_]: A set of reviews containing the values 0 (20%), 1 (10%), -1 (10%), and None (60%)
		"""
		return np.random.choice(np.array([0, 1, -1, None]), const.MATRIX_SIZE, p = np.array([const.ZERO_PERCENT, const.POS_PERCENT, const.NEG_PERCENT, const.NONE_PERCENT]))

	@staticmethod
	def rand_name(len: int) -> str:
		"""Generate a random name using alpha characters.

		Args:
				len (int): The length of the name

		Returns:
				str: The randomly generated name in Title Case
		"""
		return ''.join(random.choices(string.ascii_lowercase, k = len)).title()


class Population:
	"""The population class contains functions related to calling Person classes across the entire matrix."""

	def __init__(self, people: list[Person] = [], full_gen: bool = False) -> None:
		"""Create a population.

		Args:
				people (list[Person], optional): The list of type Person to be included in the population. If none is provided, a Population of size controls.MATRIX_SIZE is created.
		"""
		if not people:
			# If no list provided, generate a population
			self.people = self.generate_population(full_gen)
		else:
			self.people = people

	def __str__(self) -> str:
		return "\n".join([str(ppl).replace("\n"," ") for ppl in self.people])

	def generate_population(self, generate_reviews: bool) -> list[Person]:
		"""Create an N length list of "people" containing randomly generated "names" and blank reviews.

		Args:
				generate_reviews (bool): If True randomly generate reviews, else fill with NoneType objects

		Returns:
				list[Person]: A list of Person objects with reviews containing only NoneType objects
		"""
		pop: list[Person] = []
		
		for i in range(const.MATRIX_SIZE):
			# Randomly generate a name with 5 characters
			name = Person.rand_name(len = 5)

			if generate_reviews:
				# Let Person class generate random reviews
				pop.append(Person(name))
			else:
				# Create blank list of reviews (filled with None)
				pop.append(Person(name, np.full(const.MATRIX_SIZE, None)))

		return pop

	def get_review_table(self)-> npt.NDArray[np.int_]:
			"""Combine all individual 1D numpy arrays stored in people -> list[Person] into a 2D numpy array

			Returns:
					npt.NDArray[np.int_]: A 2D numpy array of all reviews
			"""
			return np.vstack(list(map(lambda x : x.reviews, self.people)))
