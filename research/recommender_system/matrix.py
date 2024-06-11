"""The review matrix (RevMatrix) and various methods to measure it."""

import numpy as np

from . import people as ppl
from . import controls as const

class RevMatrix:
	"""A matrix (2D numpy array) containing user reviews."""

	def __init__(self, pop: ppl.Population = ppl.Population()) -> None:
		self.pop = pop
		self.matrix = pop.get_review_table()
