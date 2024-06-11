import random

import numpy as np
import numpy.testing as np_test
import pytest

from research.recommender_system import matrix as mtx
from research.recommender_system import people as ppl

###################################################################################################
############################################ Test Code ############################################
###################################################################################################

@pytest.fixture
def rand():
  random.seed(10)
  np.random.seed(10)


def test_name(rand):
	# Test that individual Person names can be accessed from a Matrix instance
	mtx1 = mtx.RevMatrix()
	print(mtx1.pop.people[0])

	assert mtx1.pop.people[0].name == "Olpfv"


def test_reviews():
  # Test that manually assigning user reviews is handled correctly (vs randomly generating them)
	p1 = ppl.Person("Aob", np.array([1, 2, 3]))
	p2 = ppl.Person("Bob", np.array([4, 5, 6]))
	p3 = ppl.Person("Cob", np.array([-1, 1, 0]))
	p4 = ppl.Person("Dob", np.array([7, 8, 9]))

	pop2 = ppl.Population([p1, p2, p3, p4])

	np_test.assert_array_equal(pop2.people[2].reviews, [-1, 1, 0])
