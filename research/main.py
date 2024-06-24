import random

import numpy as np

from recommender_system import controls as const
from recommender_system.goods import Good as good
from recommender_system import matrix as mtx


review_matrix = mtx.RevMatrix()
utility_matrix = mtx.UtilMatrix()

for tick in range(const.NUMBER_OF_TICKS):

	# Check that there are still possible recommendations
	if np.count_nonzero(np.isnan(review_matrix.matrix)) == 0:
		break

	# Cycle through each Person in the Population
	for i, person in enumerate(review_matrix.pop.people):
		recommendations = []

		while person.budget >= const.CONSIDERATION_COST:
			# Recommend the user a good while they still have a budget to consume
			recommendations.append(good.recommend_good(person, review_matrix, const.COUNT_NEGATIVE_REVIEWS, previously_recommended=recommendations))

			# If a valid recommendation is possible
			if const.MATRIX_SIZE not in recommendations:

				# Considering a good has an opportunity cost that affects the budget
				person.budget -= const.CONSIDERATION_COST

				# Calculate the user's Expected Utility from consuming the recommended good
				true_utility = utility_matrix.matrix[i, recommendations[-1]]
				noise = random.normalvariate(const.UTILITY_MEAN, const.UTILITY_STD)
				expected_utility = true_utility + noise

				if person.budget >= const.USAGE_COST and expected_utility >= const.UTILITY_MEAN:
					# Consuming the good has a cost that affects the budget
					person.budget -= const.USAGE_COST

					# Rate the good after the user has consumed it
					review_matrix.matrix[i, recommendations[-1]] = good.give_rating(true_utility)

					# Update the user's utility generated from consuming the good
					person.generated_utility += true_utility

			# If a valid recommendation is not possible (the user has consumed all goods)
			else:
				break

	# After all Person's have been cycled through:
	review_matrix.pop.reset_budgets()

	# Run tests related to the experiment
