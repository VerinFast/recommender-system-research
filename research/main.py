"""The functions and operations that perform the experiment.
Run the experiment using 'python research/main.py' from the root folder of the project."""

import copy
import random

import numpy as np

from recommender_system import controls as const
from recommender_system.goods import Good as good
from recommender_system import matrix as mtx
from recommender_system import people as ppl


# Initialize matrices
utility_matrix = mtx.UtilMatrix()
review_matrix = mtx.RevMatrix(utility_matrix=utility_matrix.matrix)

for tick in range(const.NUMBER_OF_TICKS):
	# Create a copy of the review_matrix so that recommendations in the same tick do not impact each other (all reccomendations are given "simultaneously")
	# Without creating a copy, once a single positive rating is given, all following users will have a high chance to be recommended that good immediately
	# Promotes randomness of opening spread(s) of reviews as different goods are selected by the users
	copy_of_review_matrix = copy.deepcopy(review_matrix)

	# Check that there are still possible recommendations
	if np.count_nonzero(np.isnan(review_matrix.matrix)) == 0:
		break

	# Cycle through each Person in the Population
	for i, (person, copy_of_person) in enumerate(zip(review_matrix.pop.people, copy_of_review_matrix.pop.people)):
		recommendations = []

		while person.budget >= const.CONSIDERATION_COST:
			# Recommend the user a good while they still have a budget to consume
			recommendations.append(good.recommend_good(copy_of_person, copy_of_review_matrix, const.COUNT_NEGATIVE_REVIEWS, previously_recommended=recommendations))

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

	# Run tests related to the experiment each tick


	
# Add a new user to the system
new_user = ppl.Person("_User")
utility_matrix.add_user(new_user)
review_matrix.add_user(new_user)

for tick in range(const.NUMBER_OF_TICKS):
	# Check that there are still possible recommendations
	if np.count_nonzero(np.isnan(review_matrix.matrix)) == 0:
		break

	# Operate only on the new user in the matrix
	recommendations = []

	while new_user.budget >= const.CONSIDERATION_COST:
		# Recommend the new user a good while they still have a budget to consume
		recommendations.append(good.recommend_good(new_user, review_matrix, const.COUNT_NEGATIVE_REVIEWS, previously_recommended=recommendations))

		# If a valid recommendation is possible
		if const.MATRIX_SIZE not in recommendations:

			# Considering a good has an opportunity cost that affects the budget
			new_user.budget -= const.CONSIDERATION_COST

			# Calculate the new user's Expected Utility from consuming the recommended good
			true_utility = utility_matrix.matrix[-1, recommendations[-1]]
			noise = random.normalvariate(const.UTILITY_MEAN, const.UTILITY_STD)
			expected_utility = true_utility + noise

			if new_user.budget >= const.USAGE_COST and expected_utility >= const.UTILITY_MEAN:
				# Consuming the good has a cost that affects the budget
				new_user.budget -= const.USAGE_COST

				# Rate the good after the new user has consumed it
				review_matrix.matrix[-1, recommendations[-1]] = good.give_rating(true_utility)

				# Update the new user's utility generated from consuming the good
				new_user.generated_utility += true_utility

		# If a valid recommendation is not possible (the new user has consumed all goods)
		else:
			break

	# After current set of recommendations done:
	new_user.reset_budget()

	# Run tests related to the experiment each tick
