"""The functions and operations that perform the experiment.
Run the experiment using 'python research/main.py' from the root folder of the project."""

import copy
import heapq
import random

import numpy as np

from recommender_system import analysis
from recommender_system import controls as const
from recommender_system import goods as good
from recommender_system import matrix as mtx
from recommender_system import people as ppl


def green_or_red(string: str | int | float, green_logic: bool, red_logic: bool = True, space_before: bool = False, space_after: bool = False) -> str:
	return ('\033[32m' + (' ' if space_before == True else '') if green_logic else ('\033[31m' if red_logic else (' ' if space_after == True else ''))) + str(string)

print(f"\033[90;2mSEED [{const.SEED}]\033[0m")





###################################################################################################
################################# Generate Base Population Matrix #################################
###################################################################################################

# Initialize matrices
utility_matrix = mtx.UtilMatrix()
review_matrix = mtx.RevMatrix(utility_matrix=utility_matrix.matrix)

### Run simulation:
for tick in range(const.NUMBER_OF_TICKS):
	# Print out current tick information
	print('\033[96;2m' + ("\nTICK #1..." if tick == 0 else f"{tick+1}...") + '\033[0m', end=' ', flush=True)
	
	# Create a copy of the review_matrix so that recommendations in the same tick do not impact each other (all reccomendations are given "simultaneously")
	# Without creating a copy, once a single positive rating is given all following users in the loop will have a high chance to be recommended that good
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
			recommendations.append(good.recommend_good(copy_of_person, copy_of_review_matrix, previously_recommended=recommendations))

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
					person.reviews[recommendations[-1]] = good.give_rating(true_utility)

					# Update the user's utility generated from consuming the good
					person.generated_utility += true_utility

			# If a valid recommendation is not possible (the user has consumed all goods)
			else:
				break

	# After all Person's have been cycled through:
	review_matrix.pop.reset_budgets()

print()
##### Format of matrix prinout #####
# _Name's reviews: [ ·  ·  ·  ·  · ... ·  ·  ·  ·  · ] = Sum_of_Reviews → Generated_Utility (Optimal_Utility)
print("".join([str(per) + f" = {green_or_red(int(np.nansum(per.reviews)), int(np.nansum(per.reviews)) > (0 if const.RATING_SYSTEM_SCALE == 0 else (~np.isnan(per.reviews)).sum(0) * (((const.RATING_SYSTEM_SCALE + 1) / 2) if (const.RATING_SYSTEM_MEAN < 0 or const.RATING_SYSTEM_MEAN > const.RATING_SYSTEM_SCALE) else const.RATING_SYSTEM_MEAN)), int(np.nansum(per.reviews)) < (0 if const.RATING_SYSTEM_SCALE == 0 else (~np.isnan(per.reviews)).sum(0) * (((const.RATING_SYSTEM_SCALE + 1) / 2) if (const.RATING_SYSTEM_MEAN < 0 or const.RATING_SYSTEM_MEAN > const.RATING_SYSTEM_SCALE) else const.RATING_SYSTEM_MEAN)), space_before=(const.RATING_SYSTEM_SCALE == 0), space_after=(const.RATING_SYSTEM_SCALE == 0))}\033[0m → {green_or_red(round(per.generated_utility, 1), per.generated_utility >= (analysis.find_optimal_utility(per) * const.WELL_SERVED_PERCENT))}\033[0m ({round(analysis.find_optimal_utility(per), 1)})" for per in review_matrix.pop.people]))
print()

# Optimal / actual utility, and percentage of optimal utility achieved
max_util = sum(list(map(analysis.find_optimal_utility, review_matrix.pop.people)))
actual_util = sum(per.generated_utility for per in review_matrix.pop.people)
print(f"Maximum User Utility = \033[96m{round(max_util, 1)}\033[0m")
print(f"Actual User Utility = \033[96m{round(actual_util, 1)}\033[0m")
print(f"\033[96;4m{round((actual_util / max_util) * 100)}%\033[0m of optimal utility was achieved")
print()

# Percentage of the population being "well served"
print(f"\033[96m{round((sum(analysis.is_well_served(per, const.WELL_SERVED_PERCENT) for per in review_matrix.pop.people) / const.MATRIX_SIZE) * 100)}%\033[0m of the population was \"well served\" (achieved {round(const.WELL_SERVED_PERCENT * 100)}% of max utility)")
print()

# Percentage of users who consumed all of the n most popular goods
print(f"\033[96m{round((sum(analysis.all_most_popular_recommended(per, review_matrix.matrix, 1) for per in review_matrix.pop.people) / const.MATRIX_SIZE) * 100)}%\033[0m of users used the most popular good")
print(f"\033[96m{round((sum(analysis.all_most_popular_recommended(per, review_matrix.matrix, (const.ANALYZE_N_GOODS / 4).__ceil__()) for per in review_matrix.pop.people) / const.MATRIX_SIZE) * 100)}%\033[0m of users used the {(const.ANALYZE_N_GOODS / 4).__ceil__()} most popular goods")
print(f"\033[96m{round((sum(analysis.all_most_popular_recommended(per, review_matrix.matrix, (const.ANALYZE_N_GOODS / 2).__ceil__()) for per in review_matrix.pop.people) / const.MATRIX_SIZE) * 100)}%\033[0m of users used the {(const.ANALYZE_N_GOODS / 2).__ceil__()} most popular goods")





###################################################################################################
############################# Generate Optimal Recommendations Matrix #############################
###################################################################################################
print('\n\033[96;2m' + "Generating optimal utility matrix..." + '\033[0m\n')

# Make a matrix containing the optimal good consumption for each user
optimal_matrix = mtx.RevMatrix(utility_matrix=utility_matrix.matrix)

for person in optimal_matrix.pop.people:
	# Get the indexes of the goods that give the user the most utility
	recommendations = heapq.nlargest(const.NUMBER_OF_TICKS * (const.USER_BUDGET // (const.CONSIDERATION_COST + const.USAGE_COST)), range(const.MATRIX_SIZE), key=person.utility.__getitem__)

	for good_idx in recommendations:
		# Rate the good after the user has consumed it
		person.reviews[good_idx] = good.give_rating(person.utility[good_idx])

		# Update the user's utility generated from consuming the good
		person.generated_utility += person.utility[good_idx]

# Run above analysis questions on a "perfect" model
print(f"\033[96m{round((sum(analysis.all_most_popular_recommended(per, optimal_matrix.matrix, 1) for per in optimal_matrix.pop.people) / const.MATRIX_SIZE) * 100)}%\033[0m of optimal users used the most popular good")
print(f"\033[96m{round((sum(analysis.all_most_popular_recommended(per, optimal_matrix.matrix, (const.ANALYZE_N_GOODS / 4).__ceil__()) for per in optimal_matrix.pop.people) / const.MATRIX_SIZE) * 100)}%\033[0m of optimal users used the {(const.ANALYZE_N_GOODS / 4).__ceil__()} most popular goods")
print(f"\033[96m{round((sum(analysis.all_most_popular_recommended(per, optimal_matrix.matrix, (const.ANALYZE_N_GOODS / 2).__ceil__()) for per in optimal_matrix.pop.people) / const.MATRIX_SIZE) * 100)}%\033[0m of optimal users used the {(const.ANALYZE_N_GOODS / 2).__ceil__()} most popular goods")





###################################################################################################
################################ Generate New User Recommendations ################################
###################################################################################################
print("\n" + "-"*100 + "\n")

# Add a new user to the system
new_user = ppl.Person("_User")
utility_matrix.add_user(new_user)
review_matrix.add_user(new_user)

# Save usable recommendations so that their frequency can be calculated later
total_recommendations = []

### Run simulation:
for tick in range(const.NUMBER_OF_TICKS):
	# Print out current tick information
	print('\033[96;2m' + ("TICK #1..." if tick == 0 else f"{tick+1}...") + '\033[0m', end=' ')
  
	# Check that there are still possible recommendations
	if np.count_nonzero(np.isnan(review_matrix.matrix)) == 0:
		break

	recommendations = []

	while new_user.budget >= const.CONSIDERATION_COST:
		# Recommend the new user a good while they still have a budget to consume
		recommendations.append(good.recommend_good(new_user, review_matrix, previously_recommended=recommendations))

		# If a valid recommendation is possible
		if const.MATRIX_SIZE not in recommendations:

			# Considering a good has an opportunity cost that affects the budget
			new_user.budget -= const.CONSIDERATION_COST

			# Calculate the new user's Expected Utility from consuming the recommended good
			true_utility = utility_matrix.matrix[-1, recommendations[-1]]
			noise = random.normalvariate(const.UTILITY_MEAN, const.UTILITY_STD)
			expected_utility = true_utility + noise

			# Save all recommendations that have a chance to be consumed by the user
			if new_user.budget >= const.USAGE_COST:
				total_recommendations.append(recommendations[-1])

			if new_user.budget >= const.USAGE_COST and expected_utility >= const.UTILITY_MEAN:
				# Consuming the good has a cost that affects the budget
				new_user.budget -= const.USAGE_COST

				# Rate the good after the new user has consumed it
				new_user.reviews[recommendations[-1]] = good.give_rating(true_utility)

				# Update the new user's utility generated from consuming the good
				new_user.generated_utility += true_utility

		# If a valid recommendation is not possible (the new user has consumed all goods)
		else:
			break

	# After current set of recommendations done:
	new_user.reset_budget()

print("\n" + str(new_user) + "\n")





###################################################################################################
######################################### Run Final Tests #########################################
###################################################################################################

### Percentage of the n most popular goods consumed by the new user
print(f"\033[96m{round((analysis.num_most_popular_recommended(new_user, review_matrix.matrix, const.ANALYZE_N_GOODS) / const.ANALYZE_N_GOODS) * 100)}%\033[0m of the {const.ANALYZE_N_GOODS} most popular suggestions were received")

### Number of n popular recommendations compared to n unpopular recommendations
print(analysis.likely_recommended(sum(good in analysis.find_most_popular(review_matrix.matrix[:-1], 1) for good in total_recommendations), sum(good in analysis.find_least_popular(review_matrix.matrix[:-1], 1) for good in total_recommendations)))
print(analysis.likely_recommended(sum(good in analysis.find_most_popular(review_matrix.matrix[:-1], (const.ANALYZE_N_GOODS / 2).__floor__()) for good in total_recommendations), sum(good in analysis.find_least_popular(review_matrix.matrix[:-1], (const.ANALYZE_N_GOODS / 2).__floor__()) for good in total_recommendations), (const.ANALYZE_N_GOODS / 2).__floor__()))
print(analysis.likely_recommended(sum(good in analysis.find_most_popular(review_matrix.matrix[:-1], const.ANALYZE_N_GOODS) for good in total_recommendations), sum(good in analysis.find_least_popular(review_matrix.matrix[:-1], const.ANALYZE_N_GOODS) for good in total_recommendations), const.ANALYZE_N_GOODS))

print()

### User's utility if they consumed only the most optimal goods
print(f"The user's optimal utility was \033[96m{round(analysis.find_optimal_utility(new_user), 1)}\033[0m")

### User's utility if they consumed only the most popular goods
print(f"The utility of the most popular goods was \033[96m{round(analysis.find_popular_utility(new_user, review_matrix.matrix), 1)}\033[0m")

### Actual user utility
print(f"The user's actual utility was \033[96;4m{round(new_user.generated_utility, 1)}\033[0m")





###################################################################################################
##################################### Generate Random Reviews #####################################
###################################################################################################

rand_user = ppl.Person("_Rand")
rand_user.utility = new_user.utility

# Randomly select goods for the random user to consume
rand_recommendations = random.sample(range(const.MATRIX_SIZE), sum(~np.isnan(new_user.reviews)))

for good_idx in rand_recommendations:
	# Rate the good after the user has consumed it
	rand_user.reviews[good_idx] = good.give_rating(rand_user.utility[good_idx])

	# Update the user's utility generated from consuming the good
	rand_user.generated_utility += rand_user.utility[good_idx]

### Compare the new user to a user who randomly selects goods to consume
print(f"\nA new user who uses the recommender system generates \033[96m{round(new_user.generated_utility - rand_user.generated_utility, 1)}\033[0m (\033[96;4m{round((new_user.generated_utility - rand_user.generated_utility) / const.UTILITY_MEAN, 1)}x\033[0m mean) more utility compared to a user who chooses goods randomly")

print()
