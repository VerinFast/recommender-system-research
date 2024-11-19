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

###--------------------------------------------------------###
"""The functions and operations that perform the experiment"""
###--------------------------------------------------------###

#############################################################################################################
# The experiment can be run using any of the following terminal commands from the root folder of the project:
###> python research/main.py
###> python research/main.py [num_loops]
###> python research/main.py [num_loops] [matrix_size]
###> python research/main.py [num_loops] [matrix_size] [num_ticks]
#############################################################################################################

import argparse
import copy
import csv
import heapq
import os
import random

import numpy as np

from recommender_system import analysis
from recommender_system import controls as const
from recommender_system import goods as good
from recommender_system import matrix as mtx
from recommender_system import people as ppl
from __init__ import *

# Set up command line parser
parser = argparse.ArgumentParser(
	prog="RecommenderSystemResearch",
	description=program_description,
	epilog=copyright_statement,
	formatter_class=MyFormatter
)

# Add legal flags to parser
parser.add_argument(
	'-w', '--warranty',
	action='version',
	version=copyright_w,
	help='display the warranty notice'
)
parser.add_argument(
	'-c', '--conditions',
	action='version',
	version=copyright_c,
	help='display the redistribution conditions'
)

# Add display flags to parser
parser.add_argument(
	'-p', '--print',
	action='store_true',
	help='print all matrices and analysis information'
)
parser.add_argument(
	'-b', '--blank_print',
	action='store_true',
	help='remove all printing options, only prints completion message'
)

# Add experimental positional arguments to parser
parser.add_argument(
	'num_loops',
	action='store',
	nargs='?',
	default=const.NUMBER_OF_EXPERIMENTS,
	type=check_positive,
	help="The number of times the experiment will run so statistically significant results can be attained"
)
parser.add_argument(
	'matrix_size',
	action='store',
	nargs='?',
	default=const.MATRIX_SIZE,
	type=check_positive,
	help="The size (# of users / goods) used to create a (n x n) matrix for the experiment"
)
parser.add_argument(
	'num_ticks',
	action='store',
	nargs='?',
	default=const.NUMBER_OF_TICKS,
	type=check_positive,
	help="The amount of time / number of loops that the experiment should run for"
)

# Parse arguments and assign to variables defined in controls.py
parse_args = parser.parse_args()
const.NUMBER_OF_EXPERIMENTS = parse_args.num_loops
const.MATRIX_SIZE = parse_args.matrix_size
const.NUMBER_OF_TICKS = parse_args.num_ticks
full_print: bool = parse_args.print if parse_args.blank_print != True else False
blank_print: bool = parse_args.blank_print

if (not blank_print): print(dim_grey + f"SEED [{const.SEED}]" + reset)

# Collect important statistics each run
all_test_data: list[Test] = []
review_matrices = []
utility_matrices = []
new_user_review_matrices = []
new_user_utility_matrices = []



for i in range(const.NUMBER_OF_EXPERIMENTS):
	if (full_print and i != 0): print("\n" + "─"*150 + "\n") # Print a thick dividing line between experimental printouts
	if (not blank_print): print(dim_cyan + f"\n{i+1}) " + reset, end='') # Display the current experiment number [1 to const.NUMBER_OF_EXPERIMENTS]

	all_test_data.append(Test()) # Record test data for final averages

	###################################################################################################
	################################# Generate Base Population Matrix #################################
	###################################################################################################

	# Initialize matrices
	utility_matrix = mtx.UtilMatrix()
	review_matrix = mtx.RevMatrix(utility_matrix.matrix)

	### Run simulation:
	for tick in range(const.NUMBER_OF_TICKS):
		# Print out current tick information
		if (not blank_print): print(dim_cyan + ("TICK #1..." if tick == 0 else f"{tick+1}...") + reset, end=' ', flush=True)
		
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

			while person.budget >= const.CONSIDERATION_COST + const.USAGE_COST:
				# Recommend the user a good while they still have a budget to consume
				recommendations.append(good.recommend_good(copy_of_person, copy_of_review_matrix, previously_recommended=recommendations))

				# If a valid recommendation is possible
				if const.MATRIX_SIZE not in recommendations:

					# Considering a good has an opportunity cost that affects the budget
					person.budget -= const.CONSIDERATION_COST

					# Calculate the user's Expected Utility from consuming the recommended good
					true_utility = utility_matrix.matrix[i, recommendations[-1]]
					noise = random.normalvariate(0, const.UTILITY_STD)
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

	if (not blank_print): print()
	##### Format of matrix printout #####
	# _Name's reviews: [ ·  ·  ·  ·  · ... ·  ·  ·  ·  · ] = Sum_of_Reviews → Generated_Utility (Optimal_Utility)
	if (full_print): print("".join([str(per) + f" = {green_or_red(int(np.nansum(per.reviews)), int(np.nansum(per.reviews)) > (0 if const.RATING_SYSTEM_SCALE == 0 else (~np.isnan(per.reviews)).sum(0) * (((const.RATING_SYSTEM_SCALE + 1) / 2) if (const.RATING_SYSTEM_MEAN < 0 or const.RATING_SYSTEM_MEAN > const.RATING_SYSTEM_SCALE) else const.RATING_SYSTEM_MEAN)), int(np.nansum(per.reviews)) < (0 if const.RATING_SYSTEM_SCALE == 0 else (~np.isnan(per.reviews)).sum(0) * (((const.RATING_SYSTEM_SCALE + 1) / 2) if (const.RATING_SYSTEM_MEAN < 0 or const.RATING_SYSTEM_MEAN > const.RATING_SYSTEM_SCALE) else const.RATING_SYSTEM_MEAN)), space_before=(const.RATING_SYSTEM_SCALE == 0), space_after=(const.RATING_SYSTEM_SCALE == 0))}{reset} → {green_or_red(round(per.generated_utility, 1), per.generated_utility >= (analysis.find_optimal_utility(per) * const.WELL_SERVED_PERCENT))}{reset} ({round(analysis.find_optimal_utility(per), 1)})" for per in review_matrix.pop.people]))
	if (full_print): print()

	# Save matrices for CSV printout
	review_matrices.append(review_matrix.matrix)
	utility_matrices.append(utility_matrix.matrix)

	# Optimal / actual utility, and percentage of optimal utility achieved
	all_test_data[-1].max_user_util = sum(list(map(analysis.find_optimal_utility, review_matrix.pop.people)))
	all_test_data[-1].actual_user_util = sum(per.generated_utility for per in review_matrix.pop.people)
	if (full_print):
		print(f"Maximum User Utility = {cyan}{round(all_test_data[-1].max_user_util, 1)}{reset}")
		print(f"Actual User Utility = {cyan}{round(all_test_data[-1].actual_user_util, 1)}{reset}")
		print(f"{underline_cyan}{round((all_test_data[-1].actual_user_util / all_test_data[-1].max_user_util) * 100)}%{reset} of optimal utility was achieved\n")

	# Percentage of the population being "well served"
	all_test_data[-1].num_well_served = sum(analysis.is_well_served(per, const.WELL_SERVED_PERCENT) for per in review_matrix.pop.people)
	if (full_print): print(f"{cyan}{round((all_test_data[-1].num_well_served / const.MATRIX_SIZE) * 100)}%{reset} of the population was \"well served\" (achieved {round(const.WELL_SERVED_PERCENT * 100)}% of max utility)\n")

	# Define analysis constants
	analyze_quarter_of_goods = (const.ANALYZE_N_GOODS / 4).__ceil__()
	analyze_half_of_goods = (const.ANALYZE_N_GOODS / 2).__ceil__()

	# Percentage of users who consumed all of the n most popular goods
	all_test_data[-1].used_most_pop_one = sum(analysis.all_most_popular_recommended(per, review_matrix.matrix, 1) for per in review_matrix.pop.people)
	all_test_data[-1].gained_pos_util_one = sum(filter(lambda x: x != -1, list(analysis.all_most_popular_recommended_derived_pos_util(per, review_matrix.matrix, 1) for per in review_matrix.pop.people)))
	all_test_data[-1].used_most_pop_quarter = sum(analysis.all_most_popular_recommended(per, review_matrix.matrix, analyze_quarter_of_goods) for per in review_matrix.pop.people)
	all_test_data[-1].gained_pos_util_quarter = sum(filter(lambda x: x != -1, list(analysis.all_most_popular_recommended_derived_pos_util(per, review_matrix.matrix, analyze_quarter_of_goods) for per in review_matrix.pop.people)))
	all_test_data[-1].used_most_pop_half = sum(analysis.all_most_popular_recommended(per, review_matrix.matrix, analyze_half_of_goods) for per in review_matrix.pop.people)
	all_test_data[-1].gained_pos_util_half = sum(filter(lambda x: x != -1, list(analysis.all_most_popular_recommended_derived_pos_util(per, review_matrix.matrix, analyze_half_of_goods) for per in review_matrix.pop.people)))
	if (full_print):
		print(f"{cyan}{round((all_test_data[-1].used_most_pop_one / const.MATRIX_SIZE) * 100)}%{reset} of users used the most popular good {f"({cyan}{round((all_test_data[-1].gained_pos_util_one / all_test_data[-1].used_most_pop_one) * 100)}%{reset} of those users gained positive utility from it)" if all_test_data[-1].used_most_pop_one != 0 else ""}")
		print(f"{cyan}{round((all_test_data[-1].used_most_pop_quarter / const.MATRIX_SIZE) * 100)}%{reset} of users used the {analyze_quarter_of_goods} most popular goods {f"({cyan}{round((all_test_data[-1].gained_pos_util_quarter / all_test_data[-1].used_most_pop_quarter) * 100)}%{reset} of those users gained positive utility from all of them)" if all_test_data[-1].used_most_pop_quarter != 0 else ""}")
		print(f"{cyan}{round((all_test_data[-1].used_most_pop_half / const.MATRIX_SIZE) * 100)}%{reset} of users used the {analyze_half_of_goods} most popular goods {f"({cyan}{round((all_test_data[-1].gained_pos_util_half / all_test_data[-1].used_most_pop_half) * 100)}%{reset} of those users gained positive utility from all of them)" if all_test_data[-1].used_most_pop_half != 0 else ""}")





	###################################################################################################
	############################# Generate Optimal Recommendations Matrix #############################
	###################################################################################################
	if (full_print): print(dim_cyan + "\nGenerating optimal utility matrix...\n" + reset)

	# Make a matrix containing the optimal good consumption for each user
	optimal_matrix = mtx.RevMatrix(utility_matrix.matrix)

	for person in optimal_matrix.pop.people:
		# Get the indexes of the goods that give the user the most utility
		recommendations = heapq.nlargest(const.NUMBER_OF_TICKS * (const.USER_BUDGET // (const.CONSIDERATION_COST + const.USAGE_COST)), range(const.MATRIX_SIZE), key=person.utility.__getitem__)

		for good_idx in recommendations:
			# Rate the good after the user has consumed it
			person.reviews[good_idx] = good.give_rating(person.utility[good_idx])

			# Update the user's utility generated from consuming the good
			person.generated_utility += person.utility[good_idx]

	# Run above analysis questions on a "perfect" model
	all_test_data[-1].optimal_used_most_pop_one = sum(analysis.all_most_popular_recommended(per, optimal_matrix.matrix, 1) for per in optimal_matrix.pop.people)
	all_test_data[-1].optimal_used_most_pop_quarter = sum(analysis.all_most_popular_recommended(per, optimal_matrix.matrix, analyze_quarter_of_goods) for per in optimal_matrix.pop.people)
	all_test_data[-1].optimal_used_most_pop_half = sum(analysis.all_most_popular_recommended(per, optimal_matrix.matrix, analyze_half_of_goods) for per in optimal_matrix.pop.people)
	if (full_print):
		print(f"{cyan}{round((all_test_data[-1].optimal_used_most_pop_one / const.MATRIX_SIZE) * 100)}%{reset} of optimal users used the most popular good")
		print(f"{cyan}{round((all_test_data[-1].optimal_used_most_pop_quarter / const.MATRIX_SIZE) * 100)}%{reset} of optimal users used the {analyze_quarter_of_goods} most popular goods")
		print(f"{cyan}{round((all_test_data[-1].optimal_used_most_pop_half / const.MATRIX_SIZE) * 100)}%{reset} of optimal users used the {analyze_half_of_goods} most popular goods")





	if (full_print): print("\n" + "-"*110 + "\n")  # Print a dividing line between experimental printouts
	###################################################################################################
	################################ Generate New User Recommendations ################################
	###################################################################################################
	new_users = []
	most_popular_goods_consumed = []
	most_pop_rec_comp_to_one = []
	least_pop_rec_comp_to_one = []
	most_pop_rec_comp_to_half = []
	least_pop_rec_comp_to_half = []
	most_pop_rec_comp_to_full = []
	least_pop_rec_comp_to_full = []
	optimal_utilities = []
	popular_utilities = []

	new_user_count = int(const.ANALYZE_N_GOODS / 2)
	if (full_print): print(Style.underline + f"Generating {new_user_count} new users:" + reset)

	for i in range(new_user_count):
		# Add a new user to the system
		new_users.append(ppl.Person(f"_User"))
		utility_matrix.add_user(new_users[-1])
		review_matrix.add_user(new_users[-1])

		# Save usable recommendations so that their frequency can be calculated later
		total_recommendations = []

		### Run simulation:
		if (full_print): print(dim_cyan + f"\n{i+1}) " + reset, end='')
		for tick in range(const.NUMBER_OF_TICKS):
			# Print out current tick information
			if (full_print): print(dim_cyan + ("TICK #1..." if tick == 0 else f"{tick+1}...") + reset, end=' ')

			# Check that there are still possible recommendations
			if np.count_nonzero(np.isnan(review_matrix.matrix)) == 0:
				break

			recommendations = []

			while new_users[-1].budget >= const.CONSIDERATION_COST + const.USAGE_COST:
				# Recommend the new user a good while they still have a budget to consume
				recommendations.append(good.recommend_good(new_users[-1], review_matrix, previously_recommended=recommendations))

				# If a valid recommendation is possible
				if const.MATRIX_SIZE not in recommendations:

					# Considering a good has an opportunity cost that affects the budget
					new_users[-1].budget -= const.CONSIDERATION_COST

					# Calculate the new user's Expected Utility from consuming the recommended good
					true_utility = utility_matrix.matrix[-1, recommendations[-1]]
					noise = random.normalvariate(0, const.UTILITY_STD)
					expected_utility = true_utility + noise

					# Save all recommendations that have a chance to be consumed by the user
					if new_users[-1].budget >= const.USAGE_COST:
						total_recommendations.append(recommendations[-1])

					if new_users[-1].budget >= const.USAGE_COST and expected_utility >= const.UTILITY_MEAN:
						# Consuming the good has a cost that affects the budget
						new_users[-1].budget -= const.USAGE_COST

						# Rate the good after the new user has consumed it
						new_users[-1].reviews[recommendations[-1]] = good.give_rating(true_utility)

						# Update the new user's utility generated from consuming the good
						new_users[-1].generated_utility += true_utility

				# If a valid recommendation is not possible (the new user has consumed all goods)
				else:
					break

			# After current set of recommendations done:
			new_users[-1].reset_budget()

		# Run final tests on individual new users and store to find averages later
		most_popular_goods_consumed.append(analysis.num_most_popular_recommended(new_users[-1], review_matrix.matrix, const.ANALYZE_N_GOODS))
		most_pop_rec_comp_to_one.append(sum(good in analysis.find_most_popular(review_matrix.matrix[:-1], 1) for good in total_recommendations))
		least_pop_rec_comp_to_one.append(sum(good in analysis.find_least_popular(review_matrix.matrix[:-1], 1) for good in total_recommendations))
		most_pop_rec_comp_to_half.append(sum(good in analysis.find_most_popular(review_matrix.matrix[:-1], (const.ANALYZE_N_GOODS / 2).__floor__()) for good in total_recommendations))
		least_pop_rec_comp_to_half.append(sum(good in analysis.find_least_popular(review_matrix.matrix[:-1], (const.ANALYZE_N_GOODS / 2).__floor__()) for good in total_recommendations))
		most_pop_rec_comp_to_full.append(sum(good in analysis.find_most_popular(review_matrix.matrix[:-1], const.ANALYZE_N_GOODS) for good in total_recommendations))
		least_pop_rec_comp_to_full.append(sum(good in analysis.find_least_popular(review_matrix.matrix[:-1], const.ANALYZE_N_GOODS) for good in total_recommendations))
		optimal_utilities.append(analysis.find_optimal_utility(new_users[i]))
		popular_utilities.append(analysis.find_popular_utility(new_users[i], review_matrix.matrix))

		# Save matrices for CSV printout
		new_user_review_matrices.append(new_users[-1].reviews)
		new_user_utility_matrices.append(new_users[-1].utility)

		# Remove the previous user
		utility_matrix.remove_user(new_users[-1], review_matrix.pop)
		review_matrix.remove_user(new_users[-1])

		# Display user
		if (full_print): print(str(new_users[-1]) + "\n")
	if (full_print): print()




	###################################################################################################
	############################### Run Final Tests On Average New User ###############################
	###################################################################################################

	### Percentage of the n most popular goods consumed by the new user
	all_test_data[-1].most_pop_used = sum(most_popular_goods_consumed) / len(most_popular_goods_consumed)
	if (full_print): print(f"On average, {cyan}{round((all_test_data[-1].most_pop_used / const.ANALYZE_N_GOODS) * 100)}%{reset} of the {const.ANALYZE_N_GOODS} most popular goods were used")

	### Number of n popular recommendations compared to n unpopular recommendations
	all_test_data[-1].one_most_pop_good_rec = round(sum(most_pop_rec_comp_to_one) / len(most_pop_rec_comp_to_one)); all_test_data[-1].one_least_pop_good_rec = round(sum(least_pop_rec_comp_to_one) / len(least_pop_rec_comp_to_one))
	all_test_data[-1].half_most_pop_good_rec = round(sum(most_pop_rec_comp_to_half) / len(most_pop_rec_comp_to_half)); all_test_data[-1].half_least_pop_good_rec = round(sum(least_pop_rec_comp_to_half) / len(least_pop_rec_comp_to_half))
	all_test_data[-1].full_most_pop_good_rec = round(sum(most_pop_rec_comp_to_full) / len(most_pop_rec_comp_to_full)); all_test_data[-1].full_least_pop_good_rec = round(sum(least_pop_rec_comp_to_full) / len(least_pop_rec_comp_to_full))
	if (full_print):
		print("On average,", analysis.likely_recommended(all_test_data[-1].one_most_pop_good_rec, all_test_data[-1].one_least_pop_good_rec).lower())
		print("On average,", analysis.likely_recommended(all_test_data[-1].half_most_pop_good_rec, all_test_data[-1].half_least_pop_good_rec, (const.ANALYZE_N_GOODS / 2).__floor__()).lower())
		print("On average,", analysis.likely_recommended(all_test_data[-1].full_most_pop_good_rec, all_test_data[-1].full_least_pop_good_rec, const.ANALYZE_N_GOODS).lower())
		print()

	### User's utility if they consumed only the most optimal goods
	all_test_data[-1].avg_new_user_util = sum(optimal_utilities) / len(optimal_utilities)
	if (full_print): print(f"The average new user's optimal utility was {cyan}{round(all_test_data[-1].avg_new_user_util, 1)}{reset}")

	### User's utility if they consumed only the most popular goods
	all_test_data[-1].avg_most_pop_util = sum(popular_utilities) / len(popular_utilities)
	if (full_print): print(f"The average utility of the most popular goods was {cyan}{round(all_test_data[-1].avg_most_pop_util, 1)}{reset}")

	### Actual user utility
	all_test_data[-1].avg_actual_util = sum(user.generated_utility for user in new_users) / new_user_count
	if (full_print): print(f"The average user's actual utility was {underline_cyan}{round(all_test_data[-1].avg_actual_util, 1)}{reset}")





	###################################################################################################
	##################################### Generate Random Reviews #####################################
	###################################################################################################
	rand_users = []

	for i in range(new_user_count):
		rand_users.append(ppl.Person("_Rand"))
		rand_users[i].utility = new_users[i].utility

		# Randomly select goods for the random user to consume
		rand_recommendations = random.sample(range(const.MATRIX_SIZE), sum(~np.isnan(new_users[i].reviews)))

		for good_idx in rand_recommendations:
			# Rate the good after the user has consumed it
			rand_users[i].reviews[good_idx] = good.give_rating(rand_users[i].utility[good_idx])

			# Update the user's utility generated from consuming the good
			rand_users[i].generated_utility += rand_users[i].utility[good_idx]

	### Compare the new user to a user who randomly selects goods to consume
	all_test_data[-1].avg_rand_util = sum(user.generated_utility for user in rand_users) / new_user_count
	if (full_print): print(f"\nThe average new user who uses the recommender system generates {cyan}{round(all_test_data[-1].avg_actual_util - all_test_data[-1].avg_rand_util, 1)}{reset} ({underline_cyan}{round((all_test_data[-1].avg_actual_util - all_test_data[-1].avg_rand_util) / const.UTILITY_MEAN, 1)}x{reset} mean) more utility compared to an average user who chooses goods randomly\n")





###################################################################################################
######################################## Run Average Tests ########################################
###################################################################################################

# Print completion message
if (not full_print): print()
if (not blank_print): print("═"*150 + "\n") # Print a double thick dividing line after all experiments have completed
print(bold_cyan + f"All individual experiments have completed running!\n" + reset)



### PRINT AVERAGES IN SAME FORMAT AS INDIVIDUAL ###
if (not blank_print):
	print(Style.italic + "Across all experiments, on average...\n" + reset)
	print(f"Maximum User Utility = {cyan}{round(sum(test.max_user_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS, 1)}{reset}")
	print(f"Actual User Utility = {cyan}{round(sum(test.actual_user_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS, 1)}{reset}")
	print(f"{underline_cyan}{round((sum(test.actual_user_util for test in all_test_data) / sum(test.max_user_util for test in all_test_data)) * 100)}%{reset} of optimal utility was achieved\n")

	print(f"{cyan}{round((sum(test.num_well_served for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)) * 100)}%{reset} of the population was \"well served\" (achieved {round(const.WELL_SERVED_PERCENT * 100)}% of max utility)\n")

	print(f"{cyan}{round((sum(test.used_most_pop_one for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)) * 100)}%{reset} of users used the most popular good {f"({cyan}{round((sum(test.gained_pos_util_one for test in all_test_data) / sum(test.used_most_pop_one for test in all_test_data)) * 100)}%{reset} of those users gained positive utility from it)" if sum(test.used_most_pop_one for test in all_test_data) != 0 else ""}")
	print(f"{cyan}{round((sum(test.used_most_pop_quarter for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)) * 100)}%{reset} of users used the {analyze_quarter_of_goods} most popular goods {f"({cyan}{round((sum(test.gained_pos_util_quarter for test in all_test_data) / sum(test.used_most_pop_quarter for test in all_test_data)) * 100)}%{reset} of those users gained positive utility from all of them)" if sum(test.used_most_pop_quarter for test in all_test_data) != 0 else ""}")
	print(f"{cyan}{round((sum(test.used_most_pop_half for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)) * 100)}%{reset} of users used the {analyze_half_of_goods} most popular goods {f"({cyan}{round((sum(test.gained_pos_util_half for test in all_test_data) / sum(test.used_most_pop_half for test in all_test_data)) * 100)}%{reset} of those users gained positive utility from all of them)" if sum(test.used_most_pop_half for test in all_test_data) != 0 else ""}")

	print(dim_cyan + "\nBased on optimal utility matrices..." + reset)
	print(f"{cyan}{round((sum(test.optimal_used_most_pop_one for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)) * 100)}%{reset} of optimal users used the most popular good")
	print(f"{cyan}{round((sum(test.optimal_used_most_pop_quarter for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)) * 100)}%{reset} of optimal users used the {analyze_quarter_of_goods} most popular goods")
	print(f"{cyan}{round((sum(test.optimal_used_most_pop_half for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)) * 100)}%{reset} of optimal users used the {analyze_half_of_goods} most popular goods")

	print(dim_cyan + "\nBased on the average new user..." + reset)
	print(f"{cyan}{round((sum(test.most_pop_used for test in all_test_data) / (const.ANALYZE_N_GOODS * const.NUMBER_OF_EXPERIMENTS)) * 100)}%{reset} of the {const.ANALYZE_N_GOODS} most popular goods were used")
	print(analysis.likely_recommended(round(sum(test.one_most_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS), round(sum(test.one_least_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS)))
	print(analysis.likely_recommended(round(sum(test.half_most_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS), round(sum(test.half_least_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS), (const.ANALYZE_N_GOODS / 2).__floor__()))
	print(analysis.likely_recommended(round(sum(test.full_most_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS), round(sum(test.full_least_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS), const.ANALYZE_N_GOODS))

	print(f"\nThe average new user's optimal utility was {cyan}{round(sum(test.avg_new_user_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS, 1)}{reset}")
	print(f"The average utility of the most popular goods was {cyan}{round(sum(test.avg_most_pop_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS, 1)}{reset}")
	print(f"The average user's actual utility was {underline_cyan}{round(sum(test.avg_actual_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS, 1)}{reset}")

	print(f"\nThe average new user who uses the recommender system generates {cyan}{round((sum(test.avg_actual_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS) - (sum(test.avg_rand_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS), 1)}{reset} ({underline_cyan}{round(((sum(test.avg_actual_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS) - (sum(test.avg_rand_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS)) / const.UTILITY_MEAN, 1)}x{reset} mean) more utility compared to an average user who chooses goods randomly\n")





###################################################################################################
#################################### Save Analysis Data to CSV ####################################
###################################################################################################

# Create folder to hold CSV data
csv_dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSVs'))
if not os.path.exists(csv_dirpath): os.makedirs(csv_dirpath)

# Create folder to hold specific run data
run_name = "research_[" + str(const.NUMBER_OF_EXPERIMENTS) + "][" + str(const.MATRIX_SIZE) + "][" + str(const.NUMBER_OF_TICKS) + "]"
run_dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSVs', run_name))
if not os.path.exists(run_dirpath): os.makedirs(run_dirpath)
else:
	i = 1
	# Keep searching for a valid (no-duplicate) name
	while (os.path.exists(run_dirpath + f"_{i}")):
		i += 1
	run_name = run_name + f"_{i}"
	os.makedirs(run_dirpath + f"_{i}")

# Create folder to hold test matrices
matrix_dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSVs', run_name, 'matrices'))
os.makedirs(matrix_dirpath)

# Save review matrices
for i in range(const.NUMBER_OF_EXPERIMENTS):
	with open(os.path.join('CSVs', run_name, 'matrices', f'test[{i}]_review.csv'), 'w', newline='') as csvFile:
		wr = csv.writer(csvFile)
		# Save main review matrix
		wr.writerows(review_matrices[i])
		wr.writerow("")
		# Save all new user review matrices
		for j in range(int(const.ANALYZE_N_GOODS / 2)):
			wr.writerow(new_user_review_matrices[(i * int(const.ANALYZE_N_GOODS / 2)) + j])

# Save utility matrices
for i in range(const.NUMBER_OF_EXPERIMENTS):
	with open(os.path.join('CSVs', run_name, 'matrices', f'test[{i}]_utility.csv'), 'w', newline='') as csvFile:
		wr = csv.writer(csvFile)
		# Save main utility matrix
		wr.writerows(utility_matrices[i])
		wr.writerow("")
		# Save all new user utility matrices
		for j in range(int(const.ANALYZE_N_GOODS / 2)):
			wr.writerow(new_user_utility_matrices[(i * int(const.ANALYZE_N_GOODS / 2)) + j])

# Create control container
controls = [['MATRIX_SIZE', const.MATRIX_SIZE],
['NUMBER_OF_TICKS', const.NUMBER_OF_TICKS],
['NUMBER_OF_EXPERIMENTS', const.NUMBER_OF_EXPERIMENTS],
['UTILITY_MEAN', const.UTILITY_MEAN],
['UTILITY_STD', const.UTILITY_STD],
['USER_BUDGET', const.USER_BUDGET],
['CONSIDERATION_COST', const.CONSIDERATION_COST],
['USAGE_COST', const.USAGE_COST],
['RATING_SYSTEM_SCALE', const.RATING_SYSTEM_SCALE],
['RATING_SYSTEM_MEAN', const.RATING_SYSTEM_MEAN],
['RATING_SYSTEM_STD', const.RATING_SYSTEM_STD],
['ANALYZE_N_GOODS', const.ANALYZE_N_GOODS],
['WELL_SERVED_PERCENT', const.WELL_SERVED_PERCENT],
['SEED', const.SEED]]

# Save controls
with open(os.path.join('CSVs', run_name, 'controls.csv'), 'w', newline='') as csvFile:
	wr = csv.writer(csvFile)
	wr.writerows(controls)

# Create labels for CSV file
test_labels = [f"Experiment #{i+1}" for i in range(const.NUMBER_OF_EXPERIMENTS)]

# Save analysis data
with open(os.path.join('CSVs', run_name, 'analysis.csv'), 'w', newline='') as csvFile:
	wr = csv.writer(csvFile)
	wr.writerow([''] + test_labels + ['Average'])

	wr.writerow(['Max User Utility'] + [test.max_user_util for test in all_test_data] + [sum(test.max_user_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow(['Actual User Utility'] + [test.actual_user_util for test in all_test_data] + [sum(test.actual_user_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow(['% Optimal Utility Achieved'] + [(test.actual_user_util / test.max_user_util) for test in all_test_data] + [sum(test.actual_user_util for test in all_test_data) / sum(test.max_user_util for test in all_test_data)])

	wr.writerow(['% Well Served'] + [(test.num_well_served / const.MATRIX_SIZE) for test in all_test_data] + [sum(test.num_well_served for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)])

	wr.writerow(['% Used Most Popular Good'] + [(test.used_most_pop_one / const.MATRIX_SIZE) for test in all_test_data] + [sum(test.used_most_pop_one for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)])
	wr.writerow(['% of those users who gained positive utility from it'] + [(0 if test.used_most_pop_one == 0 else (test.gained_pos_util_one / test.used_most_pop_one)) for test in all_test_data] + [(0 if sum(test.used_most_pop_one for test in all_test_data) == 0 else sum(test.gained_pos_util_one for test in all_test_data) / sum(test.used_most_pop_one for test in all_test_data))])
	wr.writerow([f'% Used {analyze_quarter_of_goods} Most Popular Goods'] + [(test.used_most_pop_quarter / const.MATRIX_SIZE) for test in all_test_data] + [sum(test.used_most_pop_quarter for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)])
	wr.writerow(['% of those users who gained positive utility from all of them'] + [(0 if test.used_most_pop_quarter == 0 else (test.gained_pos_util_quarter / test.used_most_pop_quarter)) for test in all_test_data] + [(0 if sum(test.used_most_pop_quarter for test in all_test_data) == 0 else sum(test.gained_pos_util_quarter for test in all_test_data) / sum(test.used_most_pop_quarter for test in all_test_data))])
	wr.writerow([f'% Used {analyze_half_of_goods} Most Popular Goods'] + [(test.used_most_pop_half / const.MATRIX_SIZE) for test in all_test_data] + [sum(test.used_most_pop_half for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)])
	wr.writerow(['% of those users who gained positive utility from all of them']  + [(0 if test.used_most_pop_half == 0 else (test.gained_pos_util_half / test.used_most_pop_half)) for test in all_test_data] + [(0 if sum(test.used_most_pop_half for test in all_test_data) == 0 else sum(test.gained_pos_util_half for test in all_test_data) / sum(test.used_most_pop_half for test in all_test_data))])

	wr.writerow(['% Optimal Users Used Most Popular Goods'] + [(test.optimal_used_most_pop_one / const.MATRIX_SIZE) for test in all_test_data] + [sum(test.optimal_used_most_pop_one for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)])
	wr.writerow(['% Optimal Users Used {analyze_quarter_of_goods} Most Popular Good'] + [(test.optimal_used_most_pop_quarter / const.MATRIX_SIZE) for test in all_test_data] + [sum(test.optimal_used_most_pop_quarter for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)])
	wr.writerow(['% Optimal Users Used {analyze_half_of_goods} Most Popular Goods'] + [(test.optimal_used_most_pop_half / const.MATRIX_SIZE) for test in all_test_data] + [sum(test.optimal_used_most_pop_half for test in all_test_data) / (const.MATRIX_SIZE * const.NUMBER_OF_EXPERIMENTS)])

	wr.writerow([f'% of {const.ANALYZE_N_GOODS} Most Popular Goods Consumed by Average New User'] + [(test.most_pop_used / const.ANALYZE_N_GOODS) for test in all_test_data] + [sum(test.most_pop_used for test in all_test_data) / (const.ANALYZE_N_GOODS * const.NUMBER_OF_EXPERIMENTS)])
	wr.writerow(['Times Most Popular Good was Recommended'] + [test.one_most_pop_good_rec for test in all_test_data] + [sum(test.one_most_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow(['Times Least Popular Good was Recommended'] + [test.one_least_pop_good_rec for test in all_test_data] + [sum(test.one_least_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow([f'Times {(const.ANALYZE_N_GOODS / 2).__floor__()} Most Popular Goods were Recommended'] + [test.half_most_pop_good_rec for test in all_test_data] + [sum(test.half_most_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow([f'Times {(const.ANALYZE_N_GOODS / 2).__floor__()} Least Popular Goods were Recommended'] + [test.half_least_pop_good_rec for test in all_test_data] + [sum(test.half_least_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow([f'Times {const.ANALYZE_N_GOODS} Most Popular Goods were Recommended'] + [test.full_most_pop_good_rec for test in all_test_data] + [sum(test.full_most_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow([f'Times {const.ANALYZE_N_GOODS} Least Popular Goods were Recommended'] + [test.full_least_pop_good_rec for test in all_test_data] + [sum(test.full_least_pop_good_rec for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])

	wr.writerow(['Average New User Optimal Utility'] + [test.avg_new_user_util for test in all_test_data] + [sum(test.avg_new_user_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow(['Average New User Popular Good Utility'] + [test.avg_most_pop_util for test in all_test_data] + [sum(test.avg_most_pop_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])
	wr.writerow(['Average New User Actual Utility'] + [test.avg_actual_util for test in all_test_data] + [sum(test.avg_actual_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS])

	wr.writerow(['Average Difference in New User vs Random User Actual Utility'] + [(test.avg_actual_util - test.avg_rand_util)for test in all_test_data] + [(sum(test.avg_actual_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS) - (sum(test.avg_rand_util for test in all_test_data) / const.NUMBER_OF_EXPERIMENTS)])

print(bold_cyan + f"All data has been stored in {underline_cyan}{os.path.join('CSVs', run_name)}{reset}{cyan} !\n" + reset)
