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

import argparse
from colored import Fore, Style

copyright_statement = f"""
{Style.BOLD}Recommender System Research  Copyright (c) 2024  Cole Golding{Style.reset}
This program comes with ABSOLUTELY NO WARRANTY; for details use the '-w' flag.
This is free software, and you are welcome to redistribute it
under certain conditions; use the '-c' flag for details.
 """

copyright_w = f""" 
{Style.BOLD}Recommender System Research  Copyright (c) 2024  Cole Golding{Style.reset}
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

copyright_c = f""" 
{Style.BOLD}Recommender System Research  Copyright (c) 2024  Cole Golding{Style.reset}
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

program_description = """
This program is meant to showcase the possible lossy inefficiences found in recommender systems.
If no positional arguments are provided at runtime, the program will use the defaults stored in the research/controls.py file.
"""



### Color Formatters ###

reset = f'{Style.reset}'
dim = f'{Style.dim}'

cyan = f'{Fore.cyan}'
green = f'{Fore.green}'
grey = f'{Fore.dark_gray}'
red = f'{Fore.red}'

dim_cyan = f'{dim}{cyan}'
dim_grey = f'{dim}{grey}'
underline_cyan = f'{Style.underline}{cyan}'

def green_or_red(string: str | int | float, green_logic: bool, red_logic: bool = True, space_before: bool = False, space_after: bool = False) -> str:
	return (green + (' ' if space_before == True else '') if green_logic else (Fore.red if red_logic else (' ' if space_after == True else ''))) + str(string)



### argparse stub class and actiontype ###

class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
	"""A stubclass designed to combine two default argparse formatters"""; pass

def check_positive(value) -> int:
	"""Checks whether the provided value is a positive, nonzero integer.

	Args:
			value (unknown): The value to be checked, type is unknown

	Raises:
			argparse.ArgumentTypeError: Raised if the passed value is an integer but is negative or zero
			Exception: Raised if the passed value cannot be converted to an integer

	Returns:
			int: The provided value, given that it is a positive, nonzero integer
	"""
	try:
		value = int(value)
		if value <= 0:
			raise argparse.ArgumentTypeError(red + f"{value} is not a positive integer\n" + reset)
	except ValueError:
		raise argparse.ArgumentTypeError(red + f"\"{value}\" is not a positive integer\n" + reset)
	return value
