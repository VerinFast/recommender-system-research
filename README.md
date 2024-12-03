# Quantifying the Effect of Preference Expression on Utility Delivered by Recommender Systems

This program is meant to showcase the possible lossy inefficiencies found in recommender systems as part of the research being conducted by **Cole Golding** and **Jason Nichols**. Our hypothesis states that recommender systems converge on the popular at the expense of the novel (given enough input they will be less likely to recommend new and more likely to recommend old). To prove this, we first generate a "review matrix" (as described below) using our recommender system. Then, we create a number of new users whose reviews are based off of this pre-established model. By analyzing the percentage of popular goods recommended to these new users, we hope to prove our hypothesis.

## Background and Experimental Procedure
These experiments are performed by first creating an n-by-n matrix of reviews, and a corresponding n-by-n matrix of utility values [^utility]. These utility values correspond to the [*true utility*](https://www.investopedia.com/terms/u/utility.asp) each user gains upon consuming each good. This model does not utilize contextual marginal utility and instead uses a non-contextual total utility i.e. no matter when a user consumes a good they gain the same amount of actual utility from the good. These actual values are unknown to both the user and recommender though, so they are instead used to calculate an obfuscated value that the user treats as their *expected utility* (the utility they *think* they will gain by consuming the good) within our model. The user also has a budget which is depleted as they consume goods, and a set number of time (represented as "ticks") to consume goods over.

Each tick, the recommender goes user by user and checks if they have enough remaining budget to search for a recommendation and to then consume the recommended good (all recommendation searches and good consumptions have the same "price" for all users). If they have enough remaining budget, the system finds a list of users who share similar reviews to the current user, and picks the good shared by those users with the most positive reviews to give as a recommendation to the current user. The user then looks at the good, and based on their perceived expected utility from the good, chooses to consume it or search for another good instead. After consuming a good, the user leaves a review (which is stored in their review matrix) based on their actual utility gained from consuming the good. This is then repeated for every user `NUMBER_OF_TICKS` times [^budget].

After all reviews have been given, a number of questions are then analyzed about the review and utility matrices:
- What was the total maximum user utility?
- What was the total actual user utility?
  - What percentage of optimal utility was actually received?
- What percentage of the population was "well served" [^well_served]?
- What percentage of users used the n most popular goods?
  - Out of those users, how many gained positive utility from all of them?
- What percentage of "optimal" users used the n most popular goods?

Next, new users are generated and are recommended goods based on the n-by-n review matrix. The results are then used to answer a number of other questions including:
- What percentage of the n most popular goods were used by each new user?
- How many more times were the n most popular goods recommended over the n least popular goods?
- What was the average new user's optimal utility?
- What was the average utility of the most popular goods?
- What was the average user's actual utility?
- What was the difference in actual utility between the average new user vs a user who chooses goods randomly?

## Setup
Download the GitHub repo [here](https://github.com/VerinFast/recommender-system-research) to begin.
  
The experiment can be run calling any of the following terminal command formats from the root folder of the project:
```
python research/main.py
python research/main.py [num_loops]
python research/main.py [num_loops] [matrix_size]
python research/main.py [num_loops] [matrix_size] [num_ticks]
```

For additional information on the above arguments as well as additional tags, use the `-h` tag for help:
```
python research/main.py -h
```

A complete control board can be found within the [`research/recommender_system/controls.py`](research/recommender_system/controls.py) file, allowing for more granular control over experimental weights and analysis procedure. Certain default values have been designed for small example use and do not represent statistically significant findings. Additionally, some default values (listed below) stored in the controls.py file are temporarily replaced if the corresponding command line parameter is used when calling the experiment:
| controls.py Variable | Command Line Parameter |
| :------------------- | :--------------------: |
| NUMBER_OF_EXPERIMENTS | num_loops |
| MATRIX_SIZE | matrix_size |
| NUMBER_OF_TICKS | num_ticks |

## Output
All experimental analysis data is stored within the `CSVs` folder (which is generated upon running the program if it does not already exist). Inside this folder, individual folders for each run will be stored, labelled as `research_[num_loops][matrix_size][num_ticks]` e.g. for the defaults found in controls.py the corresponding analysis data can be found in: `research[10][20][10]`. If multiple commands have been run using the same core constants, it will add a trailing `_i` (where i counts up from 1) to the folder name. Inside this research folder, one can find .csv files containing all analysis information for each individual run as well as the constants used by the experiment. Additionally, a folder named `matrices` contains the review and utility matrices used in each individual run.

Colored analysis data (as seen below) is also printed to the terminal by default, and one can watch the progress of each individual test through the tick counters. A message is displayed when all individual experiments have finished running, and then a summarized display of all major collected data is displayed for the average run. Finally, the name of the folder stored in `CSVs` is displayed. The terminal output can be modified using both the `-p` and `-b` tags. Using `-p` will print each individual run's review matrices as well as a data display for that specific run. Using `-b` will remove all tick and analysis display printouts and will simply print when the experiments have completed and the location of the stored data within `CSVs`.

> Standard printout options can be seen below when **(Left)** using default commands (`python research/main.py`), **(Top Right)** using print tag and default commands (`python research/main.py -p`) [^printout], and **(Bottom Right)** using blank_print tag and default commands (`python research/main.py -b`).
> 
> ![Standard printout using -b tag ad default commands](https://u.cubeupload.com/cgolding/ExperimentalDisplayO.png)

## License
[GNU General Public License v3.0](https://www.gnu.org/licenses/)

For more information, use the `-h`, `-w`, and `-c` tags.

[^budget]: The user's budget and list of recommended goods are reset at the end of each tick. This can lead to the system repeatedly attempting to recommend a highly rated, popular good to the user due to it "forgetting" the user already was recommended the good and turned it down.
[^utility]: The utility values are randomly selected for each good using a normal distribution across each user.
[^well_served]: "Well Served" is defined as whether or not a given user's actual utility was greater than or equal to a percentage of the user's optimal utility. The optimal utility for a user is based on how many goods they actually consumed and not how many they could have consumed.
[^printout]: The Top Right printout is for a single experiment within the program. Within the review matrix printout, each user's reviews are displayed in the following format:  
`_Name's reviews: [ ·  ·  ·  ·  · ... ·  ·  ·  ·  · ] = Sum_of_Reviews → Generated_Utility (Optimal_Utility)`  
The `Sum_of_Reviews` is colored based on whether it is above (green), equal to (white), or below (red) the number of reviews by that user times the average of the review scale. The `Generated_Utility` is colored based on whether the user was "well served" (green) or not (red). The user's possible optimal utility is then displayed in parenthesis.