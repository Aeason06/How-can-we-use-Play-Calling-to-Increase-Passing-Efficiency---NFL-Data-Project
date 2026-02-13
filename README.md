# How can we use Play Calling to Increase Passing Efficiency - NFL Data Project
@AustinCEason<br>
# Introduction
In this project, I will attempt to determine whether I can use 2025 NFL data from the nfl_data_py library to identify which plays or play features increase passing efficiency. The main modules from nfl_data_py I am using are pbp_data, ftn_data, and team_desc. <br>

pbp_data contains core information about every snap from the season, such as what happened, when it happened, and who was involved. It also contains some advanced stats like EPA, which is one of my personal favorites for evaluating efficiency. ftn_data is a collection of snap data that goes more in-depth about player positioning and offensive play type. Finally, team_desc contains links to team logos that are used to make visuals. <br>

# Code and Data Analysis
Let's start with our imports: <br>
```Python
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from urllib.request import urlopen
from PIL import Image
import io
```
This is everything we will need to interpret the data and make visuals. <br>



