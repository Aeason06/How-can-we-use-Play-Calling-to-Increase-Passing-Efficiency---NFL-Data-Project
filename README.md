# How can we use Play Calling to Increase Passing Efficiency - NFL Data Project
@AustinCEason<br>
# Introduction
In this project, I will attempt to determine whether I can use 2025 NFL data from the nfl_data_py library to identify which plays or play features increase passing efficiency. The main modules from nfl_data_py I am using are pbp_data, ftn_data, and team_desc. <br>

pbp_data contains core information about every snap from the season, such as what happened, when it happened, and who was involved. It also contains some advanced stats like EPA, which is one of my personal favorites for evaluating efficiency. ftn_data is a collection of snap data that goes more in-depth about player positioning and offensive play type. Finally, team_desc contains links to team logos that are used to make visuals. <br>

The main dataset used in this project is very large, it contains around 47,000 rows and 400 columns so most of that information will not be utalized. However, it is still cool data. I attempted to upload it to the repo, but it was too large. <br>

# Code and Data Analysis
Let's start with our imports. This is everything we will need to interpret the data and make visuals. Next, we start collecting and formatting the data we want to use. <br>
```Python
# Imports for data collection and analysis.
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Imports for data viz.
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from urllib.request import urlopen
from PIL import Image
import io
```
We start with importing ftn_data for the 2025 season and renaming some of the columns so we can easily merge it to the pbp_data later. <br>
```Python
df = nfl.import_ftn_data([2025])
df.rename(columns = {'nflverse_game_id':'game_id', 'nflverse_play_id':'play_id'}, inplace = True)
```
Next, we import pbp_data and merge it with ftn_data. It's also necessary to drop the 'season' and 'week' columns prior to merging, so when we merge with ftn_data, no duplicate columns are created. <br>
```Python
pbp = nfl.import_pbp_data(years=[2025])
pbp.drop(columns=['season', 'week'], inplace=True)
df = pd.merge(df, pbp, on=['game_id', 'play_id'], how='inner')

# Save to csv for upload.
df.to_csv('Ultimate 2025 Dataset.csv')
```
After that, we filter to only pass plays because we are trying to examine passing efficiency, and initialize a new dataframe where we can collect the columns we want from df(the huge dataset). We also encode the booleans into integers so they can be interpreted by the model we are using and create 'QB_BD', which we will define as an inefficient decision made by the QB. All rows with Nan values need to be dropped so they don't cause issues when we run the model summary and check VIF. <br>
```Python
# filter to only pass plays
df = df[df['play_type'] == 'pass']

# Initialize main dataframe
main = pd.DataFrame()
main = df[['n_offense_backfield', 'is_motion', 'is_play_action', 'is_screen_pass', 'is_rpo', 'is_trick_play','is_qb_out_of_pocket']].copy()
main['QB_BD'] = (df['is_interception_worthy'] | df['is_qb_fault_sack'] | df['is_throw_away'] | df['is_contested_ball']).astype(int)

# Convert boolean columns to integers
bool_cols = main.select_dtypes(include='bool').columns
main[bool_cols] = df[bool_cols].astype(int)

# Drop any rows with NaN values
main.dropna(inplace=True)
```
CONTINUE EDITING HERE

```Python
# X is our predictors, y is what we are trying to predict.
X = main[['n_offense_backfield', 'is_motion', 'is_play_action', 'is_screen_pass', 'is_rpo', 'is_trick_play','is_qb_out_of_pocket']]
y = main['QB_BD']

X = sm.add_constant(X)  
X = X.astype(float)

vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
```
Output: <br>
```Python
               feature       VIF
0                const  6.356825
1  n_offense_backfield  1.151794
2            is_motion  1.039530
3       is_play_action  1.225280
4       is_screen_pass  1.094153
5               is_rpo  1.073837
6        is_trick_play  1.003492
7  is_qb_out_of_pocket  1.082906
```
