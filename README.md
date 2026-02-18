# How can we use Play Calling to Increase Passing Efficiency - NFL Data Project
@AustinCEason<br>
# Introduction
In this project, I will attempt to determine whether I can use 2025 NFL data from the nfl_data_py library to identify which plays or play features increase passing efficiency. The main modules from nfl_data_py I am using are pbp_data, ftn_data, and team_desc. <br>

pbp_data contains core information about every snap from the season, such as what happened, when it happened, and who was involved. It also contains some advanced stats like EPA, which is one of my personal favorites for evaluating efficiency. ftn_data is a collection of snap data that goes more in-depth about player positioning and offensive play type. Finally, team_desc contains links to team logos that are used to make visuals. <br>

The main dataset used in this project is very large, it contains around 47,000 rows and 400 columns, so most of that information will not be utilized. However, it is still cool data. I attempted to upload it to the repo, but the csv file was too large. <br>

# Code and Data Analysis
Let's start with our imports. This is everything we will need to interpret the data and make visuals. Next, we start collecting and formatting the data we want to use. <br>
```Python
# Imports for data collection and analysis.
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score

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
Here, we define what variables might be good as predictors in X, and then define y as 'QB_BD' because that is what we are trying to predict. It is also time to check VIF, which is a statistic that measures multicolinearity. Looking at the output, everything is below 5, so there is no multicollinearity here. <br>

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
Now it's model time. Start by reinitializing X and y so the features can change them without messing up the VIF output. Then print the model output so it can be interpreted. <br>
```Python
X = main[['n_offense_backfield', 'is_motion', 'is_play_action', 'is_screen_pass', 'is_rpo', 'is_trick_play','is_qb_out_of_pocket']]
y = main['QB_BD']
X = X.astype(float)
X = sm.add_constant(X) 

model = sm.Logit(y, X).fit()
print(model.summary())            
```
Simplified Output: <br>
| Variable                | Coefficient | P-Value  |
|--------------------------|------------|----------|
| Intercept                | -1.2392    | <0.001   |
| n_offense_backfield      | -0.0192    | 0.634    |
| is_motion                | -0.0525    | 0.144    |
| is_play_action           | -0.2524    | <0.001   |
| is_screen_pass           | -1.3646    | <0.001   |
| is_rpo                   | -0.7205    | <0.001   |
| is_trick_play            | 0.3800     | 0.182    |
| is_qb_out_of_pocket      | 0.5972     | <0.001   |

In the output, both 'n_offense_backfield' and 'is_screen_pass' have high negative coefficients. This makes sense considering that screen passes are very safe plays, and adding players to the backfield usually gives the QB more time, which should lead to less inefficient decisions being made. The model has a pseudo R-squared value of 0.02502, which, for a dataset of this size, is not terrible. However, this means that the model is weak for prediction. For more information lets get the ROC AUC. <br>
```Python
y_pred = model.predict(X)
roc_auc_score(y, y_pred)     
```
ROC AUC: 0.5851737452906048 <br>

The above value tells us essentially the same thing as the psuedo R-squared value. The model is usable, but very weak. A strong model would have an ROC AUC higher than 0.7. <br>
Here we load in the data and manipulate it to get the columns we want for visualizations and further interpretation. <br>
```Python
# load data
df = pd.read_csv('/Users/aeason/Desktop/Python files/Personal/NFL/Offensive Efficiency Project/Ultimate 2025 Dataset.csv')
df = df[df['season_type'] == 'REG']

# choose what data we want to use
new = df[['play_type','passer_player_name', 'posteam', 'is_play_action', 'is_screen_pass', 'is_rpo']]
new = new[(new['play_type'] == 'pass') | (new['play_type'] == 'run')]

# encode booleans as ints so we can work with them
bool_cols = new.select_dtypes(include='bool').columns
new[bool_cols] = new[bool_cols].astype(int)

# create a new column for normal drop-back plays
new['Drop Back'] = ((new['is_play_action'] == 0) & 
                  (new['is_screen_pass'] == 0) & 
                  (new['is_rpo'] == 0) &
                  (new['play_type'] == 'pass')).astype(int)

# create a new column for run plays
new['Run'] = (new['play_type'] == 'run').astype(int)

# rename columns to make them easier to interpret
new.rename(columns={'is_play_action':'Play Action', 'is_screen_pass':'Screen Pass',
                   'is_rpo':'RPO'}, inplace=True)
```
Next, I wanted to see if players who play for teams with a lower standard deviation for play call type had a higher passing EPA. To start visualization, we create a heatmap that shows us the top 40 QBs with the most pass attempts and their teams' play call frequencies, along with the standard deviation of those frequencies. <br>
```Python
# Define play type columns
play_type_cols = ['Drop Back', 'Play Action', 'Screen Pass', 'RPO']

# Get total pass attempts per QB
total_attempts = new.groupby('passer_player_name').size()

# Get top 40 QBs by total attempts
top_40_qbs = total_attempts.nlargest(40).index

# Filter to just those 40 QBs and get their play type counts
player_counts = new[new['passer_player_name'].isin(top_40_qbs)].groupby('passer_player_name')[play_type_cols].sum()

# Calculate percentages based on TOTAL attempts
player_percentages = player_counts.div(total_attempts[top_40_qbs], axis=0) * 100

# Calculate standard deviation across the 4 play types for each player
player_percentages['StdDev'] = player_percentages[play_type_cols].std(axis=1)

# Sort by standard deviation
player_percentages = player_percentages.sort_values('StdDev', ascending=False)

# Create the heatmap
plt.figure(figsize=(11, 12))
sns.heatmap(player_percentages, 
            annot=True,           
            fmt='.1f',            
            cmap='Blues',         
            linewidths=0.5,       
            cbar_kws={'label': 'Percentage (%)'})
plt.xlabel('Play Type')
plt.ylabel('Player')
plt.title('Play Type Distribution by Passer (Top 40) - Sorted by StdDev')
plt.tight_layout()
plt.show()     
```
Heatmap: <br>
![Player StDev](Heatmap players.png)
