############################################################################
### QPMwP CODING EXAMPLES - LEARNING TO RANK (LTR)
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     27.03.2025
# First version:    27.03.2025
# --------------------------------------------------------------------------




# Make sure to install the following packages before running the demo:

# pip install pyarrow fastparquet   # For reading and writing parquet files
# pip install xgboost               # For training the model with XGBoost
# pip install scikit-learn          # For calculating the loss function (ndcg_score)


# This script demonstrates the application of Learning to Rank to predict
# the cross-sectional ordering of stock returns for one point in time.



# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import ndcg_score

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from backtesting.backtest_data import BacktestData








# --------------------------------------------------------------------------
# Load data
# - market data (from parquet file)
# - jkp data (from parquet file)
# - swiss performance index, SPI (from csv file)
# --------------------------------------------------------------------------

path_to_data = 'C:/Users/User/OneDrive/Documents/QPMwP/Data/'  # <change this to your path to data>

# Load market and jkp data from parquet files
market_data = pd.read_parquet(path = f'{path_to_data}market_data.parquet')
jkp_data = pd.read_parquet(path = f'{path_to_data}jkp_data.parquet')

# Instantiate the BacktestData class
# and set the market data and jkp data as attributes
data = BacktestData()
data.market_data = market_data
data.jkp_data = jkp_data




# --------------------------------------------------------------------------
# Create a features dataframe from the jkp_data
# Reset the date index to be consistent with the date index in market_data
# --------------------------------------------------------------------------

market_data_dates = data.market_data.index.get_level_values('date').unique().sort_values()
jkp_data_dates = data.jkp_data.index.get_level_values('date').unique().sort_values()

# Find the nearest future market_data_date for each jkp_data_date
dates_map = {
    date: min(market_data_dates[market_data_dates > date])
    for date in jkp_data_dates
}

# Generates a features dataframe from the jkp_data where you reset
# the date index to b
features = data.jkp_data.reset_index()
features['date'] = features['date'].map(dates_map)
features = features.set_index(['date', 'id'])
features



# --------------------------------------------------------------------------
# Define training dates
# --------------------------------------------------------------------------

train_dates = features.index.get_level_values('date').unique().sort_values()
train_dates = train_dates[train_dates > market_data_dates[0]]
train_dates




# --------------------------------------------------------------------------
# Prepare labels (i.e., ranks of period returns)
# --------------------------------------------------------------------------

# Load return series
return_series = data.get_return_series()
return_series

# Compute period returns between the training dates
return_series_agg = (1 + return_series).cumprod().loc[train_dates].pct_change()
return_series_agg

# Shift the labels by -1 period (as we want to predict next period return ranks)
return_series_agg_shift = return_series_agg.shift(-1)
# return_series_agg_shift = return_series_agg   # ~~~~~~~~~~~~~~~~~~~~~~~~

# Stack the returns (from wide to long format)
ret = return_series_agg_shift.unstack().reorder_levels([1, 0]).dropna()
ret.name = 'ret'
ret

# Merge the returns and the features dataframes
merged_df = ret.to_frame().join(features, how='inner').sort_index()
merged_df

# Generate the labels (ranks) for the merged data
labels = merged_df.groupby('date')['ret'].rank(method='first', ascending=True).astype(int)
labels = 100 * labels / merged_df.groupby('date').size() # Normalize the ranks to be between 0 and 100
labels = labels.astype(int)  # Convert to integer type
labels

# Insert the labels into the merged data frame
merged_df.insert(0, 'label', labels)
merged_df

# Reset the index of the merged data frame
merged_df.reset_index(inplace=True)
merged_df




# --------------------------------------------------------------------------
# Split dataset into train and test
# --------------------------------------------------------------------------

split_date = '2023-03-01'
df_train = merged_df[merged_df['date'] < split_date].reset_index(drop=True)
df_test = merged_df[merged_df['date'] == split_date].reset_index(drop=True)

# Training data
X_train = (
    df_train.drop(['date', 'id', 'label', 'ret'], axis=1)
    # df_train.drop(['date', 'id', 'label'], axis=1)
#    .dropna(axis=1, how='all')
#    .dropna(axis=0, how='all')
    .fillna(0)
)
y_train = df_train['label'].loc[X_train.index]
grouped_train = df_train.groupby('date').size().to_numpy()
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(grouped_train)

# Test data
X_test = df_test.drop(['date', 'id', 'label', 'ret'], axis=1)
# X_test = df_test.drop(['date', 'id', 'label'], axis=1)
y_test = df_test['label']
grouped_test = df_test.groupby('date').size().to_numpy()
dtest = xgb.DMatrix(X_test) #, label=y_test)
dtest.set_group(grouped_test)




# --------------------------------------------------------------------------
# Configuration of the XGBoost model
# --------------------------------------------------------------------------

params = {
    # 'objective': 'rank:pairwise',  # Alternative objective for pairwise ranking.
    'objective': 'rank:ndcg',  # Optimize for NDCG (Normalized Discounted Cumulative Gain), suitable for ranking tasks.
    'ndcg_exp_gain': False,  # Disable exponential gain for NDCG calculation, useful for datasets with wide relevance ranges.
    # 'eval_metric': 'ndcg@5',  # Evaluate NDCG at the top 5 items (commented out).
    'eval_metric': 'ndcg',  # Evaluate NDCG across all items.
    'boosting_type': 'gbdt',  # Use Gradient Boosting Decision Trees as the boosting method.
    'min_child_weight': 1,  # Minimum sum of instance weights (hessian) in a child node to avoid overfitting.
    'max_depth': 6,  # Maximum depth of trees, controls model complexity and risk of overfitting.
    'eta': 0.1,  # Learning rate, controls the contribution of each tree to the model.
    'gamma': 1.0,  # Minimum loss reduction required for a split, higher values make the model more conservative.
    'n_estimators': 100,  # Number of boosting rounds (trees) to train.
    'lambda': 1,  # L2 regularization term to reduce overfitting.
    'alpha': 0,  # L1 regularization term to reduce overfitting.
}



# --------------------------------------------------------------------------
# Fit and predict
# --------------------------------------------------------------------------

# Train the model using the training data only
model = xgb.train(params, dtrain, 100)

# Predict using the test data
preds = model.predict(dtest)
ranks = pd.Series(preds).rank(method='first', ascending=True).astype(int)
y_pred = (100 * ranks / len(ranks)).astype(int)  # Normalize the ranks to be between 0 and 100



# --------------------------------------------------------------------------
# Analyze the results
# --------------------------------------------------------------------------

# Predictions vs. true (future) labels
out = pd.concat([
    y_pred,
    y_test.astype(int), 
    df_test['ret']
], axis=1)
out.index.name = 'id'
out.columns = ['y_pred', 'y_true', 'ret']

out.plot(kind='scatter', x='y_true', y='y_pred')
out.plot(kind='scatter', x='y_pred', y='ret')
out.plot(kind='scatter', x='y_true', y='ret')

out.sort_values('y_pred', ascending=False).head(10)
out.sort_values('y_true', ascending=False).head(10)



# Calculate the NDCG score

preds_reshaped = preds.reshape(1, -1)
labels_reshaped = y_test.to_numpy().reshape(1, -1)
ndcg_score(labels_reshaped, preds_reshaped, k = len(preds))
ndcg_score(labels_reshaped, preds_reshaped, k = 10)





# Feature importance

# Extract feature importance based on gain
f_importance = model.get_score(importance_type='gain')

# Sort the feature importance in descending order
sorted_f_importance = {
    k: v for k, v in sorted(
        f_importance.items(),
        key=lambda item: item[1],
        reverse=True
    )
}
sorted_f_importance


# Plotting
plt.figure(figsize=(10, 8))
plt.bar(range(len(sorted_f_importance)), list(sorted_f_importance.values()), align='center')
plt.xticks(range(len(sorted_f_importance)), list(sorted_f_importance.keys()), rotation='vertical')
plt.title('Feature Importance based on gain')
plt.ylabel('Gain')
plt.xlabel('Features')
plt.show()




# Use feature importance plot from XGBoost
xgb.plot_importance(model, importance_type='weight', max_num_features=20, title='Feature Importance (weight)')
xgb.plot_importance(model, importance_type='gain', max_num_features=20, title='Feature Importance (gain)')









# --------------------------------------------------------------------------
# Alternatively, use XGBRanker
# --------------------------------------------------------------------------

from xgboost import XGBRanker

# Define XGBRanker model
ranker = XGBRanker(
    # objective='rank:pairwise',  # Use pairwise ranking
    objective='rank:ndcg',  # Use NDCG as the ranking objective
    # learning_rate=0.1,      # Learning rate
    # gamma=1.0,              # Minimum loss reduction for a split
    # min_child_weight=0.1,   # Minimum sum of instance weight in a child
    # max_depth=6,            # Maximum tree depth
    # n_estimators=50,        # Number of boosting rounds
    eval_metric='ndcg@10'    # Evaluate NDCG at top 10
)

# Train the model
query_ids_train = df_train['date'].values
ranker.fit(
    X_train,
    y_train,
    qid=query_ids_train,
)

# Predict rankings on the test set
preds = ranker.predict(X_test)
preds









