############################################################################
### QPMwP CODING EXAMPLES - BACKTESTING - LEARNING TO RANK (LTR)
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     14.04.2025
# First version:    27.03.2025
# --------------------------------------------------------------------------




# Make sure to install the following packages before running the demo:

# pip install pyarrow fastparquet   # For reading and writing parquet files
# pip install xgboost               # For training the model with XGBoost
# pip install scikit-learn          # For calculating the loss function (ndcg_score)


# This script demonstrates the application of Learning to Rank to predict
# the cross-sectional ordering of stock returns within a backtest framework.



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
from helper_functions import (
    load_data_spi,
    load_pickle,
)
from estimation.covariance import Covariance
from optimization.optimization import ScoreVariance  # ---- New !
from backtesting.backtest_item_builder_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    bibfn_selection_min_volume,
    bibfn_selection_gaps,
    bibfn_size_dependent_upper_bounds,
    bibfn_return_series,
    bibfn_budget_constraint,
    bibfn_box_constraints,
    bibfn_turnover_constraint,
)
from backtesting.backtest_data import BacktestData
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest






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
data.bm_series = load_data_spi(path='../data/')





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




# --------------------------------------------------------------------------
# Define training dates and rebalancing dates
# --------------------------------------------------------------------------

train_dates = features.index.get_level_values('date').unique().sort_values()
train_dates = train_dates[train_dates > market_data_dates[0]]
rebdates = train_dates[train_dates >= '2015-01-01'].strftime('%Y-%m-%d').tolist()
rebdates = rebdates[0:-1]
rebdates



# --------------------------------------------------------------------------
# Prepare labels (i.e., ranks of period returns)
# --------------------------------------------------------------------------

# Load return series
return_series = data.get_return_series()

# Compute period returns between the training dates
return_series_agg = (1 + return_series).cumprod().loc[train_dates].pct_change()

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

# Add the merged data frame to the BacktestData object
data.merged_df = merged_df








# --------------------------------------------------------------------------
# Prepare backtest service
# --------------------------------------------------------------------------

def bibfn_selection_ltr(bs: 'BacktestService', rebdate: str, **kwargs) -> None:
    '''
    This function constructs labels and features for a specific rebalancing date.
    It acts as a filtering since stocks which could not be labeled or which
    do not have features are excluded from the selection.
    '''

    # Define the selection by the ids available for the current rebalancing date
    df_test = bs.data.merged_df[bs.data.merged_df['date'] == rebdate]
    ids = list(df_test['id'].unique())

    # Return a binary series indicating the selected stocks
    return pd.Series(1, index=ids, name='binary', dtype=int)






def bibfn_scores_ltr(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Constructs scores based on a Learning-to-Rank model.        
    '''

    # Arguments
    params_xgb = kwargs.get('params_xgb')
    if params_xgb is None or not isinstance(params_xgb, dict):
        raise ValueError('params_xgb is not defined or not a dictionary.')
    training_dates = kwargs.get('training_dates')

    # Extract data
    df_train = bs.data.merged_df[bs.data.merged_df['date'] < rebdate]
    df_test = bs.data.merged_df[bs.data.merged_df['date'] == rebdate]
    df_test = df_test.loc[df_test['id'].drop_duplicates(keep='first').index]
    df_test = df_test.loc[df_test['id'].isin(bs.selection.selected)]

    # Training data
    X_train = (
        df_train.drop(['date', 'id', 'label', 'ret'], axis=1)
        # df_train.drop(['date', 'id', 'label'], axis=1)  # Include ret in the features as a proof of concept
    )
    y_train = df_train['label'].loc[X_train.index]
    grouped_train = df_train.groupby('date').size().to_numpy()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(grouped_train)

    # Test data
    y_test = pd.Series(df_test['label'].values, index=df_test['id'])
    X_test = df_test.drop(['date', 'id', 'label', 'ret'], axis=1)
    # X_test = df_test.drop(['date', 'id', 'label'], axis=1)  # Include ret in the features as a proof of concept
    grouped_test = df_test.groupby('date').size().to_numpy()
    dtest = xgb.DMatrix(X_test)
    dtest.set_group(grouped_test)

    # Train the model using the training data
    if rebdate in training_dates:
        model = xgb.train(params_xgb, dtrain, 100)
        bs.model_ltr = model
    else:
        # Use the previous model for the current rebalancing date
        model = bs.model_ltr

    # Predict using the test data
    pred = model.predict(dtest)
    preds =  pd.Series(pred, df_test['id'], dtype='float64')
    ranks = preds.rank(method='first', ascending=True).astype(int)

    # Output
    scores = pd.concat({
        'scores': preds,
        'ranks': (100 * ranks / len(ranks)).astype(int),  # Normalize the ranks to be between 0 and 100
        'true': y_test,
        'ret': pd.Series(df_test['ret'].values, index=df_test['id']),
    }, axis=1)
    bs.optimization_data['scores'] = scores
    return None






# Define the selection item builders.
selection_item_builders = {
    'gaps': SelectionItemBuilder(
        bibfn = bibfn_selection_gaps,
        width = 252*3,
        n_days = 10,
    ),
    'min_volume': SelectionItemBuilder(
        bibfn = bibfn_selection_min_volume,
        width = 252,
        min_volume = 500_000,
        agg_fn = np.median,
    ),
    'ltr': SelectionItemBuilder(
        bibfn = bibfn_selection_ltr,
    ),
}


# Define the optimization item builders.
optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn = bibfn_return_series, # Data used for covariance estimation
        width = 252*3,
        fill_value = 0,
    ),
    'scores_ltr': OptimizationItemBuilder(
        bibfn = bibfn_scores_ltr,
        params_xgb = {
            'objective': 'rank:ndcg',
            'ndcg_exp_gain': False,
            'eval_metric': 'ndcg@10',
            'min_child_weight': 1,
            'max_depth': 6,
            'eta': 0.1,
            'gamma': 1.0,
            'lambda': 1,
            'alpha': 0,
        },
        # training_dates = train_dates,
        training_dates = train_dates[train_dates <= rebdates[0]],  # Only train on the first rebalancing
    ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn = bibfn_budget_constraint,
        budget = 1
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn = bibfn_box_constraints,
        upper = 0.1
    ),
}





# Initialize the backtest service
bs = BacktestService(
    data = data,
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates,
)







# --------------------------------------------------------------------------
# Run backtests
# --------------------------------------------------------------------------


# Update the backtest service with a ScoreVariance optimization object
bs.optimization = ScoreVariance(
    field = 'scores',
    covariance = Covariance(method = 'pearson'),
    risk_aversion = 1,
    solver_name = 'cvxopt',
)

# Instantiate the backtest object and run the backtest
bt_sv = Backtest()

# Run the backtest
bt_sv.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_sv.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_sv_retrain_monthly.pickle' # <change this to your desired filename>
# )




xgb.plot_importance(bs.model_ltr, importance_type='weight', max_num_features=20, title='Feature Importance (weight)')
xgb.plot_importance(bs.model_ltr, importance_type='gain', max_num_features=20, title='Feature Importance (gain)')









# --------------------------------------------------------------------------
# Run backtest v2: Adding size-dependent upper bounds
# --------------------------------------------------------------------------


# Reinitialize the backtest service with the size-dependent upper bounds
bs = BacktestService(
    data = data,
    optimization = ScoreVariance(
        field = 'scores',
        covariance = Covariance(method = 'pearson'),
        risk_aversion = 1,
        solver_name = 'cvxopt',
    ),
    selection_item_builders = selection_item_builders,
    optimization_item_builders = {
        **optimization_item_builders,
        'size_dep_upper_bounds': OptimizationItemBuilder(
            bibfn = bibfn_size_dependent_upper_bounds,
            small_cap = {'threshold': 300_000_000, 'upper': 0.02},
            mid_cap = {'threshold': 1_000_000_000, 'upper': 0.05},
            large_cap = {'threshold': 10_000_000_000, 'upper': 0.1},
        ),
    },
    rebdates = rebdates,
)

# Instantiate the backtest object and run the backtest
bt_sv_sdub = Backtest()

# Run the backtest
bt_sv_sdub.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_sv_sdub.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_sv_sdub.pickle' # <change this to your desired filename>
# )










# --------------------------------------------------------------------------
# Run backtest v3: Adding turnover constraint
# --------------------------------------------------------------------------


# Reinitialize the backtest service with the size-dependent upper bounds
bs = BacktestService(
    data = data,
    optimization = ScoreVariance(
        field = 'scores',
        covariance = Covariance(method = 'pearson'),
        risk_aversion = 1,
        solver_name = 'cvxopt',
    ),
    selection_item_builders = selection_item_builders,
    optimization_item_builders = {
        **optimization_item_builders,
        'size_dep_upper_bounds': OptimizationItemBuilder(
            bibfn = bibfn_size_dependent_upper_bounds,
            small_cap = {'threshold': 300_000_000, 'upper': 0.02},
            mid_cap = {'threshold': 1_000_000_000, 'upper': 0.05},
            large_cap = {'threshold': 10_000_000_000, 'upper': 0.1},
        ),
        'turnover_constraint': OptimizationItemBuilder(
            bibfn = bibfn_turnover_constraint,
            turnover_limit = 0.25,
        ),
    },
    rebdates = rebdates,
)

# Instantiate the backtest object and run the backtest
bt_sv_sdub_tocon = Backtest()

# Run the backtest
bt_sv_sdub_tocon.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_sv_sdub_tocon.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_sv_sdub_tocon.pickle' # <change this to your desired filename>
# )







# --------------------------------------------------------------------------
# Run backtest v4: Adding turnover constraint, no size-dependent upper bounds
# --------------------------------------------------------------------------


# Reinitialize the backtest service with the size-dependent upper bounds
bs = BacktestService(
    data = data,
    optimization = ScoreVariance(
        field = 'scores',
        covariance = Covariance(method = 'pearson'),
        risk_aversion = 1,
        solver_name = 'cvxopt',
    ),
    selection_item_builders = selection_item_builders,
    optimization_item_builders = {
        **optimization_item_builders,
        'turnover_constraint': OptimizationItemBuilder(
            bibfn = bibfn_turnover_constraint,
            turnover_limit = 0.25,
        ),
    },
    rebdates = rebdates,
)

# Instantiate the backtest object and run the backtest
bt_sv_tocon = Backtest()

# Run the backtest
bt_sv_tocon.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_sv_tocon.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_sv_tocon.pickle' # <change this to your desired filename>
# )









# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------


# Laod backtests from pickle
path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/' #<change this to your local path>

bt_sv = load_pickle(
    filename = 'backtest_sv.pickle',
    path = path,
)
bt_sv_poc = load_pickle(
    filename = 'backtest_sv_poc.pickle',
    path = path,
)
bt_sv_retrain_monthly = load_pickle(
    filename = 'backtest_sv_retrain_monthly.pickle',
    path = path,
)
bt_sv_sdub = load_pickle(
    filename = 'backtest_sv_sdub.pickle',
    path = path,
)
bt_sv_sdub_tocon = load_pickle(
    filename = 'backtest_sv_sdub_tocon.pickle',
    path = path,
)
bt_sv_tocon = load_pickle(
    filename = 'backtest_sv_tocon.pickle',
    path = path,
)



fixed_costs = 0
variable_costs = 0.002
return_series = bs.data.get_return_series()

strategy_dict = {
    'sv_poc': bt_sv_poc.strategy,
    # 'sv': bt_sv.strategy,
    # 'sv_retrain_monthly': bt_sv_retrain_monthly.strategy,
    # 'sv_sdub': bt_sv_sdub.strategy,
    # 'sv_sdub_tocon': bt_sv_sdub_tocon.strategy,
    # 'sv_tocon': bt_sv_tocon.strategy,
}

sim_dict_gross = {
    f'{key}_gross': value.simulate(
        return_series=return_series,
        fc=fixed_costs,
        vc=0,
    )
    for key, value in strategy_dict.items()
}
sim_dict_net = {
    f'{key}_net': value.simulate(
        return_series=return_series,
        fc=fixed_costs,
        vc=variable_costs,
    )
    for key, value in strategy_dict.items()
}


sim = pd.concat({
    'bm': bs.data.bm_series,
    **sim_dict_gross,
    **sim_dict_net,
}, axis = 1).dropna()




np.log((1 + sim)).cumsum().plot(title='Cumulative Performance', figsize = (10, 6))









# --------------------------------------------------------------------------
# Turnover
# --------------------------------------------------------------------------

to_sv = bt_sv.strategy.turnover(return_series=return_series)
to_sv_sdub = bt_sv_sdub.strategy.turnover(return_series=return_series)
to_sv_sdub_tocon = bt_sv_sdub_tocon.strategy.turnover(return_series=return_series)
to_sv_tocon = bt_sv_tocon.strategy.turnover(return_series=return_series)

to = pd.concat({
    'sv': to_sv,
    'sv_sdub': to_sv_sdub,
    'sv_sdub_tocon': to_sv_sdub_tocon,
    'sv_tocon': to_sv_tocon,
}, axis = 1).dropna()
to.columns = [
    'Score-Variance',
    'Score-Variance with Size Dependen Upper Bounds',
    'Score-Variance with Size Dependen Upper Bounds and Turnover Constraint',
    'Score-Variance with Turnover Constraint',
]

to.plot(title='Turnover', figsize = (10, 6))
to.mean() * 12
to







# --------------------------------------------------------------------------
# Decriptive statistics
# --------------------------------------------------------------------------

import empyrical as ep


# Compute individual performance metrics for each simulated strategy using empyrical
annual_return = {}
cumulative_returns = {}
annual_volatility = {}
sharpe_ratio = {}
max_drawdown = {}
tracking_error = {}
for column in sim.columns:
    print(f'Performance metrics for {column}')
    annual_return[column] = ep.annual_return(sim[column])
    cumulative_returns[column] = ep.cum_returns(sim[column]).tail(1).values[0]
    annual_volatility[column] = ep.annual_volatility(sim[column])
    sharpe_ratio[column] = ep.sharpe_ratio(sim[column])
    max_drawdown[column] = ep.max_drawdown(sim[column])
    tracking_error[column] = ep.annual_volatility(sim[column] - sim['bm'])


annual_returns = pd.DataFrame(annual_return, index=['Annual Return'])
cumret = pd.DataFrame(cumulative_returns, index=['Cumulative Return'])
annual_volatility = pd.DataFrame(annual_volatility, index=['Annual Volatility'])
sharpe  = pd.DataFrame(sharpe_ratio, index=['Sharpe Ratio'])
mdd = pd.DataFrame(max_drawdown, index=['Max Drawdown'])
pd.concat([annual_returns, cumret, annual_volatility, sharpe, mdd])
