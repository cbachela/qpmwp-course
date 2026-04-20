############################################################################
### QPMwP CODING EXAMPLES - Backtest 4
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     13.03.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------




# This script demonstrates how to run a backtest using the qpmwp-course library
# and single stock data which change over time.
# The script uses the 'MeanVariance' and 'LeastSquares' portfolio optimization classes.


# Make sure to install the following packages before running the demo:

# uv pip install pyarrow fastparquet   # For reading and writing parquet files










# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from helper_functions import (
    load_pickle,
    load_data_spi,
)
from estimation.covariance import Covariance
from estimation.expected_return import ExpectedReturn
from optimization.optimization import (
    MeanVariance,
    LeastSquares,
)
from backtesting.backtest_item_builder.bib_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder.bibfn_selection import (
    bibfn_selection_NA,                                          # NEW
)
from backtesting.backtest_item_builder.bibfn_optimization_data import (
    bibfn_return_series,
    bibfn_bm_series,
)
from backtesting.backtest_item_builder.bibfn_constraints import (
    bibfn_budget_constraint,
    bibfn_box_constraints,
    bibfn_bm_relative_upper_bounds,                              # NEW
)
from backtesting.backtest_data import BacktestData
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest





# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PATH_TO_DATA = '/Users/michaelebling/Documents/UZH/Sem_8/Quant._Portfolio_Mngmt_Python/Data/'     # <change this to your path to data>
SAVE_PATH = '/Users/michaelebling/Documents/UZH/Sem_8/Quant._Portfolio_Mngmt_Python/Data/'        # <change this to your path where you want to store the backtest>
WIDTH_3Y = 365 * 3       # Notice that we use 365 days bcs the dataset also contains weekends and holidays




# --------------------------------------------------------------------------
# Load data
# - market data (from parquet file)
# - jkp data (from parquet file)
# - swiss performance index, SPI (from csv file)
# --------------------------------------------------------------------------


# Load market and jkp data from parquet files
market_data = pd.read_parquet(path = f'{PATH_TO_DATA}market_data.parquet')
jkp_data = pd.read_parquet(path = f'{PATH_TO_DATA}jkp_data.parquet')


# Inspect the data

# market_data: 
# Contains the daily market data for all stocks, i.e.,
# price, market capitalization, liquidity (i.e., trading volume) and
# sector information. 
# The data is in long format with multi-index (date, id), i.e. 
# each row corresponds to a stock on a given date.
market_data.head()

# jkp_data:
# Contains the monthly data for stock specific characteristics.
# See: https://jkpfactors.com/
# The data is in long format, i.e. each row corresponds to a stock on a given date.
jkp_data.head()





# --------------------------------------------------------------------------
# Prepare backtest service
# --------------------------------------------------------------------------


# -------------------------
# First, ensure that market data and jkp data 
# have the same dates by forward filling the market data for the missing dates.
# -------------------------

market_data_dates = (
    market_data
    .index.get_level_values('date')
    .unique().sort_values()
)
jkp_data_dates = (
    jkp_data
    .index.get_level_values('date')
    .unique().sort_values()
)

# Inspect the distribution of the dates across the week for both datasets
market_data_dates.dayofweek.value_counts().sort_index()
jkp_data_dates.dayofweek.value_counts().sort_index()
# --> market data contains only trading days, while jkp data contains all calendar days (including weekends and holidays)

# Find the jkp_data_dates which are not in the market_data_dates
missing_dates = jkp_data_dates[~jkp_data_dates.isin(market_data_dates)]

# Extend the market data for the missing dates using the last available market data (i.e., forward fill).
tmp_dict = {}
for date in missing_dates:
    last_date = market_data_dates[market_data_dates <= date][-1]
    tmp_dict[date] = market_data.loc[last_date]
    
df_missing = pd.concat(tmp_dict, axis=0)
df_missing.index.names = market_data.index.names
market_data_ffill = pd.concat([market_data, df_missing]).sort_index()



# -------------------------
# Define rebalancing dates
# -------------------------

n_month = 3 # We want to rebalance every n_month months

# We want to use the dates from the jkp data for rebalancing, 
# since they are less frequent than the market data dates.
rebdates = (
    jkp_data_dates[
        jkp_data_dates > market_data_dates[0]
    ][::n_month]
    .strftime('%Y-%m-%d').tolist()
)
# Drop the first rebalancing dates which are before 2002-01-01, 
# because of poor data coverage.
rebdates = [date for date in rebdates if date > '2002-01-01']
rebdates




# -------------------------
# Instantiate the BacktestData class
# and set the market, jkp, and benchmark data as attributes
# -------------------------

data = BacktestData()
data.market_data = market_data_ffill  # notice that we use the forward filled market data here
data.jkp_data = jkp_data
data.bm_series = load_data_spi(path='../data/')



# Helper methods of class BacktestDatathat extract certain columns from
# the data objects from long format to wide format

??data.get_return_series
data.get_return_series()

??data.get_volume_series
data.get_volume_series()

??data.get_characteristic_series
data.get_characteristic_series(
    field='qmj',
)

# Plot the density of the qmj characteristic (z-scores) for the last available date
qmj = data.get_characteristic_series(
    field='qmj',
).tail(1).squeeze()
qmj
qmj.plot(kind='density', title='Density of qmj')




# -------------------------
# Define the selection item builders.
# -------------------------

# SelectionItemBuilder is a callable class which takes a function (bibfn) as argument.
# The function bibfn is a custom function that builds a selection item, i.e. a
# pandas Series of boolean values indicating the selected assets at a given rebalancing date.

# The function bibfn takes the backtest service (bs) and the rebalancing date (rebdate) as arguments.
# Additional keyword arguments can be passed to bibfn using the arguments attribute of the SelectionItemBuilder instance.

# The selection item is then added to the Selection attribute of the backtest service using the add_item method.
# To inspect the current instance of the selection object, type bs.selection.df()


selection_item_builders = {
    'NA': SelectionItemBuilder(
        bibfn=bibfn_selection_NA,
        width=WIDTH_3Y,
        na_threshold=70,
    ),
}



# -------------------------
# Define the optimization item builders.
# -------------------------

# OptimizationItemBuilder is a callable class which takes a function (bibfn) as argument.
# The function bibfn is a custom function that builds an item which is used for the optimization.

# Such items can be constraints, which are added to the constraints attribute of the optimization object,
# or datasets which are added to the instance of the OptimizationData class.

# The function bibfn takes the backtest service (bs) and the rebalancing date (rebdate) as arguments.
# Additional keyword arguments can be passed to bibfn using the arguments attribute of the OptimizationItemBuilder instance.


optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn=bibfn_return_series,
        width=WIDTH_3Y,
        weekdays_only=True,
        fill_value=0,
    ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn=bibfn_budget_constraint,
        budget=1
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn=bibfn_box_constraints,
        upper=0.1
    ),
}




# -------------------------
# Initialize the backtest service
# -------------------------

bs = BacktestService(
    data=data,
    selection_item_builders=selection_item_builders,
    optimization_item_builders=optimization_item_builders,
    rebdates=rebdates,
)






# --------------------------------------------------------------------------
# Run backtest for mean-variance portfolio
# --------------------------------------------------------------------------


# Update the backtest service with a MeanVariance optimization object
bs.optimization = MeanVariance(
    covariance=Covariance(method='pearson'),
    expected_return=ExpectedReturn(method='geometric'),
    risk_aversion=1,
    solver_name='cvxopt',
)

# Instantiate the backtest object and run the backtest
bt_mv = Backtest()

# Run the backtest
bt_mv.run(bs=bs)

# Save the backtest as a .pickle file
bt_mv.save(
     path=SAVE_PATH,
     filename='demo_backtest_4_mv.pickle' # <change this to your desired filename>
)


# Manually check how many companies are affected by the NA filter
return_series = bs.data.get_return_series(
    width=WIDTH_3Y,
    end_date=rebdates[0],
    # end_date=rebdates[-1],
    weekdays_only=True,
    fillna_value=None,
)
na_counts = return_series.isna().sum()
na_counts
(na_counts < 70).sum()






# Inspect the optimization results - i.e. the weights stored in the strategy object
bt_mv.strategy.get_weights_df()
bt_mv.strategy.get_weights_df().plot(kind='bar', stacked=True, figsize=(10, 6), legend=False, title='Portfolio Weights')

# Inspect the selection for the last rebalancing date
bs.selection.df()
bs.selection.df_binary()
bs.selection.df_binary().sum()
bs.selection.selected

# Inspect the constraints for the last rebalancing date
len(bs.optimization.constraints.box['upper'])
len(bs.optimization.constraints.ids)






# --------------------------------------------------------------------------
# Prepare the optimization for a specific date and inspect the generated data items
# i.e., the selection object, the optimization data and the optimization constraints
# --------------------------------------------------------------------------

rebalancing_date = rebdates[0]
bs.build_selection(rebdate=rebalancing_date)
bs.build_optimization(rebdate=rebalancing_date)

# Alternatively, use the wrapper: prepare_optimization, 
# which calls both build_selection and build_optimization.
# bs.prepare_rebalancing(rebalancing_date=rebalancing_date)


# Inspect the selection for the last rebalancing date
bs.selection.df()
bs.selection.df_binary()
bs.selection.df_binary().sum()
bs.selection.selected
bs.selection.filtered

# Inspect the optimization data for the last rebalancing date
bs.optimization_data

# Inspect the optimization constraints
bs.optimization.constraints.budget
bs.optimization.constraints.box
bs.optimization.constraints.linear











# --------------------------------------------------------------------------
# Run backtest for index tracking portfolio (via least squares optimization)
# --------------------------------------------------------------------------

# Define the optimization item builders for the least squares optimization.
# Notice that we add the benchmark return series as an optimization item, 
# and that we use a different box constraint which sets the upper bounds 
# of the weights to be a multiple of the benchmark weights (i.e., here: 20 times the benchmark weights).

optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn=bibfn_return_series,
        width=WIDTH_3Y,
        fill_value=0,
    ),
    'bm_series': OptimizationItemBuilder(
        bibfn=bibfn_bm_series,
        width=WIDTH_3Y,
        align=True,
    ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn=bibfn_budget_constraint,
        budget=1
    ),
    # 'box_constraints': OptimizationItemBuilder(
    #     bibfn=bibfn_box_constraints,
    #     upper=0.1
    # ),
    'box_constraints': OptimizationItemBuilder(
        bibfn=bibfn_bm_relative_upper_bounds,
        multiple=20,
    ),
}

# Initialize the backtest service
bs = BacktestService(
    data=data,
    selection_item_builders=selection_item_builders,
    optimization_item_builders=optimization_item_builders,
    rebdates=rebdates,
)

# Update the backtest service with a LeastSquares optimization object
bs.optimization = LeastSquares(
    solver_name='cvxopt',
)

# Instantiate the backtest object and run the backtest
bt_ls = Backtest()

# Run the backtest
bt_ls.run(bs=bs)

# Save the backtest as a .pickle file
bt_ls.save(
    path=SAVE_PATH,
    filename='demo_backtest_4_ls.pickle' # <change this to your desired filename>
)







# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------

# Laod backtests from pickle
bt_mv = load_pickle(
    filename='demo_backtest_4_mv.pickle',
    path=SAVE_PATH,
)
bt_ls = load_pickle(
    filename='demo_backtest_4_ls.pickle',
    path=SAVE_PATH,
)

# Simulate
fixed_costs = 0
variable_costs = 0
return_series = bs.data.get_return_series()

sim_mv = bt_mv.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)
sim_ls = bt_ls.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)

# Concatenate the simulations
sim = pd.concat({
    'bm': bs.data.bm_series,
    'mv': sim_mv,
    'ls': sim_ls,
}, axis=1).dropna()
sim.columns = ['Benchmark', 'Mean-Variance', 'Index Tracker']


# Plot the cumulative performance
np.log((1 + sim)).cumsum().plot(title='Cumulative Performance', figsize=(10, 6))


# Out-/Underperformance
def sim_outperformance(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    ans = (x.subtract(y, axis=0)).divide(1 + y, axis=0)
    return ans

sim_rel = sim_outperformance(sim, sim['Benchmark'])

np.log((1 + sim_rel)).cumsum().plot(title='Cumulative Out-/Underperformance', figsize=(10, 6))




# --------------------------------------------------------------------------
# Turnover
# --------------------------------------------------------------------------

to_mv = bt_mv.strategy.turnover(return_series=return_series)
to_ls = bt_ls.strategy.turnover(return_series=return_series)

to = pd.concat({
    'mv': to_mv,
    'ls': to_ls,
}, axis = 1).dropna()
to.columns = [
    'Mean-Variance',
    'Index Tracker',
]

to.plot(title='Turnover', figsize = (10, 6))
to.mean() * 4




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
    tracking_error[column] = ep.annual_volatility(sim[column] - sim['Benchmark'])


annual_returns = pd.DataFrame(annual_return, index=['Annual Return'])
cumret = pd.DataFrame(cumulative_returns, index=['Cumulative Return'])
annual_volatility = pd.DataFrame(annual_volatility, index=['Annual Volatility'])
sharpe  = pd.DataFrame(sharpe_ratio, index=['Sharpe Ratio'])
mdd = pd.DataFrame(max_drawdown, index=['Max Drawdown'])
tracking_error = pd.DataFrame(tracking_error, index=['Tracking Error'])
pd.concat([annual_returns, cumret, annual_volatility, sharpe, mdd, tracking_error])



