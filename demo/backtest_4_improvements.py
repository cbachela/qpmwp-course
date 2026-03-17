############################################################################
### QPMwP CODING EXAMPLES - Backtest 4 - IMPROVEMENTS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     09.03.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------




# This script builds upon the mean-variance optimization example (backtest_4.py)
# and tries to improve the strategy with smarter filtering and constraints.


# Make sure to install the following packages before running the demo:

# pip install pyarrow fastparquet   # For reading and writing parquet files








# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import empyrical as ep

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
from optimization.optimization import MeanVariance
from backtesting.backtest_item_builder.bib_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder.bibfn_selection import (
    bibfn_selection_gaps,                                          # NEW
    # bibfn_selection_min_volume,                                    # NEW
)
from backtesting.backtest_item_builder.bibfn_optimization_data import (
    bibfn_return_series,
)
from backtesting.backtest_item_builder.bibfn_constraints import (
    bibfn_budget_constraint,
    bibfn_box_constraints,
    bibfn_size_dependent_upper_bounds,                             # NEW
)
from backtesting.backtest_data import BacktestData
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest







# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PATH_TO_DATA = '...'     # <change this to your path to data>
SAVE_PATH = '...'        # <change this to your path where you want to store the backtest>
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




# --------------------------------------------------------------------------
# Prepare backtest service
# --------------------------------------------------------------------------

# -------------------------
# Allign market and jkp data on the same dates
# See backtest_4.py for details
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
missing_dates = jkp_data_dates[~jkp_data_dates.isin(market_data_dates)]
tmp_dict = {}
for date in missing_dates:
    last_date = market_data_dates[market_data_dates <= date][-1]
    tmp_dict[date] = market_data.loc[last_date]
    
df_missing = pd.concat(tmp_dict, axis=0)
df_missing.index.names = market_data.index.names
market_data_ffill = pd.concat([market_data, df_missing]).sort_index()

# Define rebalancing dates
n_month = 3 # We want to rebalance every n_month months
rebdates = (
    jkp_data_dates[
        jkp_data_dates > market_data_dates[0]
    ][::n_month]
    .strftime('%Y-%m-%d').tolist()
)
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
    'min_volume': SelectionItemBuilder(
        bibfn=bibfn_selection_min_volume,   # filter out stocks which are illiquid
        width=365,
        min_volume=500_000,
        agg_fn=np.median,
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
# Base model: mean-variance portfolio
# --------------------------------------------------------------------------

# # Update the backtest service with a MeanVariance optimization object
# bs.optimization = MeanVariance(
#     covariance=Covariance(method='pearson'),
#     expected_return=ExpectedReturn(method='geometric'),
#     risk_aversion=1,
#     solver_name='cvxopt',
# )

# # Instantiate the backtest object and run the backtest
# bt_mv = Backtest()

# # Run the backtest
# bt_mv.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_mv.save(
#     path=SAVE_PATH,
#     filename='demo_backtest_4_mv.pickle' # <change this to your desired filename>
# )

# Load backtest from pickle
bt_mv = load_pickle(
    filename='demo_backtest_4_mv.pickle',
    path=SAVE_PATH,
)


# --------------------------------------------------------------------------
# Adjustment 1: Increase the risk aversion parameter in the mean-variance portfolio
# --------------------------------------------------------------------------

risk_aversion = 5

# Update the backtest service with a MeanVariance optimization object
bs.optimization = MeanVariance(
    covariance=Covariance(method='pearson'),
    expected_return=ExpectedReturn(method='geometric'),
    risk_aversion=risk_aversion,
    solver_name='cvxopt',
)

# Instantiate the backtest object and run the backtest
bt_mv_v1 = Backtest()

# Run the backtest
bt_mv_v1.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_mv_v1.save(
#     path=SAVE_PATH,
#     filename='demo_backtest_4_mv_v1.pickle' # <change this to your desired filename>
# )




# --------------------------------------------------------------------------
# Adjustment 2: Gaps filter
# Filters out stocks which have not been traded for more than 'n_days' consecutive days
# --------------------------------------------------------------------------

# Add the gaps filter to the selection_item_builders dictionary
selection_item_builders['gaps'] = SelectionItemBuilder(
    bibfn=bibfn_selection_gaps,
    width=WIDTH_3Y,
    n_days=10  # filter out stocks which have not been traded for more than 'n_days' consecutive days
)

# Reinitialize the backtest service with the gaps filter
bs = BacktestService(
    data=data,
    optimization=MeanVariance(
        covariance=Covariance(method='pearson'),
        expected_return=ExpectedReturn(method='geometric'),
        risk_aversion=risk_aversion,
        solver_name='cvxopt',
    ),
    selection_item_builders=selection_item_builders,
    optimization_item_builders=optimization_item_builders,
    rebdates=rebdates,
)

# Instantiate the backtest object and run the backtest
bt_mv_v2 = Backtest()

# Run the backtest
bt_mv_v2.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_mv_v2.save(
#     path=SAVE_PATH,
#     filename='demo_backtest_4_mv_v2.pickle' # <change this to your desired filename>
# )


# Inspect the selection for the last rebalancing date
bs.selection.df()
bs.selection.df_binary().sum()





# --------------------------------------------------------------------------
# Adjustment 3: Minimum volume filter
# Filters out stocks which have not reached a minimum liquidity (i.e., trading volume) threshold
# --------------------------------------------------------------------------

def bibfn_selection_min_volume(bs, rebdate: str, **kwargs) -> pd.DataFrame:

    '''
    Backtest item builder function for defining the selection
    Filter stocks based on minimum volume (i.e., liquidity).
    '''

    # Arguments
    width = kwargs.get('width', 365)
    agg_fn = # <your code here>
    min_volume = # <your code here>

    # Volume data
    vol = bs.data.get_volume_series(
        # <your code here>
    )
    vol_agg = # <your code here>

    # Filtering
    # <your code here>

    # Output
    # <your code here>
    
    return filter_values


# Add the minimum volume filter to the selection_item_builders dictionary
# Filters out stocks which are illiquid.
# Here, i.e., stocks with a median daily trading volume below 500'000 in the last 365 trading days
selection_item_builders['min_volume'] = SelectionItemBuilder(
    bibfn=bibfn_selection_min_volume,
    width=WIDTH_3Y,
    agg_fn=np.median,
    min_volume=500_000,
)

# Reinitialize the backtest service with the minimum volume filter
bs = BacktestService(
    data=data,
    optimization=MeanVariance(
        covariance=Covariance(method='pearson'),
        expected_return=ExpectedReturn(method='geometric'),
        risk_aversion=risk_aversion,
        solver_name='cvxopt',
    ),
    selection_item_builders=selection_item_builders,
    optimization_item_builders=optimization_item_builders,
    rebdates=rebdates,
)

# Instantiate the backtest object and run the backtest
bt_mv_v3 = Backtest()

# Run the backtest
bt_mv_v3.run(bs=bs)

# Save the backtest as a .pickle file
# bt_mv_v3.save(
#     path=SAVE_PATH,
#     filename='demo_backtest_4_mv_v3.pickle' # <change this to your desired filename>
# )


# Inspect the selection for the last rebalancing date
bs.selection.df()
bs.selection.df_binary().sum()







# --------------------------------------------------------------------------
# Adjustment 4: Size-dependent upper bounds
# --------------------------------------------------------------------------

# Add the size-dependent upper bounds to the optimization_item_builders dictionary
optimization_item_builders['size_dep_upper_bounds'] = OptimizationItemBuilder(
    bibfn = bibfn_size_dependent_upper_bounds,
    small_cap = {'threshold': 300_000_000, 'upper': 0.02},
    mid_cap = {'threshold': 1_000_000_000, 'upper': 0.05},
    large_cap = {'threshold': 10_000_000_000, 'upper': 0.1},
)

# Reinitialize the backtest service with the size-dependent upper bounds
bs = BacktestService(
    data=data,
    optimization=MeanVariance(
        covariance=Covariance(method = 'pearson'),
        expected_return=ExpectedReturn(method = 'geometric'),
        risk_aversion=risk_aversion,
        solver_name='cvxopt',
    ),
    selection_item_builders=selection_item_builders,
    optimization_item_builders=optimization_item_builders,
    rebdates=rebdates,
)

# Instantiate the backtest object and run the backtest
bt_mv_v4 = Backtest()

# Run the backtest
bt_mv_v4.run(bs = bs)

# # Save the backtest as a .pickle file
# bt_mv_v4.save(
#     path=SAVE_PATH,
#     filename='demo_backtest_4_mv_v4.pickle' # <change this to your desired filename>
# )









# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------

# Laod backtests from pickle
bt_mv = load_pickle(
    filename='demo_backtest_4_mv.pickle',
    path=SAVE_PATH,
)
bt_mv_v1 = load_pickle(
    filename='demo_backtest_4_mv_v1.pickle',
    path=SAVE_PATH,
)
bt_mv_v2 = load_pickle(
    filename='demo_backtest_4_mv_v2.pickle',
    path=SAVE_PATH,
)
bt_mv_v3 = load_pickle(
    filename='demo_backtest_4_mv_v3.pickle',
    path=SAVE_PATH,
)
bt_mv_v4 = load_pickle(
    filename='demo_backtest_4_mv_v4.pickle',
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
sim_mv_v1 = bt_mv_v1.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)
sim_mv_v2 = bt_mv_v2.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)
sim_mv_v3 = bt_mv_v3.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)
sim_mv_v4 = bt_mv_v4.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)


# Concatenate the simulations
sim = pd.concat({
    'bm': bs.data.bm_series,
    'mv': sim_mv,
    'mv_v1': sim_mv_v1,
    # 'mv_v2': sim_mv_v2,
    # 'mv_v3': sim_mv_v3,
    # 'mv_v4': sim_mv_v4,
}, axis = 1).dropna()
sim.columns = [
    'Benchmark',
    'Mean-Variance',
    'Mean-Variance, RA5',
    # 'Mean-Variance, RA5, Gaps',
    # 'Mean-Variance, RA5, Gaps, Liq',
    # 'Mean-Variance, RA5, Gaps, Liq, SDUB'
]

# Plot the cumulative performance
np.log((1 + sim)).cumsum().plot(title='Cumulative Performance', figsize = (10, 6))




# Out-/Underperformance
def sim_outperformance(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    ans = (x.subtract(y, axis=0)).divide(1 + y, axis=0)
    return ans

sim_rel = sim_outperformance(sim, sim['Benchmark'])

np.log((1 + sim_rel)).cumsum().plot(title='Cumulative Out-/Underperformance', figsize = (10, 6))







# --------------------------------------------------------------------------
# Turnover
# --------------------------------------------------------------------------

strat_dict = {
    'mv': bt_mv.strategy,
    'mv_v1': bt_mv_v1.strategy,
    # 'mv_v2': bt_mv_v2.strategy,
    # 'mv_v3': bt_mv_v3.strategy,
    # 'mv_v4': bt_mv_v4.strategy,
}
to_dict = {
    key: strategy.turnover(return_series=return_series)
    for key, strategy in strat_dict.items()
}
to = pd.concat(to_dict, axis=1).dropna()

to.plot(title='Turnover', figsize = (10, 6))
to.mean() * 4





# --------------------------------------------------------------------------
# Decriptive statistics
# --------------------------------------------------------------------------

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



