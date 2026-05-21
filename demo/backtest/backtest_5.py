############################################################################
### QPMwP CODING EXAMPLES - Backtest 5: Turnover Constraints and Penalties
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     13.03.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------



# This script demonstrates how to run a backtest using the qpmwp-course library
# and single stock data which change over time.

# The script uses the 'MeanVariance' portfolio optimization class and implements
# a turnover constraint as well as a turnover penalty.







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
from optimization.optimization import MeanVariance
from backtesting.backtest_item_builder.bib_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder.bibfn_selection import (
    bibfn_selection_gaps,
    bibfn_selection_min_volume,
)
from backtesting.backtest_item_builder.bibfn_optimization_data import (
    bibfn_return_series,
)
from backtesting.backtest_item_builder.bibfn_constraints import (
    bibfn_budget_constraint,
    bibfn_box_constraints,
    bibfn_size_dependent_upper_bounds,
    bibfn_turnover_constraint,                            # NEW
)
from backtesting.backtest_data import BacktestData
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest





# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PATH_TO_DATA = '...'     # <change this to your path to data>
SAVE_PATH = '...'        # <change this to your path where you want to store the backtest>
WIDTH_3Y = 365 * 3




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

selection_item_builders = {
    'gaps': SelectionItemBuilder(
        bibfn=bibfn_selection_gaps,
        width=WIDTH_3Y,
        n_days=10,
    ),
    'min_volume': SelectionItemBuilder(
        bibfn=bibfn_selection_min_volume,
        width=WIDTH_3Y,
        min_volume=500_000,
        agg_fn=np.median,
    ),
}




# -------------------------
# Define the optimization item builders.
# -------------------------

optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn=bibfn_return_series,
        width=WIDTH_3Y,
        fill_value=0,
    ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn=bibfn_budget_constraint,
        budget=1,
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn=bibfn_box_constraints,
        upper=0.1,
    ),
    'size_dep_upper_bounds': OptimizationItemBuilder(
        bibfn=bibfn_size_dependent_upper_bounds,
        small_cap={'threshold': 300_000_000, 'upper': 0.02},
        mid_cap={'threshold': 1_000_000_000, 'upper': 0.05},
        large_cap={'threshold': 10_000_000_000, 'upper': 0.1},
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
# Run backtest: Mean-Variance with turnover constraint
# --------------------------------------------------------------------------

# Create a new backtest service with the turnover constraint in the optimization item builders
bs = BacktestService(
    data=data,
    optimization=MeanVariance(
        covariance=Covariance(method='pearson'),
        expected_return=ExpectedReturn(method='geometric'),
        risk_aversion=1,
        solver_name='cvxopt',
    ),
    selection_item_builders=selection_item_builders,
    optimization_item_builders={
        **optimization_item_builders,
        'turnover_constraint': OptimizationItemBuilder(
            bibfn=bibfn_turnover_constraint,
            turnover_limit=0.25,
        ),
    },
    rebdates=rebdates,
)

# Instantiate the backtest object
bt_mv_to_cons = Backtest()

# Run the backtest
bt_mv_to_cons.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_mv_to_cons.save(
#     path=SAVE_PATH,
#     filename='backtest_mv_to_cons.pickle' # <change this to your desired filename>
# )






# --------------------------------------------------------------------------
# Run backtest: Mean-Variance with turnover penalty in the objective function
# --------------------------------------------------------------------------

# In order to run a backtest with a turnover penalty, the source code was updated in the following files:

# - src/backtesting/backtest_service.py:
#   Extend method build_optimization to calculate the initial weight vector x_init

# - src/optimization/optimization.py:
#   Within method model_qpsolvers, call method linearize_turnover_objective
#   of class QuadraticProgram.

# - src/optimization/quadratic_program.py:
#   Add method linearize_turnover_objective to class QuadraticProgram.


# Update the backtest service with a MeanVariance optimization object
bs.optimization = MeanVariance(
    covariance=Covariance(method='pearson'),
    expected_return=ExpectedReturn(method='geometric'),
    risk_aversion=1,
    solver_name='cvxopt',
    turnover_penalty=1,  # Turnover penalty in the objective function
)

# Instantiate the backtest object
bt_mv_to_pnlty = Backtest()

# Run the backtest
bt_mv_to_pnlty.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_mv_to_pnlty.save(
#     path=SAVE_PATH,
#     filename='backtest_mv_to_pnlty.pickle' # <change this to your desired filename>
# )







# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------


# Laod backtests from pickle
bt_mv = load_pickle(
    filename='backtest_mv.pickle',
    path=SAVE_PATH,
)
bt_mv_to_cons = load_pickle(
    filename='backtest_mv_to_cons.pickle',
    path=SAVE_PATH,
)
bt_mv_to_pnlty = load_pickle(
    filename='backtest_mv_to_pnlty.pickle',
    path=SAVE_PATH,
)



# fixed_costs = 0.01
fixed_costs = 0
variable_costs = 0.004
return_series = bs.data.get_return_series()

strategy_dict = {
    'mv': bt_mv.strategy,
    'mv_to_cons': bt_mv_to_cons.strategy,
    'mv_to_pnlty': bt_mv_to_pnlty.strategy,
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

to_mv = bt_mv.strategy.turnover(return_series=return_series)
to_mv_to_cons = bt_mv_to_cons.strategy.turnover(return_series=return_series)
to_mv_to_pnlty = bt_mv_to_pnlty.strategy.turnover(return_series=return_series)

to = pd.concat({
    'mv': to_mv,
    'mv_to_cons': to_mv_to_cons,
    'mv_to_pnlty': to_mv_to_pnlty,
}, axis = 1).dropna()
to.columns = [
    'Mean-Variance',
    'Mean-Variance with Turnover Constraint',
    'Mean-Variance with Turnover Penalty'
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
    tracking_error[column] = ep.annual_volatility(sim[column] - sim['bm'])


annual_returns = pd.DataFrame(annual_return, index=['Annual Return'])
cumret = pd.DataFrame(cumulative_returns, index=['Cumulative Return'])
annual_volatility = pd.DataFrame(annual_volatility, index=['Annual Volatility'])
sharpe  = pd.DataFrame(sharpe_ratio, index=['Sharpe Ratio'])
mdd = pd.DataFrame(max_drawdown, index=['Max Drawdown'])
pd.concat([annual_returns, cumret, annual_volatility, sharpe, mdd])
