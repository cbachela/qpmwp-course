############################################################################
### QPMwP CODING EXAMPLES - Backtest 2 - Index Tracking
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.03.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------




# This script demonstrates how to run a backtest using the qpmwp library 
# and with data from MSCI Country Indices (which do not change over time).
# The script uses the 'LeastSquares' portfolio optimization class.





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
from helper_functions import load_data_msci
from optimization.constraints import Constraints
from optimization.optimization import LeastSquares
from backtesting.backtest_item_builder_classes import (
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    bibfn_return_series,
    bibfn_bm_series,
)
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest





# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

N = 24
data = load_data_msci(path = '../data/', n = N)
data






# --------------------------------------------------------------------------
# Prepare the constraints object
# --------------------------------------------------------------------------

# Instantiate the class
constraints = Constraints(ids = data['return_series'].columns.tolist())

# Add budget constraint
constraints.add_budget(rhs=1, sense='=')

# Add box constraints (i.e., lower and upper bounds)
constraints.add_box(lower=0, upper=1)

# # Add linear constraints
# G = pd.DataFrame(np.zeros((2, N)), columns=constraints.ids)
# G.iloc[0, 0:5] = 1
# G.iloc[1, 6:10] = 1
# h = pd.Series([0.5, 0.5])
# constraints.add_linear(G=G, sense='<=', rhs=h)


constraints.budget
constraints.box
constraints.linear







# --------------------------------------------------------------------------
# Initiate the optimization object that will be used in the backtest
# Here we use the LeastSquares class which solves a tracking error minimization problem
# --------------------------------------------------------------------------

# Instantiate the optimization object as an instance of LeastSquares
optimization = LeastSquares(
    constraints = constraints,
    solver_name = 'cvxopt'
)




# --------------------------------------------------------------------------
# Prepare backtest service
# --------------------------------------------------------------------------

# Define rebalancing dates
n_days = 21 * 3
start_date = '2010-01-01'
dates = data['return_series'].index
rebdates = dates[dates > start_date][::n_days].strftime('%Y-%m-%d').tolist()
rebdates



# Define the optimization item builders.

# OptimizationItemBuilder is a callable class which takes a function (bibfn) as argument.
# The function bibfn is a custom function that builds an item which is used for the optimization.

# Such items can be constraints, which are added to the constraints attribute of the optimization object,
# or datasets which are added to the instance of the OptimizationData class.

# The function bibfn takes the backtest service (bs) and the rebalancing date (rebdate) as arguments.
# Additional keyword arguments can be passed to bibfn using the arguments attribute of the OptimizationItemBuilder instance.


optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn = bibfn_return_series,
        width = 365 * 3,
        name = 'return_series',
    ),
    'bm_series': OptimizationItemBuilder(
        bibfn = bibfn_bm_series,
        width = 365 * 3,
        align = False,
        name = 'bm_series',
    ),
}


# Initialize the backtest service
bs = BacktestService(
    data = data,
    optimization = optimization,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates,
)







# --------------------------------------------------------------------------
# Run backtests
# --------------------------------------------------------------------------

# Instantiate the backtest object
bt_ls = Backtest()

# Run the backtest
bt_ls.run(bs = bs)






# --------------------------------------------------------------------------
# Inspect the optimization data, constraints and results
# --------------------------------------------------------------------------

# Inspect the optimization data for the last rebalancing date
bs.optimization_data

# Inspect the optimization constraints
bs.optimization.constraints.budget
bs.optimization.constraints.box
bs.optimization.constraints.linear

# Inspect the optimization results - i.e. the weights stored in the strategy object
bt_ls.strategy.get_weights_df()
bt_ls.strategy.get_weights_df().plot(
    kind='bar', stacked=True, figsize=(10, 6),
    title='Minimum Tracking Error (Least-Squares) Portfolio Weights'
)










# --------------------------------------------------------------------------
# Let's try to be smart:
# Backtest the minimum tracking error portfolio (based on the least squares formulation)
# to a slightly leveraged (1.3x) benchmark, while not allowing any leverage for the portfolio
# --------------------------------------------------------------------------

# Add a 1.3x leveraged benchmark series to the data
bs.data['bm_series_1.3x'] = bs.data['bm_series'] * 1.3

# Replace the optimization item builder which prepares the benchmarks series
# with a new one for the 1.3x levered benchmark series
optimization_item_builders['bm_series'] = OptimizationItemBuilder(
    bibfn = bibfn_bm_series,
    width = 365 * 3,
    align = False,
    name = 'bm_series_1.3x',
)


# Initialize the backtest service
bs = BacktestService(
    data = data,
    optimization = optimization,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates,
)

# Instantiate the backtest object
bt_ls_x = Backtest()

# Run the backtest
bt_ls_x.run(bs = bs)










# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------

fixed_costs = 0
variable_costs = 0
return_series = bs.data['return_series']

sim_ls = bt_ls.strategy.simulate(return_series = return_series, fc = fixed_costs, vc = variable_costs)
sim_ls_x = bt_ls_x.strategy.simulate(return_series = return_series, fc = fixed_costs, vc = variable_costs)


sim = pd.concat({
    'bm': bs.data['bm_series'],
    'bm_1.3x': bs.data['bm_series'] * 1.3,
    'ls': sim_ls,
    'ls_x': sim_ls_x,
}, axis = 1).dropna()
sim.columns = ['Benchmark', 'Benchmark 1.3x', 'Tracking Portfolio', 'Enhanced Tracking Portfolio']


np.log((1 + sim)).cumsum().plot(title='Cumulative Performance', figsize = (10, 6))




# Out-/Underperformance

def sim_outperformance(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    ans = (x.subtract(y, axis=0)).divide(1 + y, axis=0)
    return ans

sim_rel = sim_outperformance(sim[['Benchmark', 'Enhanced Tracking Portfolio']], sim['Benchmark'])

np.log((1 + sim_rel)).cumsum().plot(title='Cumulative Out-/Underperformance', figsize = (10, 6))






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













