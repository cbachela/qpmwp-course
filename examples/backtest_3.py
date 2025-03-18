############################################################################
### QPMwP CODING EXAMPLES - Backtest 3
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.03.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------




# This script demonstrates how to run a backtest using the qpmwp library
# and with data from MSCI Country Indices (which do not change over time).
# The script uses the 'MeanVariance' portfolio optimization class.
#
# The difference to example backtest_1.py is that
# - the assets considered for the optimization, (i.e., the selection) varies over time, and hence,
# - the constraints need to be (re-)defined at each rebalancing date.





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
from estimation.covariance import Covariance
from estimation.expected_return import ExpectedReturn
from optimization.optimization import MeanVariance
from backtesting.backtest_item_builder_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    bibfn_selection_data_random,
    bibfn_return_series,
    bibfn_budget_constraint,
    bibfn_box_constraints,
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
# Instantiate the expected return and covariance classes
# --------------------------------------------------------------------------

expected_return = ExpectedReturn(method='geometric')
covariance = Covariance(method='pearson')





# --------------------------------------------------------------------------
# Initiate the optimization object
# --------------------------------------------------------------------------

# Instantiate the optimization object as an instance of MeanVariance
# Notice that we do not pass any constraints here since those are
# defined at each rebalancing date.

optimization = MeanVariance(
    covariance = covariance,
    expected_return = expected_return,
    risk_aversion = 1,
    solver_name = 'cvxopt',
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




# Define the selection item builders.

# SelectionItemBuilder is a callable class which takes a function (bibfn) as argument.
# The function bibfn is a custom function that builds a selection item, i.e. a
# pandas Series of boolean values indicating the selected assets at a given rebalancing date.

# The function bibfn takes the backtest service (bs) and the rebalancing date (rebdate) as arguments.
# Additional keyword arguments can be passed to bibfn using the arguments attribute of the SelectionItemBuilder instance.

# The selection item is then added to the Selection attribute of the backtest service using the add_item method.
# To inspect the current instance of the selection object, type bs.selection.df()


selection_item_builders = {
    'data': SelectionItemBuilder(
        bibfn = bibfn_selection_data_random,
        k = 10,
    ),
}







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
    'budget_constraint': OptimizationItemBuilder(
        bibfn = bibfn_budget_constraint,
        budget = 1
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn = bibfn_box_constraints,
        lower = 0,
        upper = 1,
    ),
}





# Initialize the backtest service
bs = BacktestService(
    data = data,
    optimization = optimization,
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates,
)







# --------------------------------------------------------------------------
# Run backtests
# --------------------------------------------------------------------------

# Instantiate the backtest object and run the backtest
bt_mv = Backtest()

# Run the backtest
bt_mv.run(bs = bs)






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
bt_mv.strategy.get_weights_df()
bt_mv.strategy.get_weights_df().plot(
    kind='bar', stacked=True, figsize=(10, 6),
    title='Mean-Variance Portfolio Weights (random selection)'
)







# --------------------------------------------------------------------------
# Inspect the optimization for a specific rebalancing date
# --------------------------------------------------------------------------

# Prepare the optimization for a specific rebalancing date
# by calling the prepare_rebalancing method of the backtest service
# and passing the rebalancing date as argument.

bs.prepare_rebalancing(rebalancing_date='2010-01-04')

bs.selection.df()

bs.optimization.constraints.ids
bs.optimization.constraints.box
bs.optimization.constraints.budget







# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------

fixed_costs = 0
variable_costs = 0
return_series = bs.data['return_series']

sim_mv = bt_mv.strategy.simulate(return_series = return_series, fc = fixed_costs, vc = variable_costs)

sim = pd.concat({
    'bm': bs.data['bm_series'],
    'mv': sim_mv,
}, axis = 1).dropna()
sim.columns = ['Benchmark', 'Mean-Variance']

np.log((1 + sim)).cumsum().plot(title='Cumulative Performance', figsize = (10, 6))



# Out-/Underperformance

def sim_outperformance(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    ans = (x.subtract(y, axis=0)).divide(1 + y, axis=0)
    return ans

sim_rel = sim_outperformance(sim, sim['Benchmark'])

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
pd.concat([annual_returns, cumret, annual_volatility, sharpe, mdd])








