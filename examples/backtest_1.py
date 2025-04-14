############################################################################
### QPMwP CODING EXAMPLES - Backtest 1
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.03.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------




# This script demonstrates how to run a backtest using the qpmwp library
# and with data from MSCI Country Indices (which do not change over time).
# The script uses the 'MeanVariance' and portfolio optimization class.






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
from optimization.constraints import Constraints
from optimization.optimization import MeanVariance
from backtesting.backtest import Backtest
from backtesting.backtest_service import BacktestService
from backtesting.backtest_item_builder_classes import (
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    bibfn_return_series,
)





# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

N = 10
data = load_data_msci(path = '../data/', n = N)
data




# --------------------------------------------------------------------------
# Instantiate the expected return and covariance classes
# --------------------------------------------------------------------------

expected_return = ExpectedReturn(method='geometric')
covariance = Covariance(method='pearson')






# --------------------------------------------------------------------------
# Prepare the constraints object
# --------------------------------------------------------------------------

# Instantiate the class
constraints = Constraints(ids = data['return_series'].columns.tolist())

# Add budget constraint
constraints.add_budget(rhs=1, sense='=')

# Add box constraints (i.e., lower and upper bounds)
constraints.add_box(lower=0, upper=0.5)

# Add linear constraints
G = pd.DataFrame(np.zeros((2, N)), columns=constraints.ids)
G.iloc[0, 0:5] = 1
G.iloc[1, 6:10] = 1
h = pd.Series([0.5, 0.5])
constraints.add_linear(G=G, sense='<=', rhs=h)


constraints.budget
constraints.box
constraints.linear









# --------------------------------------------------------------------------
# Initiate the optimization object
# --------------------------------------------------------------------------

# Instantiate the optimization object as an instance of MeanVariance
optimization = MeanVariance(
    covariance = covariance,
    expected_return = expected_return,
    constraints = constraints,
    risk_aversion = 1,
    solver_name = 'cvxopt',
)






# --------------------------------------------------------------------------
# Prepare the backtest service
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
# Additional keyword arguments can be passed to bibfn using the arguments attribute of the
# OptimizationItemBuilder instance.

optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn = bibfn_return_series,
        width = 256 * 3,
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
# Run the backtest for the mean-variance model
# --------------------------------------------------------------------------

# Instantiate the backtest object
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
    title='Mean-Variance Portfolio Weights'
)







# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

fixed_costs = 0
variable_costs = 0
return_series = bs.data['return_series']

sim_mv = bt_mv.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs,
)

sim = pd.concat({
    'bm': bs.data['bm_series'],
    'mv': sim_mv,
}, axis = 1).dropna()
sim.columns = sim.columns.get_level_values(0)


np.log((1 + sim)).cumsum().plot(title='Cumulative Performance', figsize = (10, 6))




# Out-/Underperformance

def sim_outperformance(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    ans = (x.subtract(y, axis=0)).divide(1 + y, axis=0)
    return ans

sim_rel = sim_outperformance(sim, sim['bm'])

np.log((1 + sim_rel)).cumsum().plot(title='Cumulative Out-/Underperformance', figsize = (10, 6))








# --------------------------------------------------------------------------
# Decriptive statistics
# --------------------------------------------------------------------------

# pip install empyrical
import empyrical as ep

# Load your strategy returns here
strategy_returns = sim['mv']  # Replace with your actual returns data

# Compute various performance metrics
annual_return = ep.annual_return(strategy_returns)
cumulative_returns = ep.cum_returns(strategy_returns)
sharpe_ratio = ep.sharpe_ratio(strategy_returns)
max_drawdown = ep.max_drawdown(strategy_returns)
alpha_beta = ep.alpha_beta(strategy_returns, sim['bm'])

# Print the performance metrics
print(f'Annual Return: {annual_return}')
print(f'Cumulative Returns: {cumulative_returns.iloc[-1]}')
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Max Drawdown: {max_drawdown}')
print(f'Alpha: {alpha_beta[0]}, Beta: {alpha_beta[1]}')



# pip install quantstats
import quantstats as qs

# Load your strategy returns here
strategy_returns = sim['mv']

# Generate a full report
qs.reports.full(strategy_returns, benchmark=sim['bm'])

# Alternatively, you can generate a simple report
qs.reports.basic(strategy_returns, benchmark=sim['bm'])

# Compute individual performance metrics
sharpe_ratio = qs.stats.sharpe(strategy_returns)
max_drawdown = qs.stats.max_drawdown(strategy_returns)

# Print the performance metrics
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Max Drawdown: {max_drawdown}')


















