############################################################################
### QPMwP CODING EXAMPLES - Backtest 5
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     08.04.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Make sure to install the following packages before running the demo:

# pip install pyarrow fastparquet   # For reading and writing parquet files


# This script demonstrates how to run a backtest using the qpmwp library
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
from helper_functions import load_data_spi, load_pickle
from estimation.covariance import Covariance
from estimation.expected_return import ExpectedReturn
from optimization.optimization import MeanVariance
from backtesting.backtest_item_builder_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    bibfn_selection_min_volume,
    bibfn_selection_gaps,
    bibfn_return_series,
    bibfn_budget_constraint,
    bibfn_box_constraints,
    bibfn_size_dependent_upper_bounds,
)
from backtesting.backtest_data import BacktestData
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest





# --------------------------------------------------------------------------
# Load data
# - market data (from parquet file)
# - swiss performance index, SPI (from csv file)
# --------------------------------------------------------------------------

path_to_data = 'C:/Users/User/OneDrive/Documents/QPMwP/Data/'  # <change this to your path to data>

# Load market and jkp data from parquet files
market_data = pd.read_parquet(path = f'{path_to_data}market_data.parquet')

# Instantiate the BacktestData class
# and set the market data and jkp data as attributes
data = BacktestData()
data.market_data = market_data
data.bm_series = load_data_spi(path='../data/')







# --------------------------------------------------------------------------
# Prepare backtest service
# --------------------------------------------------------------------------

# Parameters (used more than once below)
expected_return = ExpectedReturn(method = 'geometric')
covariance = Covariance(method = 'pearson')
risk_aversion = 1
solver_name = 'cvxopt'


# Define rebalancing dates
n_days = 21*3  # Rebalance every n_days
market_data_dates = market_data.index.get_level_values('date').unique().sort_values(ascending=True)
rebdates = market_data_dates[market_data_dates > '2005-01-01'][::n_days].strftime('%Y-%m-%d').tolist()
# rebdates = market_data_dates[market_data_dates > '2022-01-01'][::n_days].strftime('%Y-%m-%d').tolist()
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
    'gaps': SelectionItemBuilder(
        bibfn = bibfn_selection_gaps,
        width = 252*3,
        n_days = 10  # filter out stocks which have not been traded for more than 'n_days' consecutive days
    ),
    'min_volume': SelectionItemBuilder(
        bibfn = bibfn_selection_min_volume,   # filter stocks which are illiquid
        width = 252,
        min_volume = 500_000,
        agg_fn = np.median,
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
        width = 252,
        fill_value = 0,
    ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn = bibfn_budget_constraint,
        budget = 1,
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn = bibfn_box_constraints,
        upper = 0.1,
    ),
    'size_dep_upper_bounds': OptimizationItemBuilder(
        bibfn = bibfn_size_dependent_upper_bounds,
        small_cap = {'threshold': 300_000_000, 'upper': 0.02},
        mid_cap = {'threshold': 1_000_000_000, 'upper': 0.05},
        large_cap = {'threshold': 10_000_000_000, 'upper': 0.1},
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
# Run backtest: Mean-Variance
# --------------------------------------------------------------------------

# Update the backtest service with a MeanVariance optimization object
bs.optimization = MeanVariance(
    covariance = covariance,
    expected_return = expected_return,
    risk_aversion = risk_aversion,
    solver_name = solver_name,
)

# Instantiate the backtest object and run the backtest
bt_mv = Backtest()

# Run the backtest
bt_mv.run(bs = bs)

# # Save the backtest as a .pickle file
# bt_mv.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_mv.pickle' # <change this to your desired filename>
# )






# --------------------------------------------------------------------------
# Run backtest: Mean-Variance with turnover constraint
# --------------------------------------------------------------------------


def bibfn_turnover_constraint(bs, rebdate: str, **kwargs) -> None:
    """
    Function to assign a turnover constraint to the optimization.
    """
    if rebdate > bs.settings['rebdates'][0]:

        # Arguments
        turnover_limit = kwargs.get('turnover_limit')

        # Constraints
        bs.optimization.constraints.add_l1(
            name = 'turnover',
            rhs = turnover_limit,
            x0 = bs.optimization.params['x_init'],
        )

    return None



# Update the backtest service with a MeanVariance optimization object
bs = BacktestService(
    data = data,
    optimization = MeanVariance(
        covariance = covariance,
        expected_return = expected_return,
        risk_aversion = risk_aversion,
        solver_name = solver_name,
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

# Instantiate the backtest object
bt_mv_to_cons = Backtest()

# Run the backtest
bt_mv_to_cons.run(bs = bs)

# # Save the backtest as a .pickle file
# bt_mv_to_cons.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_mv_to_cons.pickle' # <change this to your desired filename>
# )






# --------------------------------------------------------------------------
# Run backtest: Mean-Variance with turnover penalty in the objective function
# --------------------------------------------------------------------------

# In order to run a backtest with a turnover penalty, we needed to update the
# source code at the following locations:

# - src/backtesting/backtest_sercice.py:
#   Extend method build_optimization to calculate the initial weight vector x_init

# - src/optimization/optimization.py:
#   Within method model_qpsolvers, call method linearize_turnover_objective
#   of class QuadraticProgram.

# - src/optimization/quadratic_program.py:
#   Add method linearize_turnover_objective to class QuadraticProgram.


# Update the backtest service with a MeanVariance optimization object
bs.optimization = MeanVariance(
    covariance = covariance,
    expected_return = expected_return,
    risk_aversion = risk_aversion,
    solver_name = solver_name,
    turnover_penalty = 0.001,  # Turnover penalty in the objective function
)

# Instantiate the backtest object
bt_mv_to_pnlty = Backtest()

# Run the backtest
bt_mv_to_pnlty.run(bs = bs)

# # Save the backtest as a .pickle file
# bt_mv_to_pnlty.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_mv_to_pnlty.pickle' # <change this to your desired filename>
# )






# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------


# Laod backtests from pickle
path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/' #<change this to your local path>

bt_mv = load_pickle(
    filename = 'backtest_mv.pickle',
    path = path,
)
bt_mv_to_cons = load_pickle(
    filename = 'backtest_mv_to_cons.pickle',
    path = path,
)
bt_mv_to_pnlty = load_pickle(
    filename = 'backtest_mv_to_pnlty.pickle',
    path = path,
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
