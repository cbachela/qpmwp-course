############################################################################
### QPMwP CODING EXAMPLES - Backtest 4
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     31.03.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Make sure to install the following packages before running the demo:

# pip install pyarrow fastparquet   # For reading and writing parquet files


# This script builds upon the mean-variance optimization example (backtest_4.py)
# and tries improve the performance.






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
from optimization.optimization import (
    MeanVariance,
)
from backtesting.backtest_item_builder_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    bibfn_selection_min_volume,
    bibfn_return_series,
    bibfn_budget_constraint,
    bibfn_box_constraints,
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

# Instantiate the BacktestData class
# and set the market data and jkp data as attributes
data = BacktestData()
data.market_data = market_data
data.bm_series = load_data_spi(path='../data/')





# --------------------------------------------------------------------------
# Prepare backtest service
# --------------------------------------------------------------------------

# Define rebalancing dates
n_days = 21*3  # Rebalance every n_month months
market_data_dates = market_data.index.get_level_values('date').unique().sort_values()
rebdates = market_data_dates[market_data_dates > '2005-01-01'][::n_days].strftime('%Y-%m-%d').tolist()
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
    'min_volume': SelectionItemBuilder(
        bibfn = bibfn_selection_min_volume,   # filter out illiquid stocks
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
        width = 252 * 3,
        fill_value = 0,
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
# Base model: mean-variance portfolio
# --------------------------------------------------------------------------

# Update the backtest service with a MeanVariance optimization object
bs.optimization = MeanVariance(
    covariance = Covariance(method = 'pearson'),
    expected_return = ExpectedReturn(method = 'geometric'),
    risk_aversion = 1,
    solver_name = 'cvxopt',
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
# Adjustment 1: mean-variance portfolio with risk aversion of 5
# --------------------------------------------------------------------------

# Update the backtest service with a MeanVariance optimization object
bs.optimization = MeanVariance(
    covariance = Covariance(method = 'pearson'),
    expected_return = ExpectedReturn(method = 'geometric'),
    risk_aversion = 5,
    solver_name = 'cvxopt',
)

# Instantiate the backtest object and run the backtest
bt_mv_ra5 = Backtest()

# Run the backtest
bt_mv_ra5.run(bs = bs)

# Save the backtest as a .pickle file
# bt_mv_ra5.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_mv_ra5.pickle' # <change this to your desired filename>
# )





# --------------------------------------------------------------------------
# Adjustment 2: gaps filter
# --------------------------------------------------------------------------


def bibfn_selection_gaps(bs, rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection.
    Drops elements from the selection when there is a gap
    of more than n_days (i.e., consecutive zero's) in the volume series.
    '''

    # Arguments
    width = kwargs.get('width', 252)
    n_days = kwargs.get('n_days', 21)

    # Volume data
    vol = (
        bs.data.get_volume_series(
            end_date=rebdate,
            width=width
        ).fillna(0)
    )

    # Calculate the length of the longest consecutive zero sequence
    def consecutive_zeros(column):
        return (column == 0).astype(int).groupby(column.ne(0).astype(int).cumsum()).sum().max()

    gaps = vol.apply(consecutive_zeros)

    # Output
    filter_values = pd.DataFrame({
        'values': gaps,
        'binary': (gaps <= n_days).astype(int),
    }, index=gaps.index)

    return filter_values


# Add the gaps filter to the selection_item_builders dictionary
selection_item_builders['gaps'] = SelectionItemBuilder(
    bibfn = bibfn_selection_gaps,
    width = 252 * 3,
    n_days = 10  # filter out stocks which have not been traded for more than 'n_days' consecutive days
)

# Reinitialize the backtest service with the gaps filter
bs = BacktestService(
    data = data,
    optimization = MeanVariance(
        covariance = Covariance(method = 'pearson'),
        expected_return = ExpectedReturn(method = 'geometric'),
        risk_aversion = 5,
        solver_name = 'cvxopt',
    ),
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates,
)

# Instantiate the backtest object and run the backtest
bt_mv_ra5_gaps = Backtest()

# Run the backtest
bt_mv_ra5_gaps.run(bs = bs)

# # Save the backtest as a .pickle file
# bt_mv_ra5_gaps.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_mv_ra5_gaps.pickle' # <change this to your desired filename>
# )

bs.selection.df()
bs.selection.df_binary().sum()



# --------------------------------------------------------------------------
# Adjustment 3: Size-dependent upper bounds
# --------------------------------------------------------------------------

def bibfn_size_dependent_upper_bounds(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the upper bounds
    in dependence of a stock's market capitalization.
    '''

    # Arguments
    small_cap = kwargs.get('small_cap', {'threshold': 300_000_000, 'upper': 0.02})
    mid_cap = kwargs.get('small_cap', {'threshold': 1_000_000_000, 'upper': 0.05})
    large_cap = kwargs.get('small_cap', {'threshold': 10_000_000_000, 'upper': 0.1})

    # Selection
    ids = bs.optimization.constraints.ids

    # Data: market capitalization
    mcap = bs.data.market_data['mktcap']
    # Get last available valus for current rebdate
    mcap = mcap[mcap.index.get_level_values('date') <= rebdate].groupby(
        level = 'id'
    ).last()

    # Remove duplicates
    mcap = mcap[~mcap.index.duplicated(keep=False)]
    # Ensure that mcap contains all selected ids,
    # possibly extend mcap with zero values
    mcap = mcap.reindex(ids).fillna(0)

    # Generate the upper bounds
    upper = mcap * 0
    upper[mcap > small_cap['threshold']] = small_cap['upper']
    upper[mcap > mid_cap['threshold']] = mid_cap['upper']
    upper[mcap > large_cap['threshold']] = large_cap['upper']

    # Check if the upper bounds have already been set
    if not bs.optimization.constraints.box['upper'].empty:
        bs.optimization.constraints.add_box(
            box_type = 'LongOnly',
            upper = upper,
        )
    else:
        # Update the upper bounds by taking the minimum of the current and the new upper bounds
        bs.optimization.constraints.box['upper'] = np.minimum(
            bs.optimization.constraints.box['upper'],
            upper,
        )

    return None



# Add the size-dependent upper bounds to the optimization_item_builders dictionary
optimization_item_builders['size_dep_upper_bounds'] = OptimizationItemBuilder(
    bibfn = bibfn_size_dependent_upper_bounds,
    small_cap = {'threshold': 300_000_000, 'upper': 0.02},
    mid_cap = {'threshold': 1_000_000_000, 'upper': 0.05},
    large_cap = {'threshold': 10_000_000_000, 'upper': 0.1},
)

# Reinitialize the backtest service with the size-dependent upper bounds
bs = BacktestService(
    data = data,
    optimization = MeanVariance(
        covariance = Covariance(method = 'pearson'),
        expected_return = ExpectedReturn(method = 'geometric'),
        risk_aversion = 5,
        solver_name = 'cvxopt',
    ),
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates[10:],
)

# Instantiate the backtest object and run the backtest
bt_mv_ra5_gaps_sdub = Backtest()

# Run the backtest
bt_mv_ra5_gaps_sdub.run(bs = bs)

# # Save the backtest as a .pickle file
# bt_mv_ra5_gaps_sdub.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_mv_ra5_gaps_sdub.pickle' # <change this to your desired filename>
# )











# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------

path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/' #<change this to your local path>

# Laod backtests from pickle
bt_mv = load_pickle(
    filename = 'backtest_mv.pickle',
    path = path,
)
bt_mv_ra5 = load_pickle(
    filename = 'backtest_mv_ra5.pickle',
    path = path,
)
bt_mv_ra5_gaps = load_pickle(
    filename = 'backtest_mv_ra5_gaps.pickle',
    path = path,
)
bt_mv_ra5_gaps_sdub = load_pickle(
    filename = 'backtest_mv_ra5_gaps_sdub.pickle',
    path = path,
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
sim_mv_ra5 = bt_mv_ra5.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)
sim_mv_ra5_gaps = bt_mv_ra5_gaps.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)
sim_mv_ra5_gaps_sdub = bt_mv_ra5_gaps_sdub.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)


# Concatenate the simulations
sim = pd.concat({
    'bm': bs.data.bm_series,
    'mv': sim_mv,
    'mv_ra5': sim_mv_ra5,
    'mv_ra5_gaps': sim_mv_ra5_gaps,
    'mv_ra5_gaps_sdub': sim_mv_ra5_gaps_sdub,
}, axis = 1).dropna()
sim.columns = ['Benchmark', 'Mean-Variance', 'Mean-Variance RA5', 'Mean-Variance RA5 Gaps', 'Mean-Variance RA5 Gaps SDUB']

# Plot the cumulative performance
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



