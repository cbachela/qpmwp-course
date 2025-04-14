############################################################################
### QPMwP CODING EXAMPLES - Backtest 4
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     24.03.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Make sure to install the following packages before running the demo:

# pip install pyarrow fastparquet   # For reading and writing parquet files


# This script demonstrates how to run a backtest using the qpmwp library
# and single stock data which change over time.
# The script uses the 'MeanVariance' portfolio optimization class.







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
from backtesting.backtest_item_builder_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    bibfn_selection_min_volume,
    bibfn_selection_NA,
    bibfn_return_series,
    bibfn_bm_series,
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
jkp_data = pd.read_parquet(path = f'{path_to_data}jkp_data.parquet')

# Instantiate the BacktestData class
# and set the market data and jkp data as attributes
data = BacktestData()
data.market_data = market_data
data.jkp_data = jkp_data
data.bm_series = load_data_spi(path='../data/')



# Helper methods that extract the certain columns from the data objects
# from long format to wide format

??data.get_return_series
data.get_return_series()

??data.get_volume_series
data.get_volume_series()

??data.get_characteristic_series
data.get_characteristic_series(
    field = 'qmj',
)

# Plot the density of the qmj characteristic (z-scores) for the last available date
qmj = data.get_characteristic_series(
    field = 'qmj',
).tail(1).squeeze()
qmj
qmj.plot(kind='density', title='Density of qmj')



    


# --------------------------------------------------------------------------
# Prepare backtest service
# --------------------------------------------------------------------------


# Define rebalancing dates
n_month = 3 # Rebalance every n_month months
market_data_dates = market_data.index.get_level_values('date').unique().sort_values()
jkp_data_dates = jkp_data.index.get_level_values('date').unique().sort_values()
rebdates = jkp_data_dates[jkp_data_dates > market_data_dates[0]][::n_month].strftime('%Y-%m-%d').tolist()
rebdates = [date for date in rebdates if date > '2002-01-01']
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
        bibfn = bibfn_selection_min_volume,   # filter out stocks which are illiquid
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



def bibfn_bm_relative_upper_bounds(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the upper bounds
    in dependence of a stock's market capitalization.
    '''

    # Arguments
    multiple = kwargs.get('multiple', 20)

    # Selection
    # ids = bs.selection.selected
    ids = bs.optimization.constraints.ids

    # Data: market capitalization
    mcap = bs.data.market_data['mktcap']
    
    
    # Get last available valus for current rebdate
    mcap[mcap.index.get_level_values('date') <= rebdate].tail(252).groupby(
        level = 'id'
    ).mean()    

    # Remove duplicates
    mcap = mcap[~mcap.index.duplicated(keep=False)]
    # Ensure that mcap contains all selected ids,
    # possibly extend mcap with zero values
    mcap = mcap.reindex(ids).fillna(0)

    # Compute the upper bounds
    upper = (mcap / mcap.sum()) * multiple

    # Check if the upper bounds have already been set
    boxcon = bs.optimization.constraints.box
    if not boxcon['upper']:
        bs.optimization.constraints.add_box(
            box_type = 'LongOnly',
            upper = upper,
        )
    else:
        # Update the upper bounds by taking the minimum of the current and the new upper bounds
        bs.optimization.constraints.box['upper'] = np.minimum(
            bs.optimization.constraints.box['upper'],
            upper
        )

    return None


optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn = bibfn_return_series,
        width = 252 * 3,
        fill_value = 0,
    ),
    # 'bm_series': OptimizationItemBuilder(
    #     bibfn = bibfn_bm_series,
    #     width = 252 * 3,
    #     align = True,
    # ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn = bibfn_budget_constraint,
        budget = 1
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn = bibfn_box_constraints,
        upper = 0.1
    ),
    # 'box_constraints': OptimizationItemBuilder(
    #     bibfn = bibfn_bm_relative_upper_bounds,
    #     multiple = 20,
    # ),
}





# Initialize the backtest service
bs = BacktestService(
    data = data,
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates,
)







# --------------------------------------------------------------------------
# Run backtest for mean-variance portfolio
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






# Update the backtest service with a MeanVariance optimization object
bs.optimization = MeanVariance(
    covariance = Covariance(method = 'pearson'),
    expected_return = ExpectedReturn(method = 'geometric'),
    risk_aversion = 10,
    solver_name = 'cvxopt',
)

# Instantiate the backtest object and run the backtest
bt_mv_ra5 = Backtest()

# Run the backtest
bt_mv_ra5.run(bs = bs)

# # Save the backtest as a .pickle file
# bt_mv_ra5.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_mv_ra5.pickle' # <change this to your desired filename>
# )











# Inspect the optimization results - i.e. the weights stored in the strategy object
bt_mv.strategy.get_weights_df()
bt_mv.strategy.get_weights_df().plot(kind='bar', stacked=True, figsize=(10, 6))


bs.selection.df()
bs.selection.df_binary().sum()
bs.selection.selected


len(bs.optimization.constraints.box['upper'])
len(bs.optimization.constraints.ids)




# --------------------------------------------------------------------------
# Run backtest for index tracking portfolio (via least squares optimization)
# --------------------------------------------------------------------------

bs.optimization = LeastSquares(
    solver_name = 'cvxopt',
)

# Instantiate the backtest object and run the backtest
bt_ls = Backtest()

# Run the backtest
bt_ls.run(bs = bs)

# # Save the backtest as a .pickle file
# bt_ls.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_ls.pickle' # <change this to your desired filename>
# )




# --------------------------------------------------------------------------
# Prepare the optimization for a specific date and inspect the generated data items
# i.e., the selection object, the optimization data and the optimization constraints
# --------------------------------------------------------------------------

rebalancing_date = rebdates[0]
bs.build_selection(rebdate = rebalancing_date)
bs.build_optimization(rebdate = rebalancing_date)


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
# Simulate strategies
# --------------------------------------------------------------------------

# Laod backtests from pickle
bt_mv = load_pickle(
    filename = 'backtest_mv.pickle',
    path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',
)
bt_mv_ra5 = load_pickle(
    filename = 'backtest_mv_ra5.pickle',
    path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',
)
bt_ls = load_pickle(
    filename = 'backtest_ls.pickle',
    path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',
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
sim_ls = bt_ls.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs
)

# Concatenate the simulations
sim = pd.concat({
    'bm': bs.data.bm_series,
    'mv': sim_mv,
    'mv_ra5': sim_mv_ra5,
    'ls': sim_ls,
}, axis = 1).dropna()
sim.columns = ['Benchmark', 'Mean-Variance', 'Mean-Variance RA5', 'Index Tracker']


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



