############################################################################
### QPMwP CODING EXAMPLES - Backtest 7: ScoreVariance with ML-based signal
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     13.03.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------




# This script demonstrates how to run a backtest using the qpmwp-course library
# and single stock data which change over time.

# The script uses the 'ScoreVariance' portfolio optimization class and a ML-based signal.





# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.getcwd()))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from helper_functions import (
    load_pickle,
    load_data_spi,
    align_market_data_with_jkp_data,
)
from estimation.covariance import Covariance
from optimization.optimization import (
    ScoreVariance,
    PercentilePortfolio,
)
from backtesting.backtest_item_builder.bib_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder.bibfn_selection import (
    bibfn_selection_gaps,
    bibfn_selection_min_volume,
    bibfn_selection_jkp_data_scores,
)
from backtesting.backtest_item_builder.bibfn_optimization_data import (
    bibfn_return_series,
    bibfn_scores,
)
from backtesting.backtest_item_builder.bibfn_constraints import (
    bibfn_budget_constraint,
    bibfn_box_constraints,
)
from backtesting.backtest_data import BacktestData
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest





# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PATH_TO_DATA = 'C:/Users/User/OneDrive/Documents/QPMwP/Data/'     # <change this to your path to data>
SAVE_PATH = 'C:/Users/User/OneDrive/Documents/QPMwP/2026/Code/'   # <change this to your path where you want to store the backtest>
WIDTH_3Y = 365 * 3




# --------------------------------------------------------------------------
# Load data
# - market data (from parquet file)
# - jkp data (from parquet file)
# - swiss performance index, SPI (from csv file)
# - ML signal (from parquet file)
# --------------------------------------------------------------------------

# Load market and jkp data from parquet files
market_data = pd.read_parquet(path = f'{PATH_TO_DATA}market_data.parquet')
jkp_data = pd.read_parquet(path = f'{PATH_TO_DATA}jkp_data.parquet')
spi = load_data_spi(path=f'{project_root}/data/')

# Load ML signal and add to jkp_data (so that we can use it in the backtest)
ml_signal = pd.read_parquet(path=f'{PATH_TO_DATA}ltr_signal.parquet')
ml_signal = ml_signal.stack().squeeze().astype("float64")
ml_signal.name = 'ml_signal'
ml_signal.index.names = jkp_data.index.names

# Add the ml_signal to jkp_data
jkp_data = jkp_data.join(ml_signal, how='left')

# Align market data with jkp data
market_data_ffill, jkp_data = align_market_data_with_jkp_data(
    market_data=market_data,
    jkp_data=jkp_data,
)



# --------------------------------------------------------------------------
# Instantiate the BacktestData class
# and set the market, jkp, and benchmark data as attributes
# --------------------------------------------------------------------------

data = BacktestData()
data.market_data = market_data_ffill  # notice that we use the forward filled market data here
data.jkp_data = jkp_data
data.bm_series = spi



# --------------------------------------------------------------------------
# Define rebalancing dates
# --------------------------------------------------------------------------

n_month = 3 # We want to rebalance every n_month months
jkp_data_dates = (
    jkp_data
    .index.get_level_values('date')
    .unique().sort_values()
)
rebdates = (
    jkp_data_dates[
        jkp_data_dates > market_data.index.get_level_values('date').min()
    ][::n_month]
    .strftime('%Y-%m-%d').tolist()
)
rebdates = [
    date for date in rebdates if date >= '2003-01-31'
    and date < rebdates[-1]
]
rebdates




# --------------------------------------------------------------------------
# Define the selection item builders.
# --------------------------------------------------------------------------

selection_item_builders = {
    'gaps': SelectionItemBuilder(
        bibfn=bibfn_selection_gaps,
        width=WIDTH_3Y,
        n_days=10,
    ),
    'min_volume': SelectionItemBuilder(
        bibfn=bibfn_selection_min_volume,
        width=365,
        min_volume=500_000,
        agg_fn=np.median,
    ),
    'jkp_data_scores': SelectionItemBuilder(
        bibfn=bibfn_selection_jkp_data_scores,
        fields=['ml_signal'],
    ),
}




# --------------------------------------------------------------------------
# Define the optimization item builders.
# --------------------------------------------------------------------------

optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn=bibfn_return_series,
        width=WIDTH_3Y,
        fill_value=0,
    ),
    'scores': OptimizationItemBuilder(
        bibfn=bibfn_scores,
        fields=['ml_signal'],
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
# Run backtest for score-variance portfolio
# --------------------------------------------------------------------------

# Update the backtest service with a ScoreVariance optimization object

field_name = 'ml_signal'  # <change this to your desired field name from jkp_data>

bs.optimization = ScoreVariance(
    field=field_name,
    covariance=Covariance(method='pearson'),
    risk_aversion=10000,
    solver_name='cvxopt',
)

# Instantiate the backtest object and run the backtest
bt_sv = Backtest()

# Run the backtest
bt_sv.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_sv.save(
#     path=SAVE_PATH,
#     filename=f'demo_backtest_7_sv_{field_name}.pickle' # <change this to your desired filename>
# )



# --------------------------------------------------------------------------
# Run backtest for a top quintile portfolio (tqp)
# --------------------------------------------------------------------------

# Update the backtest service with a PercentilePortfolio optimization object
bs.optimization = PercentilePortfolio(
    field='ml_signal',  # <change this to your desired field name from jkp_data>
    percentile=80,
    sign='>=',
)

# Instantiate the backtest object and run the backtest
bt_tqp = Backtest()

# Run the backtest
bt_tqp.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_tqp.save(
#     path=SAVE_PATH,
#     filename=f'demo_backtest_7_tqp_{field_name}.pickle' # <change this to your desired filename>
# )



# --------------------------------------------------------------------------
# Run backtest for a bottom quintile portfolio (bqp)
# --------------------------------------------------------------------------

# Update the backtest service with a PercentilePortfolio optimization object
bs.optimization = PercentilePortfolio(
    field='ml_signal',  # <change this to your desired field name from jkp_data>
    percentile=20,
    sign='<=',
)

# Instantiate the backtest object and run the backtest
bt_bqp = Backtest()

# Run the backtest
bt_bqp.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_bqp.save(
#     path=SAVE_PATH,
#     filename=f'demo_backtest_7_bqp_{field_name}.pickle' # <change this to your desired filename>
# )



# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------

# Load backtests from pickle
bt_sv = load_pickle(
    filename=f'demo_backtest_7_sv_{field_name}.pickle',
    path=SAVE_PATH,
)
bt_tqp = load_pickle(
    filename=f'demo_backtest_7_tqp_{field_name}.pickle',
    path=SAVE_PATH,
)
bt_bqp = load_pickle(
    filename=f'demo_backtest_7_bqp_{field_name}.pickle',
    path=SAVE_PATH,
)

# Simulate
fixed_costs = 0
variable_costs = 0.002
return_series = bs.data.get_return_series(weekdays_only=False)

sim_sv_gross = bt_sv.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=0,
)
sim_sv_net = bt_sv.strategy.simulate(
    return_series=return_series,
    fc=fixed_costs,
    vc=variable_costs,
)
# sim_tqp = bt_tqp.strategy.simulate(
#     return_series=return_series,
#     fc=fixed_costs,
#     vc=variable_costs
# )
# sim_bqp = bt_bqp.strategy.simulate(
#     return_series=return_series,
#     fc=fixed_costs,
#     vc=variable_costs
# )

# Concatenate the simulations
sim = pd.concat({
    'bm': bs.data.bm_series,
    f'sv_gross_{field_name}': sim_sv_gross,
    f'sv_net_{field_name}': sim_sv_net,
    # f'tqp_{field_name}': sim_tqp,
    # f'bqp_{field_name}': sim_bqp,
}, axis=1).dropna()
sim.columns = [
    'Benchmark',
    f'Score Variance Gross ({field_name})',
    f'Score Variance Net ({field_name})',
    # f'Top Quintile ({field_name})',
    # f'Bottom Quintile ({field_name})'
]

sim = sim[sim.index <= '2022-12-31']

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

to = bt_sv.strategy.turnover(return_series=return_series)
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



