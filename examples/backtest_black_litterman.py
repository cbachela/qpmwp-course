############################################################################
### QPMwP CODING EXAMPLES - BLACK LITTERMAN MODEL
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     17.04.2025
# First version:    17.04.2025
# --------------------------------------------------------------------------



# This script demonstrates the application of the Black-Litterman model using
# the single stock datasets.





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
from estimation.black_litterman import (
    bl_posterior_mean,                              # NEW!
    generate_views_from_scores,                     # NEW!
)
from optimization.optimization import (
    Objective,
    MeanVariance,
    BlackLitterman,                                 # NEW!
)
from backtesting.backtest_item_builder_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    # Selection item builder functions
    bibfn_selection_min_volume,
    bibfn_selection_gaps,
    bibfn_selection_jkp_factor_scores,              # NEW!
    # Optimization item builder functions
    bibfn_return_series,
    bibfn_bm_series,
    bibfn_cap_weights,                              # NEW!
    bibfn_scores,                                   # NEW!
    # Constraints item builder functions
    bibfn_budget_constraint,
    bibfn_box_constraints,
)
from backtesting.backtest_data import BacktestData
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest






# JKP_FIELDS_VALUE = [
#     'at_me',
#     'be_me',
#     'bev_mev',
#     'chcsho12m',
#     'debt_me',
#     'div12m_me',
#     'ebitda_mev',
#     'eq_dur',
#     'eqnetis_at',
#     'eqnpo_12m',
#     'eqnpo_me',
#     'fcf_me',
#     'ival_me',
#     'netis_at',
#     'ni_me',
#     'ocf_me',
#     'sale_me',
# ]

# JKP_FIELDS_QMJ = [
#     'qmj_prof',
#     'qmj_growth',
#     'qmj_safety',
#     'qmj',  
# ]

JKP_FIELDS_MOMENTUM = [
    'prc_highprc_252d',
    'resff3_6_1',
    'resff3_12_1',
    'ret_3_1',
    'ret_6_1',
    'ret_9_1',
    'ret_12_1',
    'seas_1_1na',
]






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




# --------------------------------------------------------------------------
# Prepare backtest service
# --------------------------------------------------------------------------


# Define rebalancing dates
n_days = 21*3  # Rebalance every n_days
market_data_dates = market_data.index.get_level_values('date').unique().sort_values(ascending=True)
rebdates = market_data_dates[market_data_dates > '2005-01-01'][::n_days].strftime('%Y-%m-%d').tolist()
# rebdates = market_data_dates[market_data_dates > '2018-01-01'][::n_days].strftime('%Y-%m-%d').tolist()
rebdates


# Define the selection item builders
selection_item_builders = {
    'gaps': SelectionItemBuilder(
        bibfn = bibfn_selection_gaps,
        width = 252*3,
        n_days = 10,
    ),
    'min_volume': SelectionItemBuilder(
        bibfn = bibfn_selection_min_volume,
        width = 252,
        min_volume = 500_000,
        agg_fn = np.median,
    ),
    'scores': SelectionItemBuilder(             # NEW!
        bibfn = bibfn_selection_jkp_factor_scores,
        # fields = JKP_FIELDS_QMJ,
        fields = JKP_FIELDS_MOMENTUM,
    ),
}


# Define the optimization item builders
optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn = bibfn_return_series,
        width = 252*3,
        fill_value = 0,
    ),
    'bm_series': OptimizationItemBuilder(
        bibfn = bibfn_bm_series,
        width = 252*3,
        align = True,
        name = 'bm_series',
    ),
    'cap_weights': OptimizationItemBuilder(                # NEW!
        bibfn = bibfn_cap_weights,
    ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn = bibfn_budget_constraint,
        budget = 1,
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn = bibfn_box_constraints,
        upper = 0.2,
    ),
    'scores': OptimizationItemBuilder(                     # NEW!
        bibfn = bibfn_scores,
    ),
}


# Initialize the Black-Litterman
# optimization object
optimization = BlackLitterman(                             # NEW!
    solver_name='cvxopt',
    covariance=Covariance(method='pearson'),
    risk_aversion=1,
    tau_psi=0.01,
    tau_omega=0.0001,
    # view_method='quintile',
    # scalefactor=252,
    view_method='absolute',
    scalefactor=1,
    fields=JKP_FIELDS_MOMENTUM,
)


# Initialize the backtest service
bs = BacktestService(
    data=data,
    optimization=optimization,
    selection_item_builders=selection_item_builders,
    optimization_item_builders=optimization_item_builders,
    rebdates=rebdates,
)





# --------------------------------------------------------------------------
# Run backtests
# --------------------------------------------------------------------------


# Update the field argument for the B-L optimization
bs.optimization.params['fields'] = JKP_FIELDS_MOMENTUM
# bs.optimization.params['fields'] = ['ret_12_1']

# Run the backtest
bt_bl_mom = Backtest()
bt_bl_mom.run(bs=bs)

# # Save the backtest as a .pickle file
# bt_bl_mom.save(
#     path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/',  # <change this to your path where you want to store the backtest>
#     filename = 'backtest_bl_mom.pickle' # <change this to your desired filename>
# )







# --------------------------------------------------------------------------
# Go through the optimization step-by-step for a particular date
# --------------------------------------------------------------------------


# # Increase the minimum volume filter so that we have less stocks in the selection
# bs.selection_item_builders['min_volume'].arguments['min_volume'] = 100_000_000 # instead of 500_000
# bs.selection_item_builders['min_volume'].arguments['min_volume'] = 500_000




# Define a date for the optimization
date = rebdates[-1]
date

# Prepare optimization for the given date
bs.prepare_rebalancing(date)
# bs.optimization.set_objective(bs.optimization_data)




# Inspect the selction for the given date
bs.selection.df()

# Inspect the optimization data for the given date
bs.optimization_data.keys()
bs.optimization_data['scores']




# Extract scores
scores = bs.optimization_data['scores']
scores

# Extract the cap-weights
cap_weights = bs.optimization_data['cap_weights']
cap_weights

# Compute the covariance matrix
covariance = Covariance(method='pearson')
return_series = bs.optimization_data['return_series']
covariance.estimate(return_series, inplace=True)
covariance.matrix

# Calculate implied expected return from
# the cap-weights, the covariance matrix and the risk aversion
risk_aversion = 1
mu_implied = risk_aversion * covariance.matrix @ cap_weights

# Construct the views
P_tmp = {}
q_tmp = {}
for col in scores.columns:
    P_tmp[col], q_tmp[col] = generate_views_from_scores(
        scores=scores[col],
        mu_implied=mu_implied,
        # method='quintile',
        method='absolute',
        scalefactor=1,
    )
    
P = pd.concat(P_tmp, axis=0)
q = pd.concat(q_tmp, axis=0)

P
q
scores




# -------------------------
# Tuning parameters
tau_psi = 0.01
tau_omega = 0.0001
# -------------------------


# Uncertainty of the views
Omega = pd.DataFrame(
    np.diag([tau_omega] * len(q)),
    index=q.index,
    columns=q.index
)
Omega


# Compute the posterior expected return vector
mu_posterior = bl_posterior_mean(
    mu_prior=mu_implied,
    P=P,
    q=q,
    Psi=covariance.matrix * tau_psi,
    Omega=Omega,
)


# Compare the prior and the posterior expected return vectors
Mu = pd.concat({
    "mu_prior": mu_implied,
    "mu_posterior": mu_posterior,
}, axis=1) * 252
# Mu.sort_values("mu_prior", ascending=False, inplace=True)
Mu
Mu.plot(kind="bar", figsize=(10, 5))



# Compute the analytical mean-variance optimization with the posterior returns
w_posterior_analytical = pd.Series(np.linalg.inv(covariance.matrix) @ mu_posterior, index=mu_posterior.index)




# Re-run the mean-variance optimization with the posterior returns
mv = MeanVariance(
    constraints=bs.optimization.constraints,
    solver_name="cvxopt",
)
mv.objective = Objective(
    q=mu_posterior * (-1),
    P=covariance.matrix * risk_aversion,
)
mv.solve()
w_posterior_numerical = pd.Series(mv.results["weights"])


W = pd.DataFrame([cap_weights, w_posterior_analytical, w_posterior_numerical],
                 index=["w_prior", "w_posterior_analytical", "w_posterior_numerical"]).T
W.plot(kind="bar", figsize=(10, 5))



W = pd.DataFrame([cap_weights, w_posterior_numerical],
                 index=["w_prior", "w_posterior_numerical"]).T
W.plot(kind="bar", figsize=(10, 5))


P
bs.optimization_data['return_series'].corr()













# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------


# Laod backtests from pickle
path = 'C:/Users/User/OneDrive/Documents/QPMwP/Backtests/' #<change this to your local path>

bt_bl_mom = load_pickle(
    filename = 'backtest_bl_mom.pickle',
    path = path,
)



fixed_costs = 0
variable_costs = 0.002
return_series = bs.data.get_return_series()

strategy_dict = {
    'bl_mom': bt_bl_mom.strategy,
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

