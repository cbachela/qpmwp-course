############################################################################
### QPMwP - BACKTEST ITEM BUILDER FUNCTIONS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from abc import ABC, abstractmethod
from typing import Any

# Third party imports
import numpy as np
import pandas as pd






# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Selection
# --------------------------------------------------------------------------

def bibfn_selection_min_volume(bs, rebdate: str, **kwargs) -> pd.Series:

    # Arguments
    width = kwargs.get('width', 365)
    agg_fn = kwargs.get('agg_fn', np.median)
    min_volume = kwargs.get('min_volume', 500_000)

    # Volume data
    X_vol = (
        bs.data.get_volume_series(end_date = rebdate, width = width)
        .fillna(0).apply(agg_fn, axis = 0)
    )

    # Filtering
    ids = [col for col in X_vol.columns if agg_fn(X_vol[col]) >= min_volume]

    # Output
    series = pd.Series(np.ones(len(ids)), index = ids, name = 'minimum_volume')
    bs.rebalancing.selection.add_filtered(filter_name = series.name,
                                            value = series)
    return None



def bibfn_selection_data(bs: 'BacktestService', rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection
    based on all available return series.
    '''

    return_series = bs.data.get('return_series')
    if return_series is None:
        raise ValueError('Return series data is missing.')

    return pd.Series(np.ones(return_series.shape[1], dtype = int),
                     index = return_series.columns, name = 'binary')



def bibfn_selection_data_random(bs: 'BacktestService', rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection
    based on a random k-out-of-n sampling of all available return series.
    '''
    # Arguments
    k = kwargs.get('k', 10)
    return_series = bs.data.get('return_series')

    if return_series is None:
        raise ValueError('Return series data is missing.')

    # Random selection
    selected = np.random.choice(return_series.columns, k, replace = False)

    return pd.Series(np.ones(len(selected), dtype = int), index = selected, name = 'binary')






# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Optimization data
# --------------------------------------------------------------------------

def bibfn_return_series(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for return series.
    Prepares an element of bs.optimization_data with
    single stock return series that are used for optimization.
    '''

    # Arguments
    width = kwargs.get('width')

    # Data: get return series
    return_series = bs.data.get('return_series')
    if return_series is None:
        raise ValueError('Return series data is missing.')

    # Selection
    ids = bs.selection.selected
    if len(ids) == 0:
        ids = bs.data['return_series'].columns

    # Subset the return series
    return_series = return_series[return_series.index <= rebdate].tail(width)[ids]

    # Remove weekends
    return_series = return_series[return_series.index.dayofweek < 5]

    # Output
    bs.optimization_data['return_series'] = return_series
    return None


def bibfn_bm_series(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for benchmark series.
    Prepares an element of bs.optimization_data with 
    the benchmark series that is be used for optimization.
    '''

    # Arguments
    width = kwargs.get('width')
    align = kwargs.get('align', True)
    name = kwargs.get('name', 'bm_series')

    # Data
    data = bs.data.get(name)
    if data is None:
        raise ValueError('Benchmark return series data is missing.')

    # Subset the benchmark series
    bm_series = data[data.index <= rebdate].tail(width)

    # Remove weekends
    bm_series = bm_series[bm_series.index.dayofweek < 5]

    # Append the benchmark series to the optimization data
    bs.optimization_data['bm_series'] = bm_series

    # Align the benchmark series to the return series
    if align:
        bs.optimization_data.align_dates(
            variable_names = ['bm_series', 'return_series'],
            dropna = True
        )

    return None


# --------------------------------------------------------------------------
# Backtest item builder functions - Optimization constraints
# --------------------------------------------------------------------------

def bibfn_budget_constraint(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the budget constraint.
    '''

    # Arguments
    budget = kwargs.get('budget', 1)

    # Add constraint
    bs.optimization.constraints.add_budget(rhs = budget, sense = '=')
    return None


def bibfn_box_constraints(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the box constraints.
    '''

    # Arguments
    lower = kwargs.get('lower', 0)
    upper = kwargs.get('upper', 1)
    box_type = kwargs.get('box_type', 'LongOnly')

    # Constraints
    bs.optimization.constraints.add_box(box_type = box_type,
                                        lower = lower,
                                        upper = upper)
    return None
