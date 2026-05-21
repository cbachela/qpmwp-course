############################################################################
### QPMwP - BACKTEST ITEM BUILDER FUNCTIONS (bibfn) - OPTIMIZATION DATA
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     16.03.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Third party imports
import numpy as np
import pandas as pd






def bibfn_return_series(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for return series.
    Prepares an element of bs.optimization_data with
    single stock return series that are used for optimization.
    '''

    # Arguments
    width = kwargs.get('width')
    weekdays_only = kwargs.get('weekdays_only', True)
    fillna_value = kwargs.get('fillna_value', 0)

    # Data: get return series
    if hasattr(bs.data, 'get_return_series'):
        return_series = bs.data.get_return_series(
            width=width,
            end_date=rebdate,
            weekdays_only=weekdays_only,
            fillna_value=fillna_value,
        )
    else:
        return_series = bs.data.get('return_series')
        if return_series is None:
            raise ValueError('Return series data is missing.')

    # Selection
    ids = bs.selection.selected
    if len(ids) == 0:
        ids = return_series.columns

    # Subset the return series to the selected stocks
    return_series = return_series[ids]

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
    if hasattr(bs.data, name):
        data = getattr(bs.data, name)
    else:
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



def bibfn_cap_weights(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    # Selection
    ids = bs.selection.selected

    # Data - market capitalization
    mcap = bs.data.market_data['mktcap']

    # Get last available values for current rebdate
    mcap = mcap[mcap.index.get_level_values('date') <= rebdate].groupby(
        level = 'id'
    ).last()

    # Remove duplicates
    mcap = mcap[~mcap.index.duplicated(keep=False)].loc[ids]

    # Attach cap-weights to the optimization data object
    bs.optimization_data['cap_weights'] = mcap / mcap.sum()

    return None



def bibfn_scores(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Copies scores from the selection object to the optimization data object
    '''

    ids = bs.selection.selected
    scores = bs.selection.filtered['jkp_data_scores'].loc[ids]
    # Drop the 'binary' column
    bs.optimization_data['scores'] = scores.drop(columns=['binary'])
    return None
