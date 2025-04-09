############################################################################
### QPMwP - BACKTEST ITEM BUILDER FUNCTIONS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     20.03.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Third party imports
import numpy as np
import pandas as pd






# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Selection
# --------------------------------------------------------------------------

def bibfn_selection_min_volume(bs, rebdate: str, **kwargs) -> pd.DataFrame:

    '''
    Backtest item builder function for defining the selection
    Filter stocks based on minimum volume (i.e., liquidity).
    '''

    # Arguments
    width = kwargs.get('width', 365)
    agg_fn = kwargs.get('agg_fn', np.median)
    min_volume = kwargs.get('min_volume', 500_000)

    # Volume data
    vol = (
        bs.data.get_volume_series(
            end_date=rebdate,
            width=width
        ).fillna(0)
    )
    vol_agg = vol.apply(agg_fn, axis=0)

    # Filtering
    vol_binary = pd.Series(1, index=vol.columns, dtype=int, name='binary')
    vol_binary.loc[vol_agg < min_volume] = 0


    # Output
    filter_values = pd.DataFrame({
        'values': vol_agg,
        'binary': vol_binary,
    }, index=vol_agg.index)

    return filter_values



def bibfn_selection_NA(bs, rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection.
    Filters out stocks which have more than 'na_threshold' NA values in the
    return series. Remaining NA values are filled with zeros.
    '''

    # Arguments
    width = kwargs.get('width', 252)
    na_threshold = kwargs.get('na_threshold', 10)

    # Data: get return series
    return_series = bs.data.get_return_series(
        width=width,
        end_date=rebdate,
        fillna_value=None,
    )

    # Identify colums of return_series with more than 10 NA value
    # and remove them from the selection
    na_counts = return_series.isna().sum()
    na_columns = na_counts[na_counts > na_threshold].index

    # Output
    filter_values = pd.Series(1, index=na_counts.index, dtype=int, name='binary')
    filter_values.loc[na_columns] = 0

    return filter_values.astype(int)



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
    seed = kwargs.get('seed')
    if seed is None:
        seed = np.random.randint(0, 1_000_000)    
    # Add the position of rebdate in bs.settings['rebdates'] to
    # the seed to make it change with the rebdate
    seed += bs.settings['rebdates'].index(rebdate)
    return_series = bs.data.get('return_series')

    if return_series is None:
        raise ValueError('Return series data is missing.')

    # Random selection
    # Set the random seed for reproducibility
    np.random.seed(seed)
    selected = np.random.choice(return_series.columns, k, replace = False)

    return pd.Series(np.ones(len(selected), dtype = int), index = selected, name = 'binary')




# def bibfn_selection_ltr(bs, rebdate: str, **kwargs) -> pd.DataFrame:

#     '''
#     Backtest item builder function for defining the selection
#     based on a Learn-to-Rank model.
#     '''

#     # Arguments
#     params_xgb = kwargs.get('params_xgb')

#     # Selection
#     ids = bs.selection.selected

#     # Extract data
#     merged_df = bs.data.get('merged_df').copy()
#     df_train = merged_df[merged_df['DATE'] < rebdate]#.reset_index(drop = True)
#     df_test = merged_df[merged_df['DATE'] == rebdate]#.reset_index(drop = True)
#     df_test = df_test[ df_test['ID'].isin(selected) ]
#     ids = df_test['ID'].to_list()

#     # Training data
#     X_train = df_train.drop(['DATE', 'ID', 'label', 'ret'], axis=1)
#     y_train = df_train['label']
#     grouped_train = df_train.groupby('DATE').size().to_numpy()
#     dtrain = xgb.DMatrix(X_train, label = y_train)
#     dtrain.set_group(grouped_train)

#     # Evaluation data
#     X_test = df_test.drop(['DATE', 'ID', 'label', 'ret'], axis=1)
#     grouped_test = df_test.groupby('DATE').size().to_numpy()
#     dtest = xgb.DMatrix(X_test)
#     dtest.set_group(grouped_test)

#     # Train and predict
#     bst = xgb.train(params_xgb, dtrain, 100)
#     scores = bst.predict(dtest) * (-1)

#     # # Extract feature importance
#     # f_importance = bst.get_score(importance_type='gain')

#     return pd.DataFrame({'values': scores,
#                          'binary': np.ones(len(scores), dtype = int),
#                         }, index = scores.index)






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
    if hasattr(bs.data, 'get_return_series'):
        return_series = bs.data.get_return_series(
            width=width,
            end_date=rebdate,
            fillna_value=None,
        )
    else:
        return_series = bs.data.get('return_series')
        if return_series is None:
            raise ValueError('Return series data is missing.')

    # Selection
    ids = bs.selection.selected
    if len(ids) == 0:
        ids = return_series.columns

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
