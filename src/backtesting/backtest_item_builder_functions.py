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
import xgboost as xgb





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



def bibfn_selection_ltr(bs: 'BacktestService', rebdate: str, **kwargs) -> None:
    '''
    This function constructs labels and features for a specific rebalancing date.
    It acts as a filtering since stocks which could not be labeled or which
    do not have features are excluded from the selection.
    '''

    # Define the selection by the ids available for the current rebalancing date
    df_test = bs.data.merged_df[bs.data.merged_df['date'] == rebdate]
    ids = list(df_test['id'].unique())

    # Return a binary series indicating the selected stocks
    return pd.Series(1, index=ids, name='binary', dtype=int)



def bibfn_selection_jkp_factor_scores(bs, rebdate: str, **kwargs) -> pd.DataFrame:

    '''
    Backtest item builder function for defining the selection.
    Filter stocks based on available scores in the jkp factor data.
    '''

    # Arguments
    fields = kwargs.get('fields')

    # Selection
    ids = bs.selection.selected
    if ids is None:
        ids = bs.data.jkp_data.index.get_level_values('id').unique()

    # Filter rows prior to the rebdate and within one year
    df = bs.data.jkp_data[fields]
    filtered_df = df.loc[
        (df.index.get_level_values('date') < rebdate) &
        (df.index.get_level_values('date') >= pd.to_datetime(rebdate) - pd.Timedelta(days=365))
    ]

    # Extract the last available value for each id
    scores = filtered_df.groupby('id').last()

    # Output
    filter_values = scores.copy()
    filter_values['binary'] = scores.notna().all(axis=1).astype(int)

    return filter_values





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
    scores = bs.selection.filtered['scores'].loc[ids]
    # Drop the 'binary' column
    bs.optimization_data['scores'] = scores.drop(columns=['binary'])
    return None


def bibfn_scores_ltr(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Constructs scores based on a Learning-to-Rank model.        
    '''

    # Arguments
    params_xgb = kwargs.get('params_xgb')
    if params_xgb is None or not isinstance(params_xgb, dict):
        raise ValueError('params_xgb is not defined or not a dictionary.')
    training_dates = kwargs.get('training_dates')

    # Extract data
    df_train = bs.data.merged_df[bs.data.merged_df['date'] < rebdate]
    df_test = bs.data.merged_df[bs.data.merged_df['date'] == rebdate]
    df_test = df_test.loc[df_test['id'].drop_duplicates(keep='first').index]
    df_test = df_test.loc[df_test['id'].isin(bs.selection.selected)]

    # Training data
    X_train = (
        df_train.drop(['date', 'id', 'label', 'ret'], axis=1)
        # df_train.drop(['date', 'id', 'label'], axis=1)  # Include ret in the features as a proof of concept
    )
    y_train = df_train['label'].loc[X_train.index]
    grouped_train = df_train.groupby('date').size().to_numpy()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(grouped_train)

    # Test data
    y_test = pd.Series(df_test['label'].values, index=df_test['id'])
    X_test = df_test.drop(['date', 'id', 'label', 'ret'], axis=1)
    # X_test = df_test.drop(['date', 'id', 'label'], axis=1)  # Include ret in the features as a proof of concept
    grouped_test = df_test.groupby('date').size().to_numpy()
    dtest = xgb.DMatrix(X_test)
    dtest.set_group(grouped_test)

    # Train the model using the training data
    if rebdate in training_dates:
        model = xgb.train(params_xgb, dtrain, 100)
        bs.model_ltr = model
    else:
        # Use the previous model for the current rebalancing date
        model = bs.model_ltr

    # Predict using the test data
    pred = model.predict(dtest)
    preds =  pd.Series(pred, df_test['id'], dtype='float64')
    ranks = preds.rank(method='first', ascending=True).astype(int)

    # Output
    scores = pd.concat({
        'scores': preds,
        'ranks': (100 * ranks / len(ranks)).astype(int),  # Normalize the ranks to be between 0 and 100
        'true': y_test,
        'ret': pd.Series(df_test['ret'].values, index=df_test['id']),
    }, axis=1)
    bs.optimization_data['scores'] = scores
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
