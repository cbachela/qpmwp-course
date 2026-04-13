############################################################################
### QPMwP - HELPER FUNCTIONS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     13.04.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------


# Standard library imports
import os
import pickle
from typing import Optional, Union, Any

# Third party imports
import numpy as np
import pandas as pd





def load_data_msci(path: Optional[str] = None, n: int = 24) -> dict[str, pd.DataFrame]:

    '''
    Loads daily total return series from 1999-01-01 to 2023-04-18
    for MSCI country indices and for the MSCI World index.
    '''

    path = os.path.join(os.getcwd(), f'data{os.sep}') if path is None else path

    # Load msci country index return series
    df = pd.read_csv(os.path.join(path, 'msci_country_indices.csv'),
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        date_format='%d-%m-%Y')
    series_id = df.columns[0:n]
    X = df[series_id]

    # Load msci world index return series
    y = pd.read_csv(f'{path}NDDLWI.csv',
                    index_col=0,
                    header=0,
                    parse_dates=True,
                    date_format='%d-%m-%Y')

    return {'return_series': X, 'bm_series': y}



def load_data_spi(path: Optional[str] = None) -> pd.Series:

    '''
    Loads daily total return series of the swiss performance index
    '''

    path = os.path.join(os.getcwd(), f'data{os.sep}') if path is None else path

    # Load swiss performance index return series
    df = pd.read_csv(os.path.join(path, 'spi_index.csv'),
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        date_format='%d/%m/%Y')
    df.index = pd.DatetimeIndex(df.index)
    return df.squeeze()



def load_jkp_factor_series(path: Optional[str] = None) -> dict[str, pd.DataFrame]:

    '''
    Loads daily total return series from 1999-01-01 to 2023-04-18
    for MSCI country indices and for the MSCI World index.
    '''

    path = os.path.join(os.getcwd(), f'data{os.sep}') if path is None else path

    # Load msci country index return series
    
    df = pd.read_csv(
        os.path.join(path, 'jkp_factor_series_che_eqw.csv'),
        index_col=0,
        header=0,
        parse_dates=True,
        date_format='%d/%m/%Y'
    )

    # Convert the date column to datetime and set it as a column (not index)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    # Create a multi-index with date and id (id = factor name)
    df = df.set_index('date', append=True).swaplevel()
    df.index.names = ['date', 'id']

    factor_series = df['ret'].unstack(level='id').dropna()

    return factor_series



def align_market_data_with_jkp_data(
    market_data: pd.DataFrame,
    jkp_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns market data with JKP data by forward filling missing market data for dates present in JKP data.

    Parameters:
    - market_data: DataFrame containing market data with a Multiindex (date: DateTime, id: str).
    - jkp_data: DataFrame containing JKP data with a Multiindex (date: DateTime, id: str).
    
    Returns:
    - A tuple of DataFrames: (market_data_ffill, jkp_data) with market data aligned to 
      the dates in JKP data, with missing values forward filled.
    """

    # Drop rows with missing 'id' values
    market_data_clean = market_data[~market_data.index.get_level_values('id').isna()]
    jkp_data_clean = jkp_data[~jkp_data.index.get_level_values('id').isna()]

    market_data_dates = (
        market_data_clean
        .index.get_level_values('date')
        .unique().sort_values()
    )
    jkp_data_dates = (
        jkp_data_clean
        .index.get_level_values('date')
        .unique().sort_values()
    )
    missing_dates = jkp_data_dates[~jkp_data_dates.isin(market_data_dates)]
    tmp_dict = {}
    for date in missing_dates:
        last_date = market_data_dates[market_data_dates <= date][-1]
        tmp_dict[date] = market_data_clean.loc[last_date]

    df_missing = pd.concat(tmp_dict, axis=0)
    df_missing.index.names = market_data_clean.index.names
    market_data_ffill = pd.concat([market_data_clean, df_missing]).sort_index()

    return market_data_ffill, jkp_data_clean



def load_pickle(filename: str,
                path: Optional[str] = None) -> Union[Any, None]:
    if path is not None:
        filename = os.path.join(path, filename)
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except EOFError:
        print("Error: Ran out of input. The file may be empty or corrupted.")
        return None
    except Exception as ex:
        print("Error during unpickling object:", ex)
    return None



def to_numpy(data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]]) -> Optional[np.ndarray]:
    return None if data is None else (
        data.to_numpy() if hasattr(data, 'to_numpy') else data
    )



def simulate_correlated_gbm(mu, sigma, T=252, S0=None, random_seed=None) -> pd.DataFrame:
    """
    Simulate one path of correlated geometric Brownian motions.

    Parameters
    ----------
    mu : ndarray (d,)
        Drift vector.
    sigma : ndarray (d, d)
        Covariance matrix of Brownian motions.
    T : int
        Number of time steps.
    S0 : ndarray (d,), optional
        Initial values (default = ones).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape (T+1, d) with simulated paths.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    d = len(mu)

    dt = 1/T

    if S0 is None:
        S0 = np.ones(d)
    else:
        S0 = np.asarray(S0)

    # Cholesky decomposition
    L = np.linalg.cholesky(sigma)

    # Generate correlated Brownian increments
    Z = np.random.normal(size=(T, d))
    dW = np.sqrt(dt) * Z @ L.T

    # Allocate path array
    S = np.zeros((T + 1, d))
    S[0] = S0

    drift = (mu - 0.5 * np.diag(sigma)) * dt

    # Simulate
    for t in range(T):
        S[t + 1] = S[t] * np.exp(drift + dW[t])

    return pd.DataFrame(
        S,
        columns=[f"Asset_{i+1}" for i in range(d)]
    )
    