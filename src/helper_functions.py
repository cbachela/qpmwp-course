############################################################################
### QPMwP - HELPER FUNCTIONS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------


# Standard library imports
import os
from typing import Optional, Union

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
                        index_col=0, # Specifies that the first column ("dates") is used as index and is no longer in the dataframe.
                        header=0,
                        parse_dates=True,
                        date_format='%d-%m-%Y')
    series_id = df.columns[0:n] # Because the first column ('dates') is used as the index and no longer part of the dataframe, we start selecting the remaining columns from position zero.
    X = df[series_id]

    # Load msci world index return series
    y = pd.read_csv(f'{path}NDDLWI.csv',
                    index_col=0, # Specifies that the first column ("dates") is used as index and is no longer in the dataframe.
                    header=0,
                    parse_dates=True,
                    date_format='%d-%m-%Y')

    return {'return_series': X, 'bm_series': y}



def to_numpy(data: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]]) -> Optional[np.ndarray]:
    return None if data is None else (
        data.to_numpy() if hasattr(data, 'to_numpy') else data
    )
