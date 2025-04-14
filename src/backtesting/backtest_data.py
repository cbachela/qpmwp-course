############################################################################
### QPMwP - CLASS BacktestData
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from typing import Optional
import warnings

# Third party imports
import pandas as pd





class BacktestData():

    def __init__(self):
        pass

    def get_return_series(
        self,
        ids: Optional[pd.Series] = None,
        end_date: Optional[str] = None,
        width: Optional[int] = None,
        fillna_value: Optional[float] = None,
    ) -> pd.DataFrame:

        X = self.market_data.pivot_table(
            index='date',
            columns='id',
            values='price'
        )
        if ids is None:
            ids = X.columns
        if end_date is None:
            end_date = X.index.max().strftime('%Y-%m-%d')
        if width is None:
            width = X.shape[0] - 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = X[X.index <= end_date][ids].tail(width+1).pct_change(fill_method=None).iloc[1:]
            if fillna_value is not None:
                X.fillna(fillna_value, inplace=True)
            return X

    def get_volume_series(
        self,
        ids: Optional[pd.Series] = None,
        end_date: Optional[str] = None,
        width: Optional[int] = None,
    ) -> pd.DataFrame:

        X = self.market_data.pivot_table(
            index = 'date',
            columns = 'id',
            values = 'liquidity',
        )
        if ids is None:
            ids = X.columns
        if end_date is None:
            end_date = X.index.max().strftime('%Y-%m-%d')
        if width is None:
            width = X.shape[0]
        return X[X.index <= end_date][ids].tail(width)

    def get_characteristic_series(
        self,
        field: str,
        ids: Optional[pd.Series] = None,
        end_date: Optional[str] = None,
        width: Optional[int] = None,
    ) -> pd.DataFrame:

        X = self.jkp_data.pivot_table(
            index = 'date',
            columns = 'id',
            values = field,
        )
        if ids is None:
            ids = X.columns
        if end_date is None:
            end_date = X.index.max().strftime('%Y-%m-%d')
        if width is None:
            width = X.shape[0]
        return X[X.index <= end_date][ids].tail(width)

