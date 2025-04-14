############################################################################
### QPMwP - CLASS Portfolio
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from typing import Optional

# Third party imports
import numpy as np
import pandas as pd





class Portfolio:
    """
    A class to represent an asset allocation.

    Attributes:
    -----------
    rebalancing_date : Optional[str]
        The date on which the portfolio is invested.
    weights : Optional[dict[str, float]]
        A dictionary representing the weights of assets in the portfolio.
    """
    def __init__(self,
                 rebalancing_date: Optional[str] = None,
                 weights: Optional[dict[str, float]] = None):
        self.rebalancing_date = rebalancing_date
        self.weights = weights if weights is not None else {}

    @staticmethod
    def empty() -> 'Portfolio':
        return Portfolio()

    @property
    def weights(self):
        return self._weights

    def get_weights_series(self) -> pd.Series:
        return pd.Series(self._weights)

    @weights.setter
    def weights(self, new_weights: dict):
        if not isinstance(new_weights, dict):
            if hasattr(new_weights, 'to_dict'):
                new_weights = new_weights.to_dict()
            else:
                raise TypeError('weights must be a dictionary')
        self._weights = new_weights

    @property
    def rebalancing_date(self):
        return self._rebalancing_date

    @rebalancing_date.setter
    def rebalancing_date(self, new_date: str):
        if new_date and not isinstance(new_date, str):
            raise TypeError('date must be a string')
        self._rebalancing_date = new_date

    def __repr__(self):
        return f'Portfolio(rebalancing_date={self.rebalancing_date}, weights={self.weights})'

    def float_weights(self,
                      return_series: pd.DataFrame,
                      end_date: str,
                      rescale: bool = False):
        if self.weights is not None:
            return floating_weights(X=return_series,
                                    w=self.weights,
                                    start_date=self.rebalancing_date,
                                    end_date=end_date,
                                    rescale=rescale)
        else:
            return None

    def initial_weights(self,
                        selection: list[str],
                        return_series: pd.DataFrame,
                        end_date: str,
                        rescale: bool = True) -> dict[str, float]:

        w_init = dict.fromkeys(selection, 0)
        if self.rebalancing_date is not None and self.weights is not None:
            w_float = self.float_weights(
                return_series = return_series,
                end_date = end_date,
                rescale = rescale
            )
            w_floated = w_float.iloc[-1]
            for key in w_init.keys():
                if key in w_floated.keys():
                    w_init[key] = w_floated[key]
        return w_init






# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def floating_weights(X, w, start_date, end_date, rescale=True):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if start_date < X.index[0]:
        raise ValueError('start_date must be contained in dataset')
    if end_date > X.index[-1]:
        raise ValueError('end_date must be contained in dataset')

    w = pd.Series(w, index=w.keys())
    if w.isna().any():
        raise ValueError('weights (w) contain NaN which is not allowed.')
    else:
        w = w.to_frame().T
    xnames = X.columns
    wnames = w.columns

    if not all(wnames.isin(xnames)):
        raise ValueError('Not all assets in w are contained in X.')

    X_tmp = X.loc[start_date:end_date, wnames].copy().fillna(0)
    xmat = 1 + X_tmp
    xmat.iloc[0] = w.dropna(how='all').fillna(0)
    w_float = xmat.cumprod()

    if rescale:
        w_float = w_float.div(w_float.sum(axis=1), axis='index')

    return w_float
