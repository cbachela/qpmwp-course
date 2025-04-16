############################################################################
### QPMwP - CLASS ExpectedReturn
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------


# Standard library imports
from typing import Union, Optional

# Third party imports
import numpy as np
import pandas as pd




# TODO:

# - Add a docstrings

# [ ] Add mean estimator functions:
#       [ ] mean_harmonic
#       [ ] mean_ewma (exponential weighted)






class ExpectedReturnSpecification(dict):
#if we want to extend possibilities, we need to change this specification part
    def __init__(self,
                 method='geometric',
                 scalefactor=1,
                 **kwargs):
        super().__init__(
            method=method,
            scalefactor=scalefactor,
        )
        self.update(kwargs)


class ExpectedReturn:

    def __init__(self,
                 spec: Optional[ExpectedReturnSpecification] = None,
                 **kwargs):
        self.spec = ExpectedReturnSpecification() if spec is None else spec
        self.spec.update(kwargs)
        self._vector: Union[pd.Series, np.ndarray, None] = None

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        if isinstance(value, ExpectedReturnSpecification):
            self._spec = value
        else:
            raise ValueError(
                'Input value must be of type ExpectedReturnSpecification.'
            )
        return None

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, value):
        if isinstance(value, (pd.Series, np.ndarray)):
            self._vector = value
        else:
            raise ValueError(
                'Input value must be a pandas Series or a numpy array.'
            )
        return None

    def estimate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        inplace: bool = True,
    ) -> Union[pd.Series, np.ndarray, None]:

        scalefactor = self.spec.get('scalefactor', 1)
        estimation_method = self.spec['method']

        if estimation_method == 'geometric':
            mu = mean_geometric(X=X, scalefactor=scalefactor)
        elif estimation_method == 'arithmetic':
            mu = mean_arithmetic(X=X, scalefactor=scalefactor)
        else:
            raise ValueError(
                'Estimation method not recognized.'
            )
        if inplace:
            self.vector = mu
            return None
        else:
            return mu





# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------


def mean_geometric(X: Union[pd.DataFrame, np.ndarray],
                   scalefactor: Union[float, int] = 1) -> Union[pd.Series, np.ndarray]:

    mu = np.exp(np.log(1 + X).mean(axis=0) * scalefactor) - 1
    return mu

def mean_arithmetic(X: Union[pd.DataFrame, np.ndarray],
                    scalefactor: Union[float, int] = 1) -> Union[pd.Series, np.ndarray]:

    mu = X.mean(axis=0) * scalefactor
    return mu
