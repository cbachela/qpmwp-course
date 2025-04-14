############################################################################
### QPMwP - CLASS ExpectedReturn
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard/Adrian Schmidli
# This version:     14.04.2025
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
# [ ] mean_harmonic
# [ ] mean_ewma (exponential weighted)






class ExpectedReturnSpecification(dict):

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
        if inplace: # If "inplace" is set to True (the default), then "self.vector" is automatically updated with "mu". If "inplace" is set to False, the method returns "mu" but does not update "self.vector" — you have to assign it manually.
            self.vector = mu
            return None
        else:
            return mu





# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

# You don’t always need a scalefactor in mean computation — but it can be useful when you want to scale or adjust the result of a mean for a specific purpose.

def mean_geometric(X: Union[pd.DataFrame, np.ndarray],
                   scalefactor: Union[float, int] = 1) -> Union[pd.Series, np.ndarray]:
    """
    Calculates the geometric mean of a list of numbers.

    Data input: The parameter X can be either a pandas.DataFrame or a numpy.ndarray.
    Def scalefactor: The argument scalefactor can be either a float or an int. It has a default value of one.

    axis=0 → column-wise operations
    axis=1 → row-wise operations

    prod(1+X)^(1/n) - 1 = mean_geometric(X) → n = number of observations in the DataFrame or ndarray.

    Data output: The function returns a pandas series or a numpy array, depending on the input type of X.
    """
    mu = np.exp(np.log(1 + X).mean(axis=0) * scalefactor) - 1
    return mu

def mean_arithmetic(X: Union[pd.DataFrame, np.ndarray],
                    scalefactor: Union[float, int] = 1) -> Union[pd.Series, np.ndarray]:
    """
    Calculates the arithmetic mean of a list of numbers.

    Data input: The parameter X can be either a pandas.DataFrame or a numpy.ndarray.
    Def scalefactor: The argument scalefactor can be either a float or an int. It has a default value of one.

    axis=0 → column-wise operations
    axis=1 → row-wise operations

    sum(X) / n = mean_arithmetic(X) → n = number of observations in the DataFrame or ndarray.
    
    Data output: The function returns a pandas series or a numpy array, depending on the input type of X.
    """
    mu = X.mean(axis=0) * scalefactor
    return mu

def mean_harmonic(X: Union[pd.DataFrame, np.ndarray],
                    scalefactor: Union[float, int] = 1) -> Union[pd.Series, np.ndarray]:
    """
    Calculates the harmonic mean of a list of numbers.

    Data input: The parameter X can be either a pandas.DataFrame or a numpy.ndarray.
    Def scalefactor: The argument scalefactor can be either a float or an int. It has a default value of one.
    
    axis=0 → column-wise operations
    axis=1 → row-wise operations

    n / sum(1/X) = mean_harmonic(X) → n = number of observations in the DataFrame or ndarray.
    
    Data output: The function returns a pandas series or a numpy array, depending on the input type of X.
    """
    mu = (len(X) / np.sum(1 / X, axis=0)) * scalefactor
    return mu

###################################################################################
# Continue from here!!

def mean_arithmetic(X: Union[pd.DataFrame, np.ndarray],
                    scalefactor: Union[float, int] = 1) -> Union[pd.Series, np.ndarray]:
    """
    Data input: The parameter X can be either a pandas.DataFrame or a numpy.ndarray.
    Def scalefactor: The argument scalefactor can be either a float or an int. It has a default value of one.
    Data output: The function returns a pandas series or a numpy array, depending on the input type of X.


    """

    mu = X.mean(axis=0) * scalefactor
    return mu