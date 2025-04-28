############################################################################
### QPMwP - CLASS ExpectedReturn
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard/Adrian Schmidli
# This version:     24.04.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------


# Standard library imports
from typing import Union, Optional

# Third party imports
import numpy as np
import pandas as pd




# TODO:

# - Add a docstrings

# [X] Add mean estimator functions:
#   [x] mean_harmonic
#   [x] mean_ewm (exponential weighted)






class ExpectedReturnSpecification(dict): # This class predefines the default "method" and the "scalefactor" used for the expected return estimation. The argument "kwargs" allows you to add additional parameters to the dictionary.
    def __init__(self,
                 method='geometric',
                 scalefactor=1,
                 **kwargs):
        super().__init__(
            method=method,
            scalefactor=scalefactor,
        )
        self.update(kwargs)

class ExpectedReturn: # This class is used to estimate the expected return of a portfolio. It uses the ExpectedReturnSpecification class to define the method and scalefactor for the estimation.

    def __init__(self,
                 spec: Optional[ExpectedReturnSpecification] = None, # If nothing is specified within the class "ExpectedReturn()", the default method is "geometric" and the default scalefactor is one as predifined in the class "ExpectedReturnSpecification".
                 **kwargs):
        self.spec = ExpectedReturnSpecification() if spec is None else spec # If no specification is provided, it uses the default specification from the class "ExpectedReturnSpecification".
        self.spec.update(kwargs)
        self._vector: Union[pd.Series, np.ndarray, None] = None # It sets up a storage variable "_vector" that will later hold a return vector (like asset returns), but starts as "None" (empty), and is typed to be a "Series", "NumPy array" or "None".

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        if isinstance(value, ExpectedReturnSpecification): # Checks if the input value is of type "ExpectedReturnSpecification" otherwise it raises a "ValueError".
            self._spec = value
        else:
            raise ValueError(
                'Input value must be of type ExpectedReturnSpecification.'
            )
        return None # Not necessary, but it is a good practice to return None explicitly. If nothing is returned (no error), Python will return "None" by default.

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, value):
        if isinstance(value, (pd.Series, np.ndarray)): # Checks if the storage variable is of type "pd.Series" or "np.ndarray" otherwise it raises a "ValueError".
            self._vector = value
        else:
            raise ValueError(
                'Input value must be a pandas Series or a numpy array.'
            )
        return None # Not necessary, but it is a good practice to return None explicitly. If nothing is returned (no error), Python will return "None" by default.

    def estimate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        inplace: bool = True, # If "inplace" is set to "true" (the default), then "self.vector" is automatically updated with "mu". If "inplace" is set to "false", the method returns "mu" but does not update "self.vector" — you have to assign it manually.
    ) -> Union[pd.Series, np.ndarray, None]:

        scalefactor = self.spec.get('scalefactor', 1) # Get the value for 'scalefactor' from the "spec" dictionary. If it’s not found, return one as the default value.
        estimation_method = self.spec['method']

        if estimation_method == 'geometric':
            mu = mean_geometric(X=X, scalefactor=scalefactor)
        elif estimation_method == 'arithmetic':
            mu = mean_arithmetic(X=X, scalefactor=scalefactor)
        elif estimation_method == 'harmonic':
            mu = mean_harmonic(X=X, scalefactor=scalefactor)
        elif estimation_method == 'ewm':
            mu = mean_ewm(X=X, scalefactor=scalefactor, span=self.spec.get('span', 10), reverse=self.spec.get('reverse', True))
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

# You don’t always need a scalefactor in mean computation — but it can be useful when you want to scale or adjust the result of a mean for a specific purpose.

def mean_geometric(X: Union[pd.DataFrame, np.ndarray],
                   scalefactor: Union[float, int] = 1) -> Union[pd.Series, np.ndarray]:
    """
    Calculates the geometric mean of a list of numbers.

    Data input: The parameter X can be either a pandas.DataFrame or a numpy.ndarray.
    Def. scalefactor: The argument scalefactor can be either a float or an int. It has a default value of one.

    axis=0 → row-wise operations
    axis=1 → column-wise operations

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
    Def. scalefactor: The argument scalefactor can be either a float or an int. It has a default value of one.

    axis=0 → row-wise operations
    axis=1 → column-wise operations

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
    Def. scalefactor: The argument scalefactor can be either a float or an int. It has a default value of one.
    
    axis=0 → row-wise operations
    axis=1 → column-wise operations

    (n / sum(1/(1+X))) - 1 = mean_harmonic(X) → n = number of observations in the DataFrame or ndarray.
    
    Data output: The function returns a pandas series or a numpy array, depending on the input type of X.
    """
    n = X.shape[0]
    mu = ((n / np.sum(1 / (1 + X), axis=0)) * scalefactor) - 1
    return mu

def mean_ewm(X: Union[pd.DataFrame, np.ndarray], scalefactor: Union[float, int] = 1,
                    span: Union[float, int] = 10, # span = 10 it is a popular choice in practice.
                    reverse: bool = True) -> Union[pd.Series, np.ndarray]: # False = starting from the top and true = starting from the bottom.
    """
    Calculates the exponential weighted mean (ewm) of a list of numbers.

    Data input: The parameter X can be either a pandas.DataFrame or a numpy.ndarray.
    Def. scalefactor: The argument scalefactor can be either a float or an int. It has a default value of one.
    Def. span: The span parameter is a smoothing factor that determines the degree of weighting decrease. A smaller span results in more weight being given to recent observations, while a larger span gives more weight to older observations. It has a default value of ten.
    Def. reverse: If True, the function calculates the ewm starting from the bottom of the series (each column). If False, it starts from the top. This is useful for time series data where it depends where the calculation starts. It has a default value of true.

    i = t, where t is the time index of the series.
    sum(weight * value) / sum(weight) = mean_ewm(X)
    weight = alpha * (1 - alpha) ** i, where alpha = 2 / (span + 1).
    value = X[i], where i is the time index of the series.
    → It is done column by column (axis=0).
    
    Data output: The function returns a pandas series or a numpy array, depending on the input type of X.    
    """
    denominators_dict = {}
    numerators_dict = {}
    mu_dict = {}
    mu = []
    alpha = 2 / (span + 1) # Smoothing factor for ewm

    for col in X.columns:
        series = X[col].values
        if reverse:
            series = series[::-1]

        denominators = []
        numerators = []

        for t in range(len(series)):
            i = t
            weight = alpha * (1 - alpha) ** i
            value = series[i]
            numerators.append(weight * value)
            denominators.append(weight)

        # Store each list under the corresponding column name
        denominators_dict[col] = denominators
        numerators_dict[col] = numerators

    for col in X.columns:
        # print(f"Weights for {col}: {denominators_dict[col]}")
        # print(f"Weights * Value for {col}: {numerators_dict[col]}")
        mu = (np.sum(numerators_dict[col]) / np.sum(denominators_dict[col])) * scalefactor
        mu_dict[col] = mu

    return pd.Series(mu_dict, dtype = "float64")