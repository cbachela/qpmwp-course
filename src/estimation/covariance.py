############################################################################
### QPMwP - COVARIANCE
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# # Standard library imports
from typing import Union, Optional

# Third party imports
import numpy as np
import pandas as pd



# TODO:

# [ ] Add covariance functions:
#    [ ] cov_linear_shrinkage
#    [ ] cov_nonlinear_shrinkage
#    [ ] cov_factor_model
#    [ ] cov_robust
#    [ ] cov_ewma (expoential weighted moving average)
#    [ ] cov_garch
#    [ ] cov_dcc (dynamic conditional correlation)
#    [ ] cov_pc_garch (principal components garch)
#    [ ] cov_ic_garch (independent components analysis)
#    [ ] cov_constant_correlation


# [ ] Add helper methods:
#    [ ] is_pos_def
#    [ ] is_pos_semidef
#    [ ] is_symmetric
#    [ ] is_correlation_matrix
#    [ ] is_diagonal
#    [ ] make_symmetric
#    [ ] make_pos_def
#    [ ] make_correlation_matrix (from covariance matrix)
#    [ ] make_covariance_matrix (from correlation matrix)





class CovarianceSpecification(dict):

    def __init__(self,
                 method='pearson', 
                #now only one method - pearson
                #  check_positive_definite=False,
                 **kwargs):
        super().__init__(
            method=method,
            # check_positive_definite=check_positive_definite,
        )
        self.update(kwargs)


class Covariance:

    def __init__(self,
                 spec: Optional[CovarianceSpecification] = None,
                 **kwargs):
        self.spec = CovarianceSpecification() if spec is None else spec
        self.spec.update(kwargs)
        self._matrix: Union[pd.DataFrame, np.ndarray, None] = None

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        if isinstance(value, CovarianceSpecification):
            self._spec = value
        else:
            raise ValueError(
                'Input value must be of type CovarianceSpecification.'
            )
        return None

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if isinstance(value, (pd.DataFrame, np.ndarray)):
            self._matrix = value
        else:
            raise ValueError(
                'Input value must be a pandas DataFrame or a numpy array.'
            )
        return None

    def estimate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        inplace: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray, None]:

        estimation_method = self.spec['method']

        if estimation_method == 'pearson':
            cov_matrix = cov_pearson(X=X)
        else:
            raise ValueError(
                'Estimation method not recognized.'
            )

        # if self.spec.get('check_positive_definite'):
        #     if not isPD(covmat):
        #         covmat = nearestPD(covmat)

        if inplace:
            self.matrix = cov_matrix
            return None
        else:
            return cov_matrix






# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def cov_pearson(X:  Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    if isinstance(X, pd.DataFrame):
        covmat = X.cov()
    else:
        covmat = np.cov(X, rowvar=False)
    return covmat
