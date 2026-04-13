import numpy as np
from scipy.stats import rankdata
from .base import PanelTransform


class CrossSectionalPIT(PanelTransform):
    """Probability Integral Transform - converts to uniform [0,1] ranks cross-sectionally"""
    
    def __init__(self, method='average'):
        self.method = method  # rankdata method: 'average', 'min', 'max', etc.
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self._validate_panel(X)
        
        def _pit_transform(x):
            ranks = rankdata(x, method=self.method)
            return ranks / (len(ranks) + 1)
        
        return X.groupby(level="DATE").transform(_pit_transform)


class CrossSectionalWinsorize(PanelTransform):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._validate_panel(X)

        def _winsorize(x):
            lower_q = x.quantile(self.lower)
            upper_q = x.quantile(self.upper)
            return x.clip(lower_q, upper_q)

        return X.groupby(level="DATE").transform(_winsorize)


class CrossSectionalClip(PanelTransform):
    
    def __init__(self, lower=-3.0, upper=3.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._validate_panel(X)

        def _clip(x):
            return x.clip(self.lower, self.upper)

        return X.groupby(level="DATE").transform(_clip)


class CrossSectionalZScore(PanelTransform):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._validate_panel(X)

        X = X.sort_index()

        X_cs = X.groupby(level="DATE").transform(lambda x: (x - x.mean()) / x.std())

        return X_cs.replace([np.inf, -np.inf], np.nan)
