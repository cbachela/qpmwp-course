import numpy as np
from scipy.stats import rankdata
from .base import PanelTransform




class CrossSectionalRank(PanelTransform):
    """Converts values to ranks cross-sectionally"""

    def __init__(self, method='average', as_percentile=False):
        self.method = method  # rankdata method (applied to ties): 'average', 'min', 'max', etc.
        self.as_percentile = as_percentile  # if True, ranks are scaled to percentiles [0,100]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._validate_panel(X)

        def _rank_transform(x):
            ranks = rankdata(x, method=self.method)
            if self.as_percentile:
                ranks = ranks / (len(ranks) + 1) * 100
            return ranks.astype(int)

        return X.groupby(level="DATE").transform(_rank_transform)


class CrossSectionalPIT(PanelTransform):
    """Probability Integral Transform - converts to uniform [0,1] ranks cross-sectionally"""

    def __init__(self, method='average'):
        self.method = method  # rankdata method (applied to ties): 'average', 'min', 'max', etc.

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


class CrossSectionalPercentiles(PanelTransform):

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._validate_panel(X)

        X = X.sort_index()

        def percentile_bins(x, n_bins=10):
            """
            Assign each value in x to a percentile bin.
            n_bins = number of equally sized percentile buckets.
            Returns integers 0 … n_bins-1.
            """
            x = np.asarray(x)

            # Percentile cutpoints: 0%, 100/n_bins %, ..., 100%
            cuts = np.percentile(x, np.linspace(0, 100, n_bins + 1))

            # Digitize assigns bins; right=True means cuts[i] < x <= cuts[i+1]
            dec = np.digitize(x, cuts[1:-1], right=True)

            return dec

        X_cs = X.groupby(level="DATE").transform(percentile_bins, n_bins=self.n_bins)
        return X_cs
