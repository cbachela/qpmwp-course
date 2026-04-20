import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PanelTransform(BaseEstimator, TransformerMixin):
    REQUIRED_LEVELS = {"DATE", "ID"}

    def _validate_panel(self, X):
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex (DATE, ID)")

        levels = set(X.index.names)
        if not self.REQUIRED_LEVELS.issubset(levels):
            raise ValueError(f"Index must contain {self.REQUIRED_LEVELS}, got {levels}")
