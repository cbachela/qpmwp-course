import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRanker
 
 
class XGBRankerSklearnWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for XGBRanker.

    The problem: sklearn Pipeline/GridSearchCV only calls fit(X, y).
    XGBRanker additionally needs qid=... to know which rows belong to the same query (=DATE).

    Solution: qid is extracted internally from X — either from a 'qid' column 
    (if explicitly added) or from the (DATE, ID) MultiIndex.

    - no manual handling of qid in xai.py, traintest.py, etc.
    - Pipeline preprocessing steps work transparently
    """

    def __init__(
        self,
        objective: str = "rank:ndcg",
        eta: float = 0.05,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        ndcg_exp_gain: bool = False, # if True, the gain of a document is 2^rel - 1, otherwise it's rel. Setting to False makes it more interpretable and comparable to linear models in SHAP and feature importance.
        verbosity: int = 0,
    ):
        self.objective = objective
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.ndcg_exp_gain = ndcg_exp_gain
        self.verbosity = verbosity
 
    # ------------------------------------------------------------------
    # Extract qid from X, either from 'qid' column or from MultiIndex (DATE, ID)
    # ------------------------------------------------------------------
 
    def _extract_qid(self, X: pd.DataFrame):
        """
        Extracts qid and gives gives (X_without_qid, qid_array) back.
 
        Supports the following:
        1. X has 'qid' column -> qid is taken from there, X_clean is X without 'qid' column
        2. X has a (DATE, ID) MultiIndex → qid becomes integer encoding of DATE, X_clean is X as is (MultiIndex bleibt erhalten)
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if "qid" in X.columns:
            qid = X["qid"].values.astype(int)
            X_clean = X.drop(columns=["qid"])

        elif isinstance(X.index, pd.MultiIndex):
            try:
                dates = X.index.get_level_values("DATE")
            except KeyError:
                dates = X.index.get_level_values(0)
            unique_dates = pd.DatetimeIndex(dates).unique().sort_values()
            date_to_int = {d: i for i, d in enumerate(unique_dates)}
            qid = np.array([date_to_int[d] for d in dates], dtype=int)
            X_clean = X

        else:
            raise ValueError(
                "X needs either a 'qid' column or a (DATE, ID) MultiIndex."
                "Neither of the two was found."
            )

        return X_clean, qid

    # ------------------------------------------------------------------
    # fit / predict — these are the methods Pipeline and GridSearchCV call internally.
    # The wrapper handles qid extraction transparently before delegating to XGBRanker.
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_clean, qid = self._extract_qid(X)

        self.ranker_ = XGBRanker(
            objective=self.objective,
            eta=self.eta,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            verbosity=self.verbosity,
            ndcg_exp_gain=self.ndcg_exp_gain, # if True, the gain of a document is 2^rel - 1, otherwise it's rel. Setting to False makes it more interpretable and comparable to linear models in SHAP and feature importance.
        )

        self.ranker_.fit(X_clean, y, qid=qid)

        # (for SHAP and feature importance)
        self.feature_names_in_ = np.array(X_clean.columns.tolist())

        return self

    def predict(self, X):
        X_clean, _ = self._extract_qid(X) 
        return self.ranker_.predict(X_clean)
 
    # ------------------------------------------------------------------
    # For SHAP and Feature Importance
    # ------------------------------------------------------------------

    def get_booster(self):
        """Gives XGBoost Booster back for SHAP TreeExplainer."""
        return self.ranker_.get_booster()
