import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def cross_sectional_ic(y_true, y_pred):
    df = pd.DataFrame({"y": y_true, "pred": y_pred})
    if isinstance(y_true.index, pd.MultiIndex):
        df.index = y_true.index
        date_level = 0
    else:
        raise ValueError("Need MultiIndex with DATE level")

    ic_list = []
    for _, group in df.groupby(level=date_level):
        if len(group) > 5:
            ic = group["y"].corr(group["pred"], method="spearman")
            if not np.isnan(ic):
                ic_list.append(ic)
    return np.mean(ic_list) if ic_list else 0.0


def ic_score_func(y_true, y_pred):
    return cross_sectional_ic(y_true, pd.Series(y_pred, index=y_true.index))



def _cross_sectional_metric(y_true, y_pred, metric_func):
    """Helper function to compute cross-sectional metrics per date."""
    assert y_true.index.names == ['DATE', 'ID']
    assert y_pred.index.names == ['DATE', 'ID']

    y = y_true.reindex(y_pred.index).dropna()
    y_hat = y_pred.reindex(y.index).dropna()

    performance = pd.Series(index=y_hat.index.get_level_values("DATE").unique(), dtype=float)

    for date in y_hat.index.get_level_values("DATE").unique():
        y_true_date = y.reindex(y_hat.index[y_hat.index.get_level_values("DATE") == date]).dropna()
        y_pred_date = y_hat[y_hat.index.get_level_values("DATE") == date].dropna()

        assert y_true_date.index.equals(y_pred_date.index)
        performance.loc[date] = metric_func(y_true_date, y_pred_date.values.reshape(-1))
    
    return performance


def spearman_correlation_per_date(y_true, y_pred):
    return _cross_sectional_metric(y_true, y_pred, ic_score_func)


def mae_per_date(y_true, y_pred):
    def mae_score(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    return _cross_sectional_metric(y_true, y_pred, mae_score)