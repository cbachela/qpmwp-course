from typing import Union
import pandas as pd


def ensure_datetime_index(series: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Ensure the index (or first MultiIndex level) is converted to datetime."""
    if isinstance(series.index, pd.MultiIndex):
        series.index = series.index.set_levels(pd.to_datetime(series.index.levels[0]), level="DATE")
    else:
        series.index = pd.to_datetime(series.index)
    return series


def check_if_multiindex(obj):
    if not isinstance(obj.index, pd.MultiIndex) or list(obj.index.names) != ["DATE", "ID"]:
        raise ValueError("Input must be a Series or DataFrame with MultiIndex (DATE, ID).")
