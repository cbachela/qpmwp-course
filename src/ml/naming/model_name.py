import pandas as pd


def resolve_start_end_dates_from_index(idx):
    """Return (start_dt, end_dt) Timestamps extracted from the first level of `idx`.

    Handles:
    - DatetimeIndex
    - MultiIndex (uses level 0)
    - Index of tuple-labels where first element is a date-like value
    """
    if idx is None or len(idx) == 0:
        raise ValueError("index is empty")

    if isinstance(idx, pd.MultiIndex):
        dates = pd.DatetimeIndex(idx.get_level_values(0))
    else:
        first = idx[0]
        if isinstance(first, tuple):
            # assume first element of tuple is the date level
            try:
                dates = pd.DatetimeIndex([t[0] for t in idx])
            except Exception:
                dates = pd.DatetimeIndex(idx)
        else:
            dates = pd.DatetimeIndex(idx)

    dates = dates.sort_values()
    return dates[0], dates[-1]


def resolve_target_name(y, target_asset=None):
    """Resolve the target asset name from `y` and `target_asset` parameters.

    If `y` is a Series, use `y.name` or fallback to "target" if `target_asset` is not provided.
    If `y` is a DataFrame, require `target_asset` to be provided and validate it exists in `y.columns`.
    """
    if isinstance(y, pd.Series):
        if target_asset is None:
            return y.name or "target"
        else:
            return target_asset
    else:
        assert target_asset is not None, "target_asset must be provided when y is a DataFrame"
        assert target_asset in y.columns, f"no label found for {target_asset}"
        return target_asset
