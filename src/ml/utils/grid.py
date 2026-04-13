import pandas as pd
import numpy as np
from tqdm import tqdm


def compound_returns_on_grid(
    return_series: pd.Series,
    time_grid: pd.DatetimeIndex,
    date_level: str = "DATE",
    id_level: str = "ID",
) -> pd.DataFrame:
    """
    Compounds daily returns between consecutive dates in time_grid.

    Parameters
    ----------
    return_series : pd.Series
        Daily returns indexed by MultiIndex (DATE, ID).
    time_grid : pd.DatetimeIndex
        Sorted dates defining the feature / prediction grid.
    date_level : str
        Name of the date level in the MultiIndex.
    id_level : str
        Name of the asset identifier level.

    Returns
    -------
    pd.DataFrame
        Compounded returns indexed by (DATE, ID),
        where DATE corresponds to the end of each grid period.
    """

    # ensure proper ordering
    time_grid = pd.DatetimeIndex(time_grid).sort_values()
    return_series = return_series.sort_index()

    out = []

    for t0, t1 in tqdm(zip(time_grid[:-1], time_grid[1:]), desc="grid returns..."):
        mask = (return_series.index.get_level_values(date_level) > t0) & (
            return_series.index.get_level_values(date_level) <= t1
        )

        period_ret = (1.0 + return_series[mask]).groupby(level=id_level).prod().sub(1.0)

        # the observation is only measurable at the end of the t1 day
        # hence it is crucial that t1 is used.
        period_ret.index = pd.MultiIndex.from_product(
            [[t1], period_ret.index],
            names=[date_level, id_level],
        )
        out.append(period_ret)

    if not out:
        return pd.DataFrame(columns=["return"])

    return pd.concat(out).to_frame("return")


def shift_grid_returns(grid_ret: pd.DataFrame, shift: int = -1) -> pd.Series:
    """
    Shifts returns in grid_ret by a specified number of periods within each asset group to align with the prediction target.

    Parameters
    ----------
    grid_ret : pd.DataFrame
        DataFrame containing returns indexed by MultiIndex (DATE, ID).

    Returns
    -------
    pd.Series
        Shifted returns indexed by MultiIndex (DATE, ID), where each return corresponds to the next period's return.
    """
    y = grid_ret.groupby(level="ID")["return"].shift(shift).dropna()
    return y
