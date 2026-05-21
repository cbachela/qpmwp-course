from typing import Iterator, Tuple, Union
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.model_selection import BaseCrossValidator


def iter_rolling_dates(
    live_start_dt: str,
    live_end_dt: str,
    train_days: int = 365 * 3,
    skip_days_between_train_test: int = 10,
    rolling_freq: Union[str, DateOffset] = "12MS",
) -> Iterator[Tuple[str, str, str, str]]:
    """
    Generates rolling date ranges for train-test splits in a time-series setting.

    Parameters:
    ----------
    live_start_dt : str
        The start date of the live testing period, in 'YYYY-MM-DD' format.
    live_end_dt : str
        The end date of the live testing period, in 'YYYY-MM-DD' format.
    train_days : int, optional
        The number of days included in the training period for each split.
        Default is 3 years (365 * 3 days).
    skip_days_between_train_test : int, optional
        The number of days to skip between the end of the training period and
        the start of the testing period. Default is 10 days.
    rolling_freq : Union[str, DateOffset], optional
        The frequency of rolling test periods. Default is "12MS" (12 months start frequency).

    Returns:
    -------
    Iterator[Tuple[str, str, str, str]]
        An iterator yielding tuples of date strings for each split:
        - `train_start_dt`: Start date of the training period.
        - `train_end_dt`: End date of the training period.
        - `test_start_dt`: Start date of the testing period.
        - `test_end_dt`: End date of the testing period.
    """

    date_range = pd.date_range(live_start_dt, live_end_dt, freq=rolling_freq)

    if len(date_range) == 1:
        date_range = [
            pd.to_datetime(live_start_dt),
            pd.to_datetime(live_end_dt),
        ]

    def _make_rolling_date(test_start, test_end):
        return (
            # train_start_dt
            (test_start - pd.Timedelta(days=train_days)).strftime("%Y-%m-%d"),
            # train_end_dt
            (test_start - pd.Timedelta(days=skip_days_between_train_test + 1)).strftime("%Y-%m-%d"),
            # test_start_dt
            test_start.strftime("%Y-%m-%d"),
            # test_end_dt
            test_end.strftime("%Y-%m-%d"),
        )

    for idx in range(1, len(date_range)):
        test_start: pd.Timestamp = date_range[idx - 1]
        test_end: pd.Timestamp = date_range[idx] - pd.Timedelta(days=1)

        yield _make_rolling_date(test_start=test_start, test_end=test_end)

    test_start: pd.Timestamp = date_range[-1]
    test_end = pd.to_datetime(live_end_dt)

    if test_start != test_end:
        yield _make_rolling_date(test_start=test_start, test_end=test_end)


def add_buffer_days_to_rolling_dates(rolling_date, start_buffer_days, end_buffer_days):
    """
    Adds buffer days to rolling train-test split dates.

    Parameters:
    ----------
    rolling_date : Tuple[str, str, str, str]
        A tuple of date strings in the format:
        - `rolling_date[0]`: Start date of the training period.
        - `rolling_date[1]`: End date of the training period.
        - `rolling_date[2]`: Start date of the testing period.
        - `rolling_date[3]`: End date of the testing period.
    start_buffer_days : int
        Number of buffer days to subtract from the start dates of both the training
        and testing periods.
    end_buffer_days : int
        Number of buffer days to add to the end date of the testing period.

    Returns:
    -------
    Tuple[str, str, str, str]
        A tuple of adjusted date strings with buffer days applied:
        - `insample_start_dt`: Adjusted start date of the training period.
        - `insample_end_dt`: End date of the training period (unchanged).
        - `outofsample_start_dt`: Adjusted start date of the testing period.
        - `outofsample_end_dt`: Adjusted end date of the testing period.
    """

    insample_start_dt = (
        pd.to_datetime(rolling_date[0]) - pd.Timedelta(days=start_buffer_days)
    ).strftime("%Y-%m-%d")
    insample_end_dt = (pd.to_datetime(rolling_date[1])).strftime("%Y-%m-%d")
    outofsample_start_dt = (
        pd.to_datetime(rolling_date[2]) - pd.Timedelta(days=start_buffer_days)
    ).strftime("%Y-%m-%d")
    outofsample_end_dt = (
        pd.to_datetime(rolling_date[3]) + pd.Timedelta(days=end_buffer_days)
    ).strftime("%Y-%m-%d")

    return (
        insample_start_dt,
        insample_end_dt,
        outofsample_start_dt,
        outofsample_end_dt,
    )


class RollingTimeSeriesSplit(BaseCrossValidator):
    """
    Custom time-series cross-validator that splits the data into rolling train-test sets.

    Parameters:
    ----------
    start_dt : str, default="2000-01-01"
        Start date for the rolling periods, in 'YYYY-MM-DD' format.
    end_dt : str, default="2023-12-31"
        End date for the rolling periods, in 'YYYY-MM-DD' format.
    train_days : int, default=365 * 3
        The number of days included in each training period.
    skip_days_between_train_test : int, default=21
        Number of days to skip between the end of the training period and the start of
        the testing period for each split.
    rolling_freq : Union[str, DateOffset], default="12MS"
        Frequency of the rolling test periods. Can be specified as a string (e.g., "12MS")
        or a Pandas `DateOffset` object.

    Methods:
    -------
    split(X, y=None, groups=None):
        Generates train-test splits based on the rolling date ranges.
    get_n_splits(X, y=None, groups=None):
        Returns the number of train-test splits.
    iter_rolling_dates():
        Iterates over rolling date ranges based on the specified parameters.

    Notes:
    ------
    - This cross-validator is designed for time-series data and does not shuffle or randomize data.
    - Train-test splits are generated sequentially with training periods ending before testing periods begin.
    - Ensure that `start_dt` and `end_dt` cover the range of the input data (`X` or `y`).
    - The training period length is determined by `train_days`, while the testing period length is defined
      implicitly by the `rolling_freq` and the distance between successive test periods.
    """

    def __init__(
        self,
        start_dt: str = "2000-01-01",
        end_dt: str = "2023-12-31",
        train_days: int = 365 * 3,
        skip_days_between_train_test: int = 21,
        rolling_freq: Union[str, DateOffset] = "12MS",
    ):
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.train_days = train_days
        self.skip_days_between_train_test = skip_days_between_train_test
        self.rolling_freq = rolling_freq
        assert train_days > skip_days_between_train_test

    def split(self, X, y=None, groups=None):
        if groups is not None:
            raise NotImplementedError("groups is not supported")

        data = X if X is not None else y
        assert isinstance(data, (pd.DataFrame, pd.Series)) or (
            hasattr(data, "loc") and hasattr(data, "index")
        )

        for rolling_date in self.iter_rolling_dates():
            (
                insample_start_dt,
                insample_end_dt,
                outofsample_start_dt,
                outofsample_end_dt,
            ) = rolling_date

            train, test = (
                data.loc[insample_start_dt:insample_end_dt].index,
                data.loc[outofsample_start_dt:outofsample_end_dt].index,
            )
            if len(train) == 0 or len(test) == 0:
                continue

            yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        n = 0
        for _ in self.split(X=X, y=y, groups=groups):
            n += 1

        return n

    def iter_rolling_dates(self) -> Iterator[Tuple[str, str, str, str]]:
        for rolling_date in iter_rolling_dates(
            live_start_dt=self.start_dt,
            live_end_dt=self.end_dt,
            train_days=self.train_days,
            skip_days_between_train_test=self.skip_days_between_train_test,
            rolling_freq=self.rolling_freq,
        ):
            yield rolling_date


def iter_rolling_dates_from_grid(
    observation_dates: pd.DatetimeIndex,
    train_window_obs: int,
    skip_obs_between_train_test: int = 1,
    retrain_stride: int = 1,
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate rolling train/test date tuples based on an observation grid.

    Parameters
    - observation_dates: sorted DatetimeIndex of available observation timestamps
    - train_window_obs: number of observations to include in the training window
    - skip_obs_between_train_test: number of observation steps to skip between train end and test start
    - retrain_stride: step in observation units between successive test starts

    Yields tuples: (train_start_dt, train_end_dt, test_start_dt, test_end_dt) as Timestamps
    """
    if len(observation_dates) == 0:
        return

    dates = pd.DatetimeIndex(sorted(observation_dates))
    n = len(dates)

    # earliest possible test start index that allows a full training window
    first_test_idx = train_window_obs + skip_obs_between_train_test
    if first_test_idx >= n:
        return

    test_start_positions = list(range(first_test_idx, n, retrain_stride))
    

    for idx_pos, test_start_pos in enumerate(test_start_positions):
        test_start = dates[test_start_pos]
        # test end is next test start - 1 observation, or end of dates
        if idx_pos + 1 < len(test_start_positions):
            test_end_pos = test_start_positions[idx_pos + 1] - 1
        else:
            test_end_pos = n - 1
        test_end = dates[test_end_pos]

        train_end_pos = test_start_pos - skip_obs_between_train_test - 1
        train_start_pos = train_end_pos - train_window_obs + 1

        if train_start_pos < 0 or train_end_pos < 0:
            # not enough history for this split
            continue

        train_start = dates[train_start_pos]
        train_end = dates[train_end_pos]

        yield (train_start, train_end, test_start, test_end)


class ObservationGridRollingSplit(BaseCrossValidator):
    """Simple observation-grid based rolling splitter.

    This splitter operates on an observation grid (sequence of timestamps)
    and counts windows in observation-units (e.g., months, quarters, days).

    Parameters
    - observation_dates: Optional[Iterable[pd.Timestamp]]; if not provided
      the splitter will infer unique DATE values from the passed data index.
    - train_window_obs: int number of observations for training window.
    - skip_obs_between_train_test: int number of observations to skip to avoid leakage.
    - retrain_stride: int stride in observation units between successive test starts.
    """

    def __init__(
        self,
        observation_dates: Union[pd.DatetimeIndex, None] = None,
        train_window_obs: int = 36,
        skip_obs_between_train_test: int = 1,
        retrain_stride: int = 1,
    ):
        self.observation_dates = (
            pd.DatetimeIndex(sorted(observation_dates)) if observation_dates is not None else None
        )
        self.train_window_obs = train_window_obs
        self.skip_obs_between_train_test = skip_obs_between_train_test
        self.retrain_stride = retrain_stride

    def _infer_observation_dates_from_data(self, data):
        # Accept DataFrame/Series with DatetimeIndex or MultiIndex (DATE, ID)
        if hasattr(data, "index"):
            idx = data.index
            if isinstance(idx, pd.MultiIndex):
                # first level assumed to be DATE
                dates = pd.DatetimeIndex(idx.get_level_values(0))
            else:
                dates = pd.DatetimeIndex(idx)
            # unique and sorted
            return pd.DatetimeIndex(sorted(dates.unique()))
        raise ValueError("Cannot infer observation dates from provided data")

    def split(self, X, y=None, groups=None):
        if groups is not None:
            raise NotImplementedError("groups is not supported")

        data = X if X is not None else y
        assert data is not None, "Either X or y must be provided"

        obs_dates = (
            self.observation_dates
            if self.observation_dates is not None
            else self._infer_observation_dates_from_data(data)
        )

        for train_start, train_end, test_start, test_end in iter_rolling_dates_from_grid(
            obs_dates,
            train_window_obs=self.train_window_obs,
            skip_obs_between_train_test=self.skip_obs_between_train_test,
            retrain_stride=self.retrain_stride,
        ):
            # select rows where DATE in [train_start, train_end] and [test_start, test_end]
            idx = data.index
            if isinstance(idx, pd.MultiIndex):
                dates_level = idx.get_level_values(0)
            else:
                dates_level = pd.DatetimeIndex(idx)

            train_mask = (dates_level >= train_start) & (dates_level <= train_end)
            test_mask = (dates_level >= test_start) & (dates_level <= test_end)

            train_idx = idx[train_mask]
            test_idx = idx[test_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    # ---- Convenience helpers for human-readable inspection ----
    def _extract_date_level(self, idx):
        """Return a DatetimeIndex representing the date level for `idx`.

        Handles MultiIndex with a `DATE` level, MultiIndex with first level as date,
        or a plain DatetimeIndex.
        """
        if isinstance(idx, pd.MultiIndex):
            # prefer named level `DATE` when present
            if "DATE" in idx.names:
                dates = pd.DatetimeIndex(idx.get_level_values("DATE"))
            else:
                dates = pd.DatetimeIndex(idx.get_level_values(0))
        else:
            dates = pd.DatetimeIndex(idx)
        return dates

    def split_info(self, train_idx, test_idx):
        """Return a dict with human-readable info about a single split.

        Keys: `train_start`, `train_end`, `train_n`, `train_unique_dates`, `train_dates`,
              `test_start`, `test_end`, `test_n`, `test_unique_dates`, `test_dates`.
        """
        t_dates = self._extract_date_level(train_idx)
        v_dates = self._extract_date_level(test_idx)

        train_dates = t_dates.unique().sort_values()
        test_dates = v_dates.unique().sort_values()

        info = {
            "train_start": pd.Timestamp(train_dates.min()),
            "train_end": pd.Timestamp(train_dates.max()),
            "train_n": len(train_idx),
            "train_unique_dates": len(train_dates),
            "train_dates": train_dates.strftime("%Y-%m-%d").tolist(),
            "test_start": pd.Timestamp(test_dates.min()),
            "test_end": pd.Timestamp(test_dates.max()),
            "test_n": len(test_idx),
            "test_unique_dates": len(test_dates),
            "test_dates": test_dates.strftime("%Y-%m-%d").tolist(),
        }
        return info

    def print_splits(self, X, limit=None):
        """Iterate the splits and print human-friendly information.

        Parameters
        - X: DataFrame or Series used to drive the split (passed to `split`).
        - limit: Optional[int] maximum number of splits to print.
        """
        for i, (train_idx, test_idx) in enumerate(self.split(X=X), start=1):
            if limit is not None and i > limit:
                break

            info = self.split_info(train_idx, test_idx)

            print(f"\n{'=' * 12} [iteration {i:02d}] {'=' * 12}")
            print("training_period")
            print(
                "start <-> end:",
                info["train_start"].strftime("%Y-%m-%d"),
                "<->",
                info["train_end"].strftime("%Y-%m-%d"),
            )
            print("number of samples:", info["train_n"])
            print("number of unique dates:", info["train_unique_dates"])
            print("dates:", info["train_dates"])

            print("testing_period")
            print(
                "start <-> end:",
                info["test_start"].strftime("%Y-%m-%d"),
                "<->",
                info["test_end"].strftime("%Y-%m-%d"),
            )
            print("number of samples:", info["test_n"])
            print("number of unique dates:", info["test_unique_dates"])
            print("dates:", info["test_dates"])

    def get_n_splits(self, X, y=None, groups=None):
        n = 0
        for _ in self.split(X=X, y=y, groups=groups):
            n += 1
        return n

    def get_split_at(self, X, t):
        for i, (train_idx, test_idx) in enumerate(self.split(X=X)):
            if i == t:
                return train_idx, test_idx
        raise IndexError(f"split_index {t} out of range (only {i} splits available)")



class PanelTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=3, date_level="DATE"):
        self.n_splits = n_splits
        self.date_level = date_level

    def split(self, X, y=None, groups=None):
        dates = (
            X.index.get_level_values(self.date_level)
            .unique()
            .sort_values()
        )

        n_dates = len(dates)

        # Create split points evenly spaced over dates
        split_points = np.linspace(
            0,
            n_dates,
            self.n_splits + 2,   # +2 gives us train/val structure
            dtype=int
        )

        for i in range(self.n_splits):
            train_end = split_points[i + 1]
            val_end = split_points[i + 2]

            train_dates = dates[:train_end]
            val_dates = dates[train_end:val_end]

            train_mask = X.index.get_level_values(self.date_level).isin(train_dates)
            val_mask = X.index.get_level_values(self.date_level).isin(val_dates)

            yield np.where(train_mask)[0], np.where(val_mask)[0]

    def show(self, X):
        for train_idx, val_idx in self.split(X=X):
            print("TRAIN:", X.index[train_idx].get_level_values(self.date_level).unique())
            print("VAL:", X.index[val_idx].get_level_values(self.date_level).unique())
            print("-" * 20)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# # old version misses last 2 quarters fo rexample
# class PanelTimeSeriesSplit(BaseCrossValidator):
#     def __init__(self, n_splits=3, date_level="DATE"):
#         self.n_splits = n_splits
#         self.date_level = date_level

#     def split(self, X, y=None, groups=None):
#         dates = X.index.get_level_values(self.date_level).unique().sort_values()

#         fold_size = len(dates) // (self.n_splits + 1)

#         for i in range(self.n_splits):
#             train_end = fold_size * (i + 1)
#             val_end = fold_size * (i + 2)

#             train_dates = dates[:train_end]
#             val_dates = dates[train_end:val_end]

#             train_mask = X.index.get_level_values(self.date_level).isin(train_dates)
#             val_mask = X.index.get_level_values(self.date_level).isin(val_dates)

#             train_idx = np.where(train_mask)[0]
#             val_idx = np.where(val_mask)[0]

#             yield train_idx, val_idx

#     def get_n_splits(self, X=None, y=None, groups=None):
#         return self.n_splits
