############################################################################
### QPMwP - CLASS Strategy
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from typing import Union

# Third party imports
import numpy as np
import pandas as pd

# Local modules imports
from backtesting.portfolio import Portfolio




class Strategy:
    """
    A class to represent a trading strategy composed of portfolios
    that (may) vary over time.

    Attributes:
    ----------
    portfolios : list[Portfolio]
        A list of Portfolio objects representing the strategy over time.

    Methods:
    -------
    get_rebalancing_dates() -> list[str]
        Returns a list of rebalancing dates for the portfolios in the strategy.
    
    get_weights(rebalancing_date: str) -> dict[str, float]
        Returns the weights of the portfolio for a given rebalancing date.
    
    get_weights_df() -> pd.DataFrame
        Returns a DataFrame of portfolio weights with rebalancing dates as the index.
    
    get_portfolio(rebalancing_date: str) -> Portfolio
        Returns the portfolio for a given rebalancing date.
    
    has_previous_portfolio(rebalancing_date: str) -> bool
        Checks if there is a portfolio before the given rebalancing date.
    
    get_previous_portfolio(rebalancing_date: str) -> Portfolio
        Returns the portfolio immediately before the given rebalancing date.
    
    turnover(return_series: pd.DataFrame, rescale: bool = True) -> pd.Series
        Calculates the turnover for each rebalancing date.
    
    simulate(return_series: pd.DataFrame, fc: float = 0, vc: float = 0, n_days_per_year: int = 252) -> pd.Series
        Simulates the strategy's performance over time, accounting for fixed and variable costs.
    """
    def __init__(self, portfolios: list[Portfolio]):
        self.portfolios = portfolios

    @property
    def portfolios(self):
        return self._portfolios

    @portfolios.setter
    def portfolios(self, new_portfolios: list[Portfolio]):
        if not isinstance(new_portfolios, list):
            raise TypeError('portfolios must be a list')
        if not all(isinstance(portfolio, Portfolio) for portfolio in new_portfolios):
            raise TypeError('all elements in portfolios must be of type Portfolio')
        self._portfolios = new_portfolios

    def get_rebalancing_dates(self):
        return [portfolio.rebalancing_date for portfolio in self.portfolios]

    def get_weights(self, rebalancing_date: str) -> Union[dict[str, float], None]:
        for portfolio in self.portfolios:
            if portfolio.rebalancing_date == rebalancing_date:
                return portfolio.weights
        return None

    def get_weights_df(self) -> pd.DataFrame:
        weights_dict = {}
        for portfolio in self.portfolios:
            weights_dict[portfolio.rebalancing_date] = portfolio.weights
        return pd.DataFrame(weights_dict).T

    def get_portfolio(self, rebalancing_date: str) -> Portfolio:
        if rebalancing_date in self.get_rebalancing_dates():
            idx = self.get_rebalancing_dates().index(rebalancing_date)
            return self.portfolios[idx]
        else:
            raise ValueError(f'No portfolio found for rebalancing date {rebalancing_date}')

    def has_previous_portfolio(self, rebalancing_date: str) -> bool:
        dates = self.get_rebalancing_dates()
        ans = False
        if len(dates) > 0:
            ans = dates[0] < rebalancing_date
        return ans

    def get_previous_portfolio(self, rebalancing_date: str) -> Portfolio:
        if not self.has_previous_portfolio(rebalancing_date):
            return Portfolio.empty()
        else:
            yesterday = [x for x in self.get_rebalancing_dates() if x < rebalancing_date][-1]
            return self.get_portfolio(yesterday)

    def turnover(self, return_series: pd.DataFrame, rescale: bool=True):
    
        raise NotImplementedError('The method turnover is not yet implemented.')

    def simulate(self,
                 return_series: pd.DataFrame,
                 fc: float = 0,
                 vc: float = 0,
                 n_days_per_year: int = 252) -> pd.Series:

        rebdates = self.get_rebalancing_dates()
        ret_list = []
        for rebdate in rebdates:
            next_rebdate = (
                rebdates[rebdates.index(rebdate) + 1]
                if rebdate < rebdates[-1]
                else return_series.index[-1]
            )

            portfolio = self.get_portfolio(rebdate)
            w_float = portfolio.float_weights(
                return_series=return_series,
                end_date=next_rebdate,
                rescale=False # Notice that rescale is hardcoded to False.
            )
            level = w_float.sum(axis=1)
            ret_tmp = level.pct_change(1)  # 1 for one day lookback
            ret_list.append(ret_tmp)

        portf_ret = pd.concat(ret_list).dropna()

        if vc != 0:
            raise NotImplementedError('Variable costs are not yet implemented.')

        if fc != 0:
            raise NotImplementedError('Fixed costs are not yet implemented.')

        return portf_ret

