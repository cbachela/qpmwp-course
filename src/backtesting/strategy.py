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
from backtesting.portfolio import Portfolio, floating_weights




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

        dates = self.get_rebalancing_dates()
        to = {}
        to[dates[0]] = float(1)
        for rebalancing_date in dates[1:]:

            previous_portfolio = self.get_previous_portfolio(rebalancing_date=rebalancing_date)
            current_portfolio = self.get_portfolio(rebalancing_date=rebalancing_date)

            if current_portfolio.rebalancing_date is None or previous_portfolio.rebalancing_date is None:
                raise ValueError('Portfolios must have a rebalancing date')

            if current_portfolio.rebalancing_date < previous_portfolio.rebalancing_date:
                raise ValueError('The previous portfolio must be older than the current portfolio')

            # Get the union of the ids of the weights in both portfolios
            ids_union = list(
                set(
                    current_portfolio.weights.keys())
                    .union(set(previous_portfolio.weights.keys())
                )
            )

            # Extend the weights of the portfolio of the previous rebalancing
            # to the the union of ids in both portfolios by adding zeros
            w0 = pd.Series(previous_portfolio.weights, index=ids_union).fillna(0)

            # Float the weights according to the price drifts in the market
            # until the new rebalancing date
            w_init = floating_weights(
                X=return_series,
                w=w0,
                start_date=previous_portfolio.rebalancing_date,
                end_date=current_portfolio.rebalancing_date,
                rescale=rescale
            )

            # Extract the weights of the portfolio of the current rebalancing date
            w_current = pd.Series(current_portfolio.weights, index=ids_union).fillna(0)

            # Calculate the turnover
            to[rebalancing_date] = (
                pd.Series(w_init.iloc[-1])
                .sub(pd.Series(w_current), fill_value=0)
                .abs().sum()
            )
        return pd.Series(to)

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
            ret_tmp = level.pct_change(1)
            ret_list.append(ret_tmp)

        portf_ret = pd.concat(ret_list).dropna()

        if vc != 0:
            # Calculate turnover and variable cost (vc) as a fraction of turnover
            # Subtract the variable cost from the returns at each rebalancing date
            to = self.turnover(return_series=return_series,
                               rescale=False)
            varcost = to * vc
            portf_ret[0] -= varcost[0]
            portf_ret[varcost[1:].index] -= varcost[1:].values

        if fc != 0:
            # Calculate number of days between returns
            # Calculate daily fixed cost based on the annual fixed cost (fc),
            # the number of days between two rebalancings and the number of days per year.
            # Subtract the daily fixed cost from the daily returns
            n_days = (portf_ret.index[1:] - portf_ret.index[:-1]).to_numpy().astype('timedelta64[D]').astype(int)
            fixcost = (1 + fc) ** (n_days / n_days_per_year) - 1
            portf_ret[1:] -= fixcost

        return portf_ret
