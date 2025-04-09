############################################################################
### QPMwP - CLASS Backtest
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from typing import Optional
import pickle
import os

# Third party imports
import numpy as np
import pandas as pd

# Local modules imports
from backtesting.portfolio import Portfolio
from backtesting.strategy import Strategy
from backtesting.backtest_service import BacktestService





class Backtest:

    """
    A class to perform backtesting of a trading strategy.

    Attributes:
    ----------
    strategy : Strategy
        Contains the list of Portfolio objects constructed during the backtesting,
        i.e, one for each rebalancing date.
    """
    def __init__(self):
        self._strategy = Strategy([])

    @property
    def strategy(self):
        return self._strategy

    def run(self, bs: BacktestService) -> None:
        """
        Executes the backtest by iterating over rebalancing dates,
        preparing and solving the optimization problem,
        and appending the resulting portfolio to the strategy object.

        Parameters:
        ----------
        bs : BacktestService
            The backtest service object containing settings and optimization data.
        """

        for rebalancing_date in bs.settings['rebdates']:

            if not bs.settings.get('quiet'):
                print(f'Rebalancing date: {rebalancing_date}')

            # Prepare the rebalancing, i.e., the optimization problem
            bs.prepare_rebalancing(rebalancing_date=rebalancing_date,
                                   strategy=self.strategy)

            # Solve the optimization problem
            try:
                bs.optimization.set_objective(optimization_data=bs.optimization_data)
                bs.optimization.solve()
            except Exception as error:
                raise RuntimeError(error)

            # Extract the portfolio weights from the optimization results,
            # create a Portfolio object and append it to the strategy
            weights = bs.optimization.results['weights']
            portfolio = Portfolio(rebalancing_date=rebalancing_date,
                                  weights=weights)
            self.strategy.portfolios.append(portfolio)

        return None

    def save(self,
             filename: str,
             path: Optional[str] = None) -> None:
        try:
            if path is not None and filename is not None:
                filename = os.path.join(path, filename)   #// alternatively, use pathlib package
            with open(filename, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object:", ex)

        return None
