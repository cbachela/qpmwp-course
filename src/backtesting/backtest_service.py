############################################################################
### QPMwP - CLASS BacktestService
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from typing import Optional

# Third party imports
import numpy as np
import pandas as pd

# Local modules imports
from optimization.optimization import (
    Optimization,
    EmptyOptimization,
)
from optimization.optimization_data import OptimizationData
from optimization.constraints import Constraints
from backtesting.selection import Selection
from backtesting.strategy import Strategy
from backtesting.portfolio import floating_weights
from backtesting.backtest_item_builder_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)




class BacktestService():
    """
    A class to manage the backtesting process of a trading strategy.

    Attributes:
    ----------
    data : BacktestData
        The data required for backtesting.
    optimization_item_builders : dict[str, OptimizationItemBuilder]
        A dictionary of optimization item builders.
    selection_item_builders : Optional[dict[str, SelectionItemBuilder]]
        A dictionary of selection item builders.
    optimization : Optional[Optimization]
        The optimization object used for solving the optimization problem.
    settings : Optional[dict]
        A dictionary of settings for the backtesting process.

    Methods:
    -------
    prepare_rebalancing(rebalancing_date: str) -> None
        Prepares the rebalancing by building the selection and optimization
        for a given rebalancing date.

    build_selection(rebdate: str) -> None
        Builds the selection for a given rebalancing date.
    
    build_optimization(rebdate: str) -> None
        Builds the optimization for a given rebalancing date.
    """
    def __init__(self,
                 data: 'BacktestData',
                 optimization_item_builders: dict[str, OptimizationItemBuilder],
                 selection_item_builders: Optional[dict[str, SelectionItemBuilder]] = None,
                 optimization: Optional[Optimization] = None,
                 settings: Optional[dict] = None,
                 **kwargs) -> None:
        self.data = data
        self.optimization_item_builders = optimization_item_builders
        self.selection_item_builders = (
            selection_item_builders if selection_item_builders is not None else {}
        )
        self.optimization = EmptyOptimization() if optimization is None else optimization
        self.settings = settings if settings is not None else {}
        self.settings.update(kwargs)
        # Initialize the selection and optimization data
        self.selection = Selection()
        self.optimization_data = OptimizationData([])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value):
        if not isinstance(value, Selection):
            raise TypeError("Expected a Selection instance for 'selection'")
        self._selection = value

    @property
    def selection_item_builders(self):
        return self._selection_item_builders

    @selection_item_builders.setter
    def selection_item_builders(self, value):
        if not isinstance(value, dict) or not all(
            isinstance(v, SelectionItemBuilder) for v in value.values()
        ):
            raise TypeError(
                "Expected a dictionary containing SelectionItemBuilder instances "
                "for 'selection_item_builders'"
            )
        self._selection_item_builders = value

    @property
    def optimization(self):
        return self._optimization

    @optimization.setter
    def optimization(self, value):
        if not isinstance(value, Optimization):
            raise TypeError("Expected an Optimization instance for 'optimization'")
        self._optimization = value

    @property
    def optimization_item_builders(self):
        return self._optimization_item_builders

    @optimization_item_builders.setter
    def optimization_item_builders(self, value):
        if not isinstance(value, dict) or not all(
            isinstance(v, OptimizationItemBuilder) for v in value.values()
        ):
            raise TypeError(
                "Expected a dictionary containing OptimizationItemBuilder instances "
                "for 'optimization_item_builders'"
            )
        self._optimization_item_builders = value

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        if not isinstance(value, dict):
            raise TypeError("Expected a dictionary for 'settings'")
        self._settings = value

    def prepare_rebalancing(self, rebalancing_date: str, strategy: Optional[Strategy] = None) -> None:
        self.build_selection(rebdate = rebalancing_date)
        self.build_optimization(rebdate=rebalancing_date, strategy=strategy)
        return None

    def build_selection(self, rebdate: str) -> None:
        # Loop over the selection_item_builders (unless the dictionary is empty)
        if self.selection_item_builders:
            for key, item_builder in self.selection_item_builders.items():
                item_builder.arguments['item_name'] = key
                item_builder(self, rebdate)
        return None

    def build_optimization(self, rebdate: str, strategy: Optional[Strategy] = None) -> None:

        # Calculate the initial weights (i.e., the weights from the previous rebalancing
        # floated with market returns to the current date)
        if strategy is None:
            strategy = Strategy([])


        # Get the previous portfolio and the current selection
        previous_portfolio = strategy.get_previous_portfolio(rebalancing_date=rebdate)
        current_selection = self.selection.selected

        # Cut out return series of the ids in the union of 
        # the current selection and the previous portfolio
        if hasattr(self.data, 'get_return_series'):
            return_series = self.data.get_return_series(
                    ids=list(
                        set(current_selection)
                        .union(set(previous_portfolio.weights.keys()))
                    ),
                    end_date=rebdate,
                    fillna_value=0,
                )
        else:
            return_series = self.data['return_series']

        # Float the weights of the previous portfolio with the market returns
        # until the new rebalancing date
        x_init = previous_portfolio.initial_weights(
            selection=current_selection,
            return_series=return_series,
            end_date=rebdate,
            rescale=True,
        )

        # Add the initial weights to the optimization specifications
        self.optimization.params['x_init'] = x_init

        # Add the floated initial weights to the selection object
        self.selection.add_filtered(
            filter_name='initial_weights',
            value=pd.Series(x_init, name='floated values')
        )

        # Initialize the optimization constraints
        # unless the selection is empty because no selection_item_builder was called
        if self.selection_item_builders:
            self.optimization.constraints = Constraints(ids = self.selection.selected)

        # Loop over the optimization_item_builders
        for item_builder in self.optimization_item_builders.values():
            item_builder(self, rebdate)

        return None
