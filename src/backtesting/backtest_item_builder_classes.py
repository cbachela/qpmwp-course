############################################################################
### QPMwP - BACKTEST ITEM BUILDER CLASSES
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from abc import ABC, abstractmethod
from typing import Any

# Third party imports
import numpy as np
import pandas as pd





class BacktestItemBuilder(ABC):
    '''
    Base class for building backtest items.
    This class should be inherited by specific item builders.
    
    # Arguments
    kwargs: Keyword arguments that fill the 'arguments' attribute.
    '''

    def __init__(self, **kwargs):
        self._arguments = {}
        self._arguments.update(kwargs)

    @property
    def arguments(self) -> dict[str, Any]:
        return self._arguments

    @arguments.setter
    def arguments(self, value: dict[str, Any]) -> None:
        self._arguments = value

    @abstractmethod
    def __call__(self, service, rebdate: str) -> None:
        raise NotImplementedError("Method '__call__' must be implemented in derived class.")



class SelectionItemBuilder(BacktestItemBuilder):
    '''
    Callable Class for building selection items in a backtest.
    '''

    def __call__(self, bs: 'BacktestService', rebdate: str) -> None:
        '''
        Build selection item from a custom function.

        :param bs: The backtest service.
        :param rebdate: The rebalance date.
        :raises ValueError: If 'bibfn' is not defined or not callable.
        '''

        selection_item_builder_fn = self.arguments.get('bibfn')
        if selection_item_builder_fn is None or not callable(selection_item_builder_fn):
            raise ValueError('bibfn is not defined or not callable.')

        item_value = selection_item_builder_fn(bs = bs, rebdate = rebdate, **self.arguments)
        item_name = self.arguments.get('item_name')

        # Add selection item
        bs.selection.add_filtered(filter_name = item_name, value = item_value)
        return None



class OptimizationItemBuilder(BacktestItemBuilder):
    '''
    Callable Class for building optimization data items in a backtest.
    '''

    def __call__(self, bs, rebdate: str) -> None:

        '''
        Build optimization data item from a custom function.

        :param bs: The backtest service.
        :param rebdate: The rebalance date.
        :raises ValueError: If 'bibfn' is not defined or not callable.
        '''

        optimization_item_builder_fn = self.arguments.get('bibfn')
        if optimization_item_builder_fn is None or not callable(optimization_item_builder_fn):
            raise ValueError('bibfn is not defined or not callable.')

        # Call the backtest item builder function. Notice that the function returns None,
        # it modifies the backtest service in place.
        optimization_item_builder_fn(bs = bs, rebdate = rebdate, **self.arguments)
        return None
    