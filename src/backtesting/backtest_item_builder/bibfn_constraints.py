############################################################################
### QPMwP - BACKTEST ITEM BUILDER FUNCTIONS (bibfn) - CONSTRAINTS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     16.03.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Third party imports
import numpy as np
import pandas as pd






def bibfn_budget_constraint(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the budget constraint.
    '''

    # Arguments
    budget = kwargs.get('budget', 1)

    # Add constraint
    bs.optimization.constraints.add_budget(rhs=budget, sense='=')
    return None


def bibfn_box_constraints(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the box constraints.
    '''

    # Arguments
    lower = kwargs.get('lower', 0)
    upper = kwargs.get('upper', 1)
    box_type = kwargs.get('box_type', 'LongOnly')

    # Constraints
    bs.optimization.constraints.add_box(
        box_type=box_type,
        lower=lower,
        upper=upper
    )
    return None



def bibfn_size_dependent_upper_bounds(
    bs: 'BacktestService',
    rebdate: str,
    **kwargs
) -> None:

    '''
    Backtest item builder function for setting the upper bounds
    in dependence of a stock's market capitalization.
    '''

    # Arguments
    small_cap = kwargs.get('small_cap', {'threshold': 300_000_000, 'upper': 0.02})
    mid_cap = kwargs.get('mid_cap', {'threshold': 1_000_000_000, 'upper': 0.05})
    large_cap = kwargs.get('large_cap', {'threshold': 10_000_000_000, 'upper': 0.1})

    # Selection
    ids = bs.optimization.constraints.ids

    # Data: market capitalization
    mcap = bs.data.market_data['mktcap']
    # Get last available values for current rebdate
    mcap = mcap[mcap.index.get_level_values('date') <= rebdate].groupby(
        level = 'id'
    ).last()

    # Remove duplicates
    mcap = mcap[~mcap.index.duplicated(keep=False)]
    # Ensure that mcap contains all selected ids,
    # possibly extend mcap with zero values
    mcap = mcap.reindex(ids).fillna(0)

    # Generate the upper bounds
    upper = mcap * 0
    upper[mcap > small_cap['threshold']] = small_cap['upper']
    upper[mcap > mid_cap['threshold']] = mid_cap['upper']
    upper[mcap > large_cap['threshold']] = large_cap['upper']

    # Check if the upper bounds have already been set
    if not bs.optimization.constraints.box['upper'].empty:
        bs.optimization.constraints.add_box(
            box_type = 'LongOnly',
            upper = upper,
        )
    else:
        # Update the upper bounds by taking the minimum of the current and the new upper bounds
        bs.optimization.constraints.box['upper'] = np.minimum(
            bs.optimization.constraints.box['upper'],
            upper,
        )

    return None



def bibfn_bm_relative_upper_bounds(
    bs: 'BacktestService',
    rebdate: str,
    **kwargs
) -> None:

    '''
    Backtest item builder function for setting the upper bounds
    in dependence of a stock's market capitalization.
    '''

    # Arguments
    multiple = kwargs.get('multiple', 20)

    # Selection
    # ids = bs.selection.selected
    ids = bs.optimization.constraints.ids

    # Data: market capitalization
    mcap = bs.data.market_data['mktcap']
    mcap

    # Get mean market capitalization over the past year for each stock
    mcap = mcap[mcap.index.get_level_values('date') <= rebdate].tail(365).groupby(
        level='id'
    ).mean()

    # Remove duplicates
    mcap = mcap[~mcap.index.duplicated(keep=False)]
    # Ensure that mcap contains all selected ids,
    # possibly extend mcap with zero values
    mcap = mcap.reindex(ids).fillna(0)

    # Compute the upper bounds
    upper = (mcap / mcap.sum()) * multiple
    upper

    # Check if the upper bounds have already been set
    boxcon = bs.optimization.constraints.box
    if not boxcon['upper']:
        bs.optimization.constraints.add_box(
            box_type='LongOnly',
            upper=upper,
        )
    else:
        # Update the upper bounds by taking the minimum of the current and the new upper bounds
        bs.optimization.constraints.box['upper'] = np.minimum(
            bs.optimization.constraints.box['upper'],
            upper,
        )

    return None



def bibfn_turnover_constraint(bs, rebdate: str, **kwargs) -> None:
    """
    Function to assign a turnover constraint to the optimization.
    """
    if rebdate > bs.settings['rebdates'][0]:

        # Arguments
        turnover_limit = kwargs.get('turnover_limit')

        # Constraints
        bs.optimization.constraints.add_l1(
            name='turnover',
            rhs=turnover_limit,
            x0=bs.optimization.params['x_init'],
        )

    return None
