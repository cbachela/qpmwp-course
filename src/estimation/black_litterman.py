############################################################################
### QPMwP - BLACK LITTERMAN
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     28.04.2025
# First version:    28.04.2025
# --------------------------------------------------------------------------




# Standard library imports
from typing import Union

# Third party imports
import numpy as np
import pandas as pd






def bl_posterior_mean(
    mu_prior: pd.Series,
    P: Union[np.ndarray, pd.DataFrame],
    q: Union[np.ndarray, pd.Series],
    Psi: Union[np.ndarray, pd.DataFrame],
    Omega: Union[np.ndarray, pd.DataFrame],
) -> pd.Series:
    """
    Computes the posterior mean returns using the Black-Litterman model.

    Parameters:
    -----------
    mu_prior : pd.Series
        The prior mean returns.
    P : Union[np.ndarray, pd.DataFrame]
        The pick matrix representing the views.
    q : Union[np.ndarray, pd.Series]
        The expected returns for the views.
    Psi : Union[np.ndarray, pd.DataFrame]
        The uncertainty matrix for the prior.
    Omega : Union[np.ndarray, pd.DataFrame]
        The uncertainty matrix for the views.

    Returns:
    --------
    pd.Series
        The posterior mean returns as a pandas Series.
    """

    # Ensure all matrices have the same ordering
    # before converting them to numpy arrays
    ids = mu_prior.index
    if isinstance(mu_prior, pd.Series):
        mu_prior = mu_prior.to_numpy()
    if isinstance(P, pd.DataFrame):
        P = P[ids].to_numpy()
    if isinstance(q, pd.Series):
        q = q.to_numpy()
    if isinstance(Psi, pd.DataFrame):
        Psi = Psi.loc[ids, ids].to_numpy()
    if isinstance(Omega, pd.DataFrame):
        Omega = Omega.to_numpy()

     # Compute the posterior returns
    Psi_inv = np.linalg.inv(Psi)
    Omega_inv = np.linalg.inv(Omega)
    V = Psi_inv + P.T @ Omega_inv @ P
    V_inv = np.linalg.inv(V)
    posterior_returns = V_inv @ (
        Psi_inv @ mu_prior + P.T @ Omega_inv @ q
    )

    return pd.Series(posterior_returns, index=ids)


def view_from_scores_quintile(
    scores: pd.Series,
    mu_implied: pd.Series,
    scalefactor: int = 252,
) -> (pd.DataFrame, pd.Series):
    """
    Generate views based on quintile thresholds of scores.

    Parameters:
    -----------
    scores : pd.Series
        The scores used to determine long and short positions.
    mu_implied : pd.Series
        The implied mean returns.
    scalefactor : int, optional
        A scaling factor for the expected returns (default is 252).

    Returns:
    --------
    P : pd.DataFrame
        The pick matrix representing the views.
    q : pd.Series
        The expected returns for the views.
    """
    # Compute quintile thresholds
    lower_threshold, upper_threshold = np.percentile(scores, [20, 80])

    # Identify long and short positions
    s_short = scores[scores <= lower_threshold]
    s_long = scores[scores >= upper_threshold]

    # Create long-short weights
    w_ls = pd.Series(0.0, index=scores.index)
    if not s_short.empty:
        w_ls[s_short.index] = -1 / len(s_short)
    if not s_long.empty:
        w_ls[s_long.index] = 1 / len(s_long)

    # Create the pick matrix (P)
    P = w_ls.to_frame().T.reset_index(drop=True)

    # Compute view portfolio expected return (q) by a long-short
    # portfolio of the best versus worst implied returns
    mu_low, mu_high = np.percentile(mu_implied, [20, 80])
    mu_short = mu_implied[mu_implied <= mu_low]
    mu_long = mu_implied[mu_implied >= mu_high]
    q = pd.Series([mu_long.mean() - mu_short.mean()]) * scalefactor

    return P, q


def view_from_scores_absolute(
    scores: pd.Series,
    mu_implied: pd.Series,
    scalefactor: int = 252,
) -> (pd.DataFrame, pd.Series):
    """
    Generate views based on full ranking of scores.

    Parameters:
    -----------
    scores : pd.Series
        The scores used to determine the ranking.
    mu_implied : pd.Series
        The implied mean returns.
    scalefactor : int, optional
        A scaling factor for the expected returns (default is 252).

    Returns:
    --------
    P : pd.DataFrame
        The pick matrix representing the views.
    q : pd.Series
        The expected returns for the views.
    """
    # Rank the scores in descending order
    scores_rank = scores.rank(ascending=False).astype(int)

    # Create the pick matrix (P) as an identity matrix
    P = pd.DataFrame(
        np.eye(len(scores)),
        index=scores.index,
        columns=scores.index
    )

    # Align the implied returns with the rank of the scores
    sorted_mu = mu_implied.sort_values(ascending=False)
    q = pd.Series(
        sorted_mu.iloc[scores_rank-1].values,  # Align ranks with sorted returns
        index=mu_implied.index
    ) * scalefactor

    return P, q


def generate_views_from_scores(
    scores: pd.Series,
    mu_implied: pd.Series,
    method: str = 'quintile',
    scalefactor: int = 252,
) -> (pd.DataFrame, pd.Series):
    """
    Generate views based on scores using the specified method.

    Parameters:
    -----------
    scores : pd.Series
        The scores used to generate views.
    mu_implied : pd.Series
        The implied mean returns.
    method : str, optional
        The method to generate views ('quintile' or 'absolute').
        Default is 'quintile'.
    scalefactor : int, optional
        A scaling factor for the expected returns (default is 252).

    Returns:
    --------
    P : pd.DataFrame
        The pick matrix representing the views.
    q : pd.Series
        The expected returns for the views.
    """
    if method == 'quintile':
        return view_from_scores_quintile(scores, mu_implied, scalefactor)
    elif method == 'absolute':
        return view_from_scores_absolute(scores, mu_implied, scalefactor)
    else:
        raise ValueError("Invalid method. Use 'quintile' or 'absolute'.")
