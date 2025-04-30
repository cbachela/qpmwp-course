############################################################################
### QPMwP CODING EXAMPLES - BLACK LITTERMAN MODEL - COUNTRY INDEXES DATASETS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     17.04.2025
# First version:    17.04.2025
# --------------------------------------------------------------------------



# This script demonstrates the application of the Black-Litterman model using 
# msci country index data.




# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from helper_functions import load_data_msci
from estimation.covariance import Covariance
from estimation.black_litterman import bl_posterior_mean







# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

N = 24
data = load_data_msci(path = '../data/', n = N)





# --------------------------------------------------------------------------
# Generate Cap-Weights
# See: https://www.msci.com/documents/10199/178e6643-6ae6-47b9-82be-e1fc565ededb
# --------------------------------------------------------------------------

country_names = [
    'US',
    'JP',
    'GB',
    'CA',
    'FR',
]
cap_weights = pd.Series(
    data=[
        0.7199,   # US
        0.051,    # JP
        0.0378,   # GB
        0.031,    # CA
        0.0288,   # FR
    ],
    index=country_names,
)
# Normalize such that weights sum to one
cap_weights = cap_weights / cap_weights.sum()







# --------------------------------------------------------------------------
# Implied Expected Returns (Prior)
# --------------------------------------------------------------------------

# Step 1: Compute the covariance matrix for the selected country indices
covariance = Covariance(method='pearson')
return_series = data['return_series'][country_names]
covmat = covariance.estimate(return_series, inplace=False) * 252  # Annualize the covariance matrix
covmat

# Step 2. Calculate implied expected return from the cap-weights
mu_implied = covmat @ cap_weights
mu_implied.plot(kind='bar', figsize=(10, 5))

# Step 3: Assert that analytical solution to the unconstrained mean-variance optimization
# using the implied expected returns gives the same weights as the cap-weights
w_star = pd.Series(np.linalg.inv(covmat) @ mu_implied, index=country_names)

W = pd.DataFrame([cap_weights, w_star], index=["w_prior", "w_star"]).T
W.plot(kind="bar", figsize=(10, 5))
W
np.abs(W['w_star'] - W['w_prior']).max()


# Step 4: Show input sensitivity of the mean-variance optimization.
# For that, modify the implied expected returns and recompute the optimal weights.
mu_implied_mod = mu_implied.copy()
mu_implied_mod.iloc[0] = mu_implied_mod.iloc[0] * 0.8
mu_implied_mod = mu_implied_mod / mu_implied_mod.sum() * mu_implied.sum()

Mu = pd.concat({
    'mu_implied': mu_implied,
    'mu_implied_mod': mu_implied_mod,
}, axis=1)
Mu.plot(kind='bar', figsize=(10, 5))

# Optimal weights with modified expected returns
w_star_mod = pd.Series(np.linalg.inv(covmat) @ mu_implied_mod, index=country_names)
W = pd.DataFrame([cap_weights, w_star, w_star_mod], index=["w_prior", "w_star", "w_star_mod"]).T
W.plot(kind="bar", figsize=(10, 5))


Mu.corr()
W.corr()




# --------------------------------------------------------------------------
# Creating Views
# --------------------------------------------------------------------------

# Let's assume we have the following views:
# View 1: JP will have an expected return of 0.04 / 252 (vs. 0.000033 implied)
# View 2: The US will underperform the average of the other countries by 2%
# View 3: The US will outperform Japan by 1% (Notice that this somewhat contradicts Views 1 and 2)

P = pd.DataFrame(
    data=[
        [],                    # Absolute view on JP
        [],                    # Relative view on US vs. others
        [],                    # Relative view on US vs. JP
    ],
    index=['View1', 'View2', 'View3'],
    columns=country_names,
)

q = pd.Series(
    data=[],
    index=P.index,
)


P
q






# -------------------------
# Tuning parameters
tau_psi = 0.05
tau_omega = 0.05
# -------------------------

# Uncertainty of the prior
Psi = covmat * tau_psi

# Uncertainty of the views
Omega = pd.DataFrame(
    np.diag([tau_omega] * len(q)),
    index=q.index,
    columns=q.index
)
Omega


# # Alternatively:
# Omega = P @ covmat @ P.T * 100
# Psi = covmat * 0.01




# --------------------------------------------------------------------------
# Posterior Expected Returns
# --------------------------------------------------------------------------

# Compute the posterior expected return vector
mu_posterior = bl_posterior_mean(
    mu_prior=mu_implied,
    P=P,
    q=q,
    Psi=Psi,
    Omega=Omega,
)


Mu = pd.concat({
    "mu_prior": mu_implied,
    "mu_posterior": mu_posterior,
}, axis=1)

Mu
Mu.plot(kind="bar", figsize=(10, 5))
Mu.corr()





# --------------------------------------------------------------------------
# Portfolio Optimization
# --------------------------------------------------------------------------

# Analytical mean-variance optimization with the posterior returns
w_star_post = pd.Series(np.linalg.inv(covmat) @ mu_posterior, index=country_names)
W = pd.DataFrame([cap_weights, w_star_post], index=["w_prior", "w_posterior"]).T
W.plot(kind="bar", figsize=(10, 5))
W.corr()
W



# Re-run the mean-variance optimization, using a solver, with the posterior returns
from optimization.optimization import MeanVariance, Objective
from optimization.constraints import Constraints


constraints = Constraints(country_names)
constraints.add_budget()
constraints.add_box(lower=0, upper=0.3)

mv = MeanVariance(
    constraints=constraints,
    solver_name="cvxopt",
)
mv.objective = Objective(
    q=mu_posterior * (-1),
    P=covmat,
)
mv.solve()
w_posterior = pd.Series(mv.results["weights"])


W = pd.DataFrame([cap_weights, w_star_post, w_posterior],
                 index=["w_prior", "w_posterior_analytical", "w_posterior_numerical"]).T
W.plot(kind="bar", figsize=(10, 5))
W.sum()


data['return_series'][country_names].corr()   # Notice the correlation between US and CA





