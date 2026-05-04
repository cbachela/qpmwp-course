############################################################################
### QPMwP CODING EXAMPLES - OPTIMIZATION 1 - USING LIBRARY CVXPY
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     26.01.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Under Terminal, click on New Terminal.
# In the Terminal window, select Command Prompt.
# Create and activate a virtual environment, and install required packages. For that, type the following commands:

# .venv\Scripts\activate

# uv pip install ipykernel
# uv pip install pandas
# uv pip install matplotlib
# uv pip install cvxpy




# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp






# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

# Load msci country index return series

path_to_data = '../data/'
# N = 24
N = 10
df = pd.read_csv(os.path.join(path_to_data, 'msci_country_indices.csv'),
                    index_col=0,
                    header=0,
                    parse_dates=True,
                    date_format='%d-%m-%Y')
series_id = df.columns[0:N]
return_series = df[series_id]

# Create 'level' series from return series
level_series = (1 + return_series).cumprod()

# # Alternatively, compute returns from level series
# returns = level_series.pct_change(1).dropna()

# Visualization
return_series.plot()

plt.figure(figsize=(10, 4))
# level_series.plot(alpha=1, legend=True)
np.log(level_series).plot(alpha=1, legend=True)
plt.grid()
plt.show()






# --------------------------------------------------------------------------
# Estimates of the expected returns and covariance matrix (using sample mean and covariance)
# --------------------------------------------------------------------------

scalefactor = 1  # could be set to 252 (trading days) for annualized returns


# Expected returns

##  This would be wrong:
##  mu = X.mean()

## This is correct:
mu = np.exp(np.log(1 + return_series).mean(axis=0) * scalefactor) - 1

# Covariance matrix
covmat = return_series.cov() * scalefactor

mu, covmat



# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


# We represent the portfolio domain with the form
# P = {x | Gx <= h, Ax = b, lb <= x <= ub}


# Lower and upper bounds
lb = np.zeros(covmat.shape[0])
# ub = np.repeat(0.2, N)
ub = np.repeat(1, N)

lb, ub


# Budget constraint
A = np.ones((1, N))
b = np.array(1.0)

A, b


# Linear inequality constraints
G = np.zeros((2, N))
G[0, 0:5] = 1
G[1, 5:10] = 1
h = np.array([0.5, 0.5])

G, h






# --------------------------------------------------------------------------
# Solve mean-variance optimal portfolios with cvxpy
# --------------------------------------------------------------------------

# Objective function parameters
risk_aversion = 1.0
q = mu.to_numpy() * -1
P = covmat.to_numpy() * risk_aversion

# Decision vector
x = cp.Variable(N, name='weights')

# Constraints
cons_list = [x >= lb, x <= ub]
if G is not None:
    cons_list.append(G @ x <= h)
if A is not None:
    cons_list.append(A @ x == b)

# Objective function
obj = q @ x + 0.5 * cp.quad_form(x, P)

# Finalize problem
model = cp.Problem(cp.Minimize(obj), cons_list)
model

# Solve the problem
model.solve(solver=cp.CVXOPT, verbose=False)

# Extract solution and objective
if model.status not in ["optimal", "optimal_inaccurate"]:
    raise ValueError(f"Optimization failed. Status: {model.status}")

x_opt = model.variables()[0].value
x_opt = pd.Series(x_opt, index=mu.index)
obj_val = model.value
status = model.status

print("Optimal weights:", x_opt)
print("Optimal objective value:", obj_val)
print("Solver status:", status)

x_opt.plot(kind='bar')


# Dual variables (Lagrange multipliers)
model.constraints[0].dual_value  # for first constraint
model.constraints[1].dual_value  # for second constraint
model.constraints[2].dual_value  # for third constraint







