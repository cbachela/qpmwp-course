############################################################################
### QPMwP CODING EXAMPLES - OPTIMIZATION 2
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# %reload_ext autoreload
# %autoreload 2




# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from helper_functions import load_data_msci
from estimation.covariance import Covariance
from estimation.expected_return import ExpectedReturn
from optimization.constraints import Constraints
from optimization.quadratic_program import QuadraticProgram
from optimization.optimization_data import OptimizationData
from optimization.optimization import MeanVariance







# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

N = 10
data = load_data_msci(path = '../data/', n = N) #discrete returns for the first 10 countries
data





# --------------------------------------------------------------------------
# Estimates of the expected returns and covariance matrix
# --------------------------------------------------------------------------

X = data['return_series']
scalefactor = 1  # could be set to 252 (trading days) for annualized returns


expected_return = ExpectedReturn(method='geometric', scalefactor=scalefactor) #estimate expected returns using geometric mean
expected_return.estimate(X=X, inplace=True)
# Or:
mu = expected_return.estimate(X=X, inplace=False)

covariance = Covariance(method='pearson')
covariance.estimate(X=X, inplace=True)
# Or:
Sigma = covariance.estimate(X=X, inplace=False)





# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------

# Instantiate the class
constraints = Constraints(ids = X.columns.tolist()) 

# Add budget constraint
constraints.add_budget(rhs=1, sense='=')

# Add box constraints (i.e., lower and upper bounds)
constraints.add_box(lower=0, upper=0.2) #box = weights

# Add linear constraints
G = pd.DataFrame(np.zeros((2, N)), columns=constraints.ids)
G.iloc[0, 0:5] = 1
G.iloc[1, 6:10] = 1
h = pd.Series([0.5, 0.5])
constraints.add_linear(G=G, sense='<=', rhs=h)


constraints.budget
constraints.box
constraints.linear




# --------------------------------------------------------------------------
# Solve mean-variance optimal portfolios - using class QuadraticProgram
# --------------------------------------------------------------------------


# Extract the constraints in the format required by the solver
GhAb = constraints.to_GhAb()
GhAb


risk_aversion = 3

qp = QuadraticProgram(
    #everything needs to be numpy arrays for qp
    P = covariance.matrix.to_numpy() * risk_aversion,
    q = expected_return.vector.to_numpy() * -1, #we minimise
    G = GhAb['G'],
    h = GhAb['h'],
    A = GhAb['A'],
    b = GhAb['b'],
    lb = constraints.box['lower'].to_numpy(),
    ub = constraints.box['upper'].to_numpy(),
    solver = 'cvxopt',
)

qp.problem_data

qp.is_feasible() #check whether we could find a solution

qp.solve()
qp.objective_value()

solution = qp.results.get('solution')
solution

solution.found
solution.primal_residual()
solution.dual_residual()
solution.duality_gap()[0]






# --------------------------------------------------------------------------
# Solve mean-variance optimal portfolios - using class MeanVariance
# --------------------------------------------------------------------------


mv = MeanVariance(
    covariance=covariance,
    expected_return=expected_return,
    constraints=constraints,
    risk_aversion=1,
    solver_name='cvxopt',
)

mv.params


# Create an OptimizationData object that contains an element `return_series` holding
# the last 256 observations (weekdays) of the return series
optimization_data = OptimizationData(return_series=X.tail(256))

# Set the objective function
mv.set_objective(optimization_data=optimization_data)
mv.objective.coefficients

# Solve the optimization problem
mv.solve()
mv.results

# Extract the optimal weights
weights_mv = pd.Series(mv.results['weights'], index=X.columns)
weights_mv








# --------------------------------------------------------------------------
# Solve for a tracking-error minimizing portfolio by least-squares
# Using class LeastSquares
# (Lecture 3)
# --------------------------------------------------------------------------

from optimization.optimization import LeastSquares


# Instantiate the optimization object
ls = LeastSquares(
    constraints=constraints,
    solver_name='cvxopt',
)

# Create an OptimizationData object that contains an element `return_series` holding
# the last 256 observations (weekdays) of the return series as well as the benchmark
# return series for the same period
y = data['bm_series']
optimization_data = OptimizationData(return_series=X.tail(256),
                                     bm_series=y,
                                     align=True)

# Set the objective and solve
ls.set_objective(optimization_data=optimization_data)
ls.solve()

weights_ls = pd.Series(ls.results['weights'], index=X.columns)
weights_ls






# --------------------------------------------------------------------------
# Simulations
# --------------------------------------------------------------------------


weights_mat = pd.concat({
    'mv': weights_mv,
    'ls': weights_ls
}, axis=1)


sim = X @ weights_mat
sim['benchmark'] = data['bm_series']
sim.dropna(how='all', inplace=True)

np.log((1 + sim).cumprod()).plot()
