############################################################################
### QPMwP - CLASS QuadraticProgram
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Standard library imports
from typing import Optional, Union

# Third party imports
import pandas as pd
import numpy as np
import qpsolvers
import scipy.sparse as spa




ALL_SOLVERS = {'clarabel', 'cvxopt', 'daqp', 'ecos', 'gurobi', 'highs', 'mosek', 'osqp', 'piqp', 'proxqp', 'qpalm', 'quadprog', 'scs'}
SPARSE_SOLVERS = {'clarabel', 'ecos', 'gurobi', 'mosek', 'highs', 'qpalm', 'osqp', 'qpswift', 'scs'}
IGNORED_SOLVERS = {
    'gurobi',  # Commercial solver
    'mosek',  # Commercial solver
    'ecos',
    'scs',
    'piqp',
    'proxqp',
    'clarabel'
} #those solvers could create some issues so we just ignore them and remove them from the usable solvers
USABLE_SOLVERS = ALL_SOLVERS - IGNORED_SOLVERS






# TODO:
# [ ] Add method to linearize absolute value functions (e.g. leverage constraint, turnover constraint)





class QuadraticProgram():

    def __init__(
        self,
        P: Union[np.ndarray, spa.csc_matrix],
        q: np.ndarray,
        G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
        b: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        **kwargs,
    ):
        self._results = {}
        self._solver_settings = {'solver': 'cvxopt', 'sparse': True}
        self._problem_data = {
            'P': P,
            'q': q,
            'G': G,
            'h': h,
            'A': A,
            'b': b,
            'lb': lb,
            'ub': ub,
        }
        # Update the solver_settings dictionary with the keyword arguments
        self.solver_settings.update(kwargs)
        if self.solver_settings['solver'] not in USABLE_SOLVERS:
            raise ValueError(
                f"Solver '{self.solver_settings['solver']}' is not available. "
                f'Choose from: {USABLE_SOLVERS}'
            )

    @property
    def solver_settings(self) -> dict:
        return self._solver_settings

    @property
    def problem_data(self) -> dict:
        return self._problem_data

    @property
    def results(self) -> dict:
        return self._results

    def update_problem_data(self, value: dict) -> None:
        '''
        Update the problem_data dict with the given value.

        Parameters:
        ----------
        value : dict
            The value to update the problem_data with.
        '''
        self._problem_data.update(value)

    def update_results(self, value: dict) -> None:
        '''
        Update the results dict with the given value.

        Parameters:
        ----------
        value : dict
            The value to update the results with.
        '''
        self._results.update(value)

    def solve(self) -> None:
        '''
        Solve the quadratic programming problem using the specified solver.

        This method sets up and solves the quadratic programming problem defined by the problem data.
        It supports various solvers and can convert the problem data to sparse matrices for better performance
        with certain solvers.

        The problem is defined as:
            minimize    (1/2) * x.T * P * x + q.T * x
            subject to  G * x <= h
                        A * x  = b
                        lb <= x <= ub

        The solution is stored in the results dictionary.

        Raises:
        -------
        ValueError:
            If the specified solver is not available.

        Notes:
        ------
        - The method converts the problem data to sparse matrices if the solver supports sparse matrices
        and the 'sparse' setting is enabled.
        - The method reshapes the vector 'b' if it has a single element and the solver is one of 'ecos', 'scs', or 'clarabel'.

        Examples:
        ---------
        >>> qp = QuadraticProgram(P, q, G, h, A, b, lb, ub, solver='cvxopt')
        >>> qp.solve()
        >>> solution = qp.results['solution']
        '''
        if self.solver_settings['solver'] in ['ecos', 'scs', 'clarabel']:
            if self.problem_data.get('b').size == 1:
                self.problem_data['b'] = np.array(self.problem_data['b']).reshape(-1)

        # P = self.get('P')
        # if P is not None and not isPD(P):
        #     self['P'] = nearestPD(P)

        # Create the problem
        problem = qpsolvers.Problem(
            P=self.problem_data.get('P'),
            q=self.problem_data.get('q'),
            G=self.problem_data.get('G'),
            h=self.problem_data.get('h'),
            A=self.problem_data.get('A'),
            b=self.problem_data.get('b'),
            lb=self.problem_data.get('lb'),
            ub=self.problem_data.get('ub')
        )

        # Convert to sparse matrices for best performance
        if self.solver_settings['solver'] in SPARSE_SOLVERS:
            if self.solver_settings['sparse']:
                if problem.P is not None:
                    problem.P = spa.csc_matrix(problem.P)
                if problem.A is not None:
                    problem.A = spa.csc_matrix(problem.A)
                if problem.G is not None:
                    problem.G = spa.csc_matrix(problem.G)

        # Solve the problem
        solution = qpsolvers.solve_problem(
            problem=problem,
            solver=self.solver_settings['solver'],
            initvals=self.solver_settings.get('x0'),
            verbose=False
        )
        self.update_results({'solution': solution})
        return None

    def is_feasible(self) -> bool:
        '''
        Check if the quadratic programming problem is feasible.

        This method sets up and solves a feasibility problem based on the current problem data.
        It creates a new QuadraticProgram instance with zero objective coefficients and the same
        constraints as the original problem. The feasibility problem is then solved to determine
        if there exists a solution that satisfies all the constraints.

        Returns:
        --------
        bool:
            True if the feasibility problem has a solution, indicating that the original problem
            is feasible. False otherwise.

        Notes:
        ------
        - The feasibility problem is defined with zero objective coefficients (P and q) to focus
        solely on the constraints.
        - The solution to the feasibility problem is stored in the results dictionary of the new
        QuadraticProgram instance.

        Examples:
        ---------
        >>> qp = QuadraticProgram(P, q, G, h, A, b, lb, ub, solver='cvxopt')
        >>> feasible = qp.is_feasible()
        >>> print(feasible)
        True
        '''
        qp = QuadraticProgram(
            P = np.zeros(self.problem_data['P'].shape),
            q = np.zeros(self.problem_data['q'].shape[0]),
            G = self.problem_data.get('G'),
            h = self.problem_data.get('h'),
            A = self.problem_data.get('A'),
            b = self.problem_data.get('b'),
            lb = self.problem_data.get('lb'),
            ub = self.problem_data.get('ub'),
        )
        qp.solve()
        return qp.results['solution'].found

    def objective_value(self,
                        x: Optional[np.ndarray] = None,
                        constant: Union[bool, float, int] = True) -> float:
        '''
        Calculate the objective value of the quadratic program.

        The objective value is calculated as:
        0.5 * x' * P * x + q' * x + const
        
        Parameters:
        x (Optional[np.ndarray]): The solution vector. If None, use the solution from results.
        constant (Union[bool, float, int]): If True, include the constant term from problem data.
                                            If a float or int, use that value as the constant term.
        
        Returns:
        float: The objective value.
        '''
        # 0.5 * x' * P * x + q' * x + const
        if x is None:
            x = self.results['solution'].x

        if isinstance(constant, bool):
            constant = (
                0 if self.problem_data.get('constant') is None
                else self.problem_data.get('constant').item()
            )
        elif not isinstance(constant, (float, int)):
            raise ValueError('constant must be a boolean, float, or int.')

        P = self.problem_data['P']
        q = self.problem_data['q']

        return (0.5 * (x @ P @ x) + q @ x).item() + constant
