"""Utilities for symbolic Jacobian computation.

Adapted from :mod:`chaste_codegen._jacobian` under the MIT licence.
"""

from sympy import Matrix, cse


def get_jacobian_matrix(state_vars, derivative_equations):
    """Return common subexpressions and the Jacobian matrix.

    Parameters
    ----------
    state_vars: Sequence[sympy.Symbol]
        Ordered state variables.
    derivative_equations: Sequence[sympy.Eq]
        Equations defining derivatives for each state variable.
    """
    assert all(
        eq.lhs == sv for eq, sv in zip(derivative_equations, state_vars)
    ), "Derivative equations must correspond to state variables"

    rhs = [eq.rhs for eq in derivative_equations]
    jac = Matrix(rhs).jacobian(Matrix(state_vars))
    cse_eqs, jac = cse(jac, order="none")
    return cse_eqs, Matrix(jac)