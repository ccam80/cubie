"""Utilities for symbolic Jacobian computation.

Adapted from :mod:`chaste_codegen._jacobian` under the MIT licence.
"""
from typing import TYPE_CHECKING

from sympy import Matrix, cse, symbols
if TYPE_CHECKING:
    from cubie.systemmodels.symbolic.file_generation import GeneratedFile


JACOBIAN_TEMPLATE = (
    "\n\n\ndef {func_name}(constants, precision):\n"
    '    """Auto-generated Jacobian factory."""\n'
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision[:,:]),\n"
    "              device=True,\n"
    "             inline=True)\n"
    "    def jac_v(state, parameters, driver, dstate, Jv):\n"
    "    {body}\n"
    "    \n"
    "    return jac_v\n"
)

def generate_jacobian_function(code_lines,
                               file: "GeneratedFile"):
    return file.generate_import(code_lines,
                               "_build_symbolic_jacobian",
                               JACOBIAN_TEMPLATE)

def get_jacobian_matrix(state_vars, equations):
    """Return common subexpressions and the Jacobian matrix.

    Parameters
    ----------
    state_vars: Sequence[sympy.Symbol]
        Ordered state variables.
    equations: Sequence[sympy.Eq]
        Equations defining derivatives for each state variable as well as
        other intermediate expressions.
    """
    # Separate derivative equations (state variable derivatives) from intermediate expressions
    derivative_equations = []
    intermediate_equations = []

    for eq in equations:
        if eq.lhs in state_vars:
            derivative_equations.append(eq)
        else:
            intermediate_equations.append(eq)

    assert len(derivative_equations) == len(state_vars), (
        f"Expected {len(state_vars)} derivative equations, got {len(derivative_equations)}"
    )

    # Substitute intermediate expressions into derivative equations recursively
    substitutions = {eq.lhs: eq.rhs for eq in intermediate_equations}
    substituted_rhs = [
        eq.rhs.subs(substitutions, simultaneous=True) for eq in derivative_equations
    ]

    # Compute Jacobian with respect to state variables
    jac = Matrix(substituted_rhs).jacobian(Matrix(state_vars))
    cse_eqs, jac = cse(jac, order="none")
    return cse_eqs, Matrix(jac)

# def jac_v(state_vars, equations, v=None):
#     """Compute Jacobian-vector product symbolically.
#
#     This function uses the same logic as get_jacobian_matrix to compute the Jacobian,
#     then multiplies it by a vector v to produce the symbolic result.
#
#     Parameters
#     ----------
#     state_vars: Sequence[sympy.Symbol]
#         Ordered state variables.
#     equations: Sequence[sympy.Eq]
#         Equations defining derivatives for each state variable as well as
#         other intermediate expressions.
#     v: sympy.Matrix or None
#         Vector to multiply with Jacobian. If None, creates symbolic vector.
#
#     Returns
#     -------
#     tuple
#         (cse_eqs, jac_v_result) where cse_eqs are common subexpressions
#         and jac_v_result is the symbolic Jacobian-vector product.
#     """
#     # Separate derivative equations from intermediate expressions
#     derivative_equations = []
#     intermediate_equations = []
#
#     for eq in equations:
#         if eq.lhs in state_vars:
#             derivative_equations.append(eq)
#         else:
#             intermediate_equations.append(eq)
#
#     assert len(derivative_equations) == len(state_vars), (
#         f"Expected {len(state_vars)} derivative equations, got {len(derivative_equations)}"
#     )
#
#     # Substitute intermediate expressions into derivative equations recursively
#     substitutions = {eq.lhs: eq.rhs for eq in intermediate_equations}
#     substituted_rhs = [
#         eq.rhs.subs(substitutions, simultaneous=True) for eq in derivative_equations
#     ]
#
#     # Compute Jacobian with respect to state variables
#     jac = Matrix(substituted_rhs).jacobian(Matrix(state_vars))
#
#     # Create vector v if not provided
#     if v is None:
#         v = Matrix([symbols(f'v_{i}') for i in range(len(state_vars))])
#     elif not isinstance(v, Matrix):
#         v = Matrix(v)
#
#     # Compute Jacobian-vector product
#     jac_v_product = jac * v
#
#     # Apply common subexpression elimination to the result
#     cse_eqs, jac_v_result = cse(jac_v_product, order="none")
#
#     return cse_eqs, Matrix(jac_v_result)
