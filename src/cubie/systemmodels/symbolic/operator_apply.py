"""Code generation for the linear operator ``β·M·v − γ·h·J·v``.

The mass matrix ``M`` is provided at code-generation time either as a NumPy
array or a SymPy matrix. Its entries are embedded directly into the generated
device routine to avoid extra passes or buffers.
"""

from typing import Iterable, Tuple, Dict
import sympy as sp
from numba import cuda

from cubie.systemmodels.symbolic.parser import IndexedBases
from cubie.systemmodels.symbolic.numba_cuda_printer import print_cuda_multiple
from cubie.systemmodels.symbolic.jacobian import generate_analytical_jvp

OPERATOR_APPLY_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED LINEAR OPERATOR FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0):\n"
    '    """Auto-generated linear operator.\n'
    "    Computes out = beta * (M @ v) - gamma * h * (J @ v)\n"
    "    Returns device function:\n"
    "      operator_apply(state, parameters, drivers, h, v, out)\n"
    '    """\n'
    "    from numba import cuda\n"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def operator_apply(state, parameters, drivers, h, v, out):\n"
    "{body}\n"
    "    return operator_apply\n"
)


def _split_jvp_expressions(exprs: Iterable[Tuple[sp.Symbol, sp.Expr]]):
    """Split topologically-sorted (lhs, rhs) into auxiliaries and jvp terms."""
    aux = []
    jvp_terms: Dict[int, sp.Expr] = {}
    for lhs, rhs in exprs:
        lhs_str = str(lhs)
        if lhs_str.startswith("jvp["):
            idx = int(lhs_str.split("[")[1].split("]")[0])
            jvp_terms[idx] = rhs
        else:
            aux.append((lhs, rhs))
    return aux, jvp_terms

def _build_body_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: sp.Matrix,
) -> str:
    """Return code body computing ``β·M·v − γ·h·Jv``."""
    aux, jvp_terms = _split_jvp_expressions(jvp_exprs)

    n_out = len(index_map.dxdt.ref_map)
    n_in = len(index_map.states.index_map)
    v = sp.IndexedBase("v")
    beta_sym = sp.Symbol("beta")
    gamma_sym = sp.Symbol("gamma")
    h_sym = sp.Symbol("h")

    mass_assigns = []
    out_updates = []
    for i in range(n_out):
        mv = sp.S.Zero
        for j in range(n_in):
            entry = M[i, j]
            if entry == 0:
                continue
            sym = sp.Symbol(f"m_{i}{j}")
            mass_assigns.append((sym, entry))
            mv += sym * v[j]
        rhs = beta_sym * mv - gamma_sym * h_sym * jvp_terms[i]
        out_updates.append((sp.Symbol(f"out[{i}]"), rhs))

    exprs = mass_assigns + aux + out_updates
    lines = print_cuda_multiple(exprs, symbol_map=index_map.all_arrayrefs)
    if not lines:
        return "        pass"
    return "\n".join("        " + ln for ln in lines)


def generate_operator_apply_code_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M: sp.Matrix,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
) -> str:
    """Emit code for the operator apply factory using precomputed JVP expressions."""
    body = _build_body_from_jvp(jvp_exprs, index_map, M)
    return OPERATOR_APPLY_TEMPLATE.format(func_name=func_name, body=body)


def generate_operator_apply_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    M=None,
    func_name: str = "operator_apply_factory",
    cse: bool = True,
) -> str:
    """High-level entry: build JVP expressions, then emit operator apply code."""
    if M is None:
        n = len(index_map.states.index_map)
        M_mat = sp.eye(n)
    else:
        M_mat = sp.Matrix(M)
    jvp_exprs = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    return generate_operator_apply_code_from_jvp(
        jvp_exprs=jvp_exprs,
        index_map=index_map,
        M=M_mat,
        func_name=func_name,
        cse=cse,
    )


# ---------------------------------------------------------------------------
# Neumann preconditioner code generation
# ---------------------------------------------------------------------------

NEUMANN_TEMPLATE = (
    "\n"
    "# AUTO-GENERATED NEUMANN PRECONDITIONER FACTORY\n"
    "def {func_name}(constants, precision, beta=1.0, gamma=1.0, iterations=1):\n"
    '    """Auto-generated Neumann preconditioner.\n'
    "    Approximates (beta*M - gamma*h*J)^{-1} via a truncated\n"
    "    Neumann series. Returns device function:\n"
    "      preconditioner(state, parameters, drivers, h, v, out)\n"
    '    """\n'
    "    n = {n_out}\n"
    "    from numba import cuda\n"
    "    h_eff_factor = gamma / beta\n"
    "    @cuda.jit((precision[:],\n"
    "               precision[:],\n"
    "               precision[:],\n"
    "               precision,\n"
    "               precision[:],\n"
    "               precision[:]),\n"
    "              device=True,\n"
    "              inline=True)\n"
    "    def preconditioner(state, parameters, drivers, h, v, out):\n"
    "        tmp = cuda.local.array(n, precision)\n"
    "        for i in range(n):\n"
    "            out[i] = v[i]\n"
    "        h_eff = h * h_eff_factor\n"
    "        for _ in range(iterations):\n"
    "{jv_body}\n"
    "            for i in range(n):\n"
    "                out[i] = out[i] + h_eff * out[i] + tmp[i]\n"
    "        for i in range(n):\n"
    "            out[i] = beta * out[i]\n"
    "        return out\n"
    "    return preconditioner\n"
)


def _build_jv_eval_body(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    vec_name: str = "out",
    out_name: str = "tmp",
) -> str:
    """Return code evaluating ``J`` times ``vec_name`` into ``out_name``.

    Parameters
    ----------
    jvp_exprs : iterable of tuple
        Topologically sorted symbolic expressions representing the Jacobian
        vector product.
    index_map : IndexedBases
        Mapping of symbolic arrays to their CUDA representations.
    vec_name : str, optional
        Name of the input vector in the generated code, default ``"out"``.
    out_name : str, optional
        Name of the output vector in the generated code, default ``"tmp"``.

    Returns
    -------
    str
        CUDA code lines computing ``J * vec_name`` into ``out_name``.
    """
    symbol_map = dict(index_map.all_arrayrefs)
    symbol_map[sp.Symbol("v")] = vec_name
    symbol_map[sp.Symbol("jvp")] = out_name
    lines = print_cuda_multiple(jvp_exprs, symbol_map=symbol_map)
    if not lines:
        lines = ["pass"]
    return "\n".join("            " + ln for ln in lines)


def generate_neumann_preconditioner_code_from_jvp(
    jvp_exprs: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_factory",
    iterations: int = 1,
    cse: bool = True,
) -> str:
    """Generate code for a Neumann preconditioner.

    Parameters
    ----------
    jvp_exprs : iterable of tuple
        Precomputed Jacobian-vector product expressions.
    index_map : IndexedBases
        Mapping of symbolic arrays to CUDA references.
    func_name : str, optional
        Name of the emitted factory, default
        ``"neumann_preconditioner_factory"``.
    iterations : int, optional
        Number of Neumann iterations to inline, default ``1``.
    cse : bool, optional
        Unused placeholder for API symmetry, default ``True``.

    Returns
    -------
    str
        Source code for the factory function.
    """
    n_out = len(index_map.dxdt.ref_map)
    jv_body = _build_jv_eval_body(jvp_exprs, index_map)
    return NEUMANN_TEMPLATE.format(
        func_name=func_name, n_out=n_out, jv_body=jv_body
    )


def generate_neumann_preconditioner_code(
    equations: Iterable[Tuple[sp.Symbol, sp.Expr]],
    index_map: IndexedBases,
    func_name: str = "neumann_preconditioner_factory",
    iterations: int = 1,
    cse: bool = True,
) -> str:
    """High-level entry for Neumann preconditioner code generation.

    Parameters
    ----------
    equations : iterable of tuple
        Differential equations defining the system.
    index_map : IndexedBases
        Mapping of symbolic arrays to CUDA references.
    func_name : str, optional
        Name of the emitted factory, default
        ``"neumann_preconditioner_factory"``.
    iterations : int, optional
        Number of Neumann iterations to inline, default ``1``.
    cse : bool, optional
        Apply common-subexpression elimination, default ``True``.

    Returns
    -------
    str
        Source code for the factory function.
    """
    jvp_exprs = generate_analytical_jvp(
        equations,
        input_order=index_map.states.index_map,
        output_order=index_map.dxdt.index_map,
        observables=index_map.observable_symbols,
        cse=cse,
    )
    return generate_neumann_preconditioner_code_from_jvp(
        jvp_exprs=jvp_exprs,
        index_map=index_map,
        func_name=func_name,
        iterations=iterations,
        cse=cse,
    )



def residual_end_state_factory(
    base_state,                  # device array with the fixed base state for this solve
    dxdt,                        # device function
    mass_apply=None,             # optional device function: M(out) := M(in)
    beta=1.0,                    # matches β in your linear operator
    gamma=1.0,                   # matches γ in your linear operator
):
    USE_M = 1 if mass_apply is not None else 0

    @cuda.jit(device=True)
    def residual_function(state, parameters, drivers, h, linear_rhs, residual):
        """
        Residual: β·M·(state - base_state) - γ·h·f(eval_point, drivers) == 0
        Here eval_point == state (end-of-step unknown).
        - state: current guess for the end-of-step state (length n)
        - drivers: external inputs; passed through to dxdt
        - h: step size
        - linear_rhs: scratch buffer (length n), used for f(state)
        - residual: output mismatch (length n)
        """
        n = state.shape[0]

        # 1) rate at the evaluation point (the guess itself)
        dxdt(state, parameters, drivers, linear_rhs)  # linear_rhs := f(state)

        # 2) β·(state - base_state)
        for i in range(n):
            residual[i] = beta * (state[i] - base_state[i])

        # 3) apply mass if present: residual := β·M·(state - base)
        if USE_M:
            # Signature: mass_apply(state, parameters, drivers, in_vec, out_vec)
            mass_apply(state, parameters, drivers, residual, residual)

        # 4) subtract γ·h·f(state)
        for i in range(n):
            residual[i] -= gamma * h * linear_rhs[i]

    return residual_function



def stage_residual_factory(
    base_state,                  # device array with the fixed base for this stage
    a_ii,                        # scalar weight for how the increment contributes to the eval point
    dxdt,                        # device function
    mass_apply=None,             # optional device function: M(out) := M(in)
    beta=1.0,                    # matches β in your linear operator
    gamma=1.0,                   # matches γ in your linear operator
):
    USE_M = 1 if mass_apply is not None else 0

    @cuda.jit(device=True)
    def residual_function(stage, parameters, drivers, h, linear_rhs, residual):
        """
        Residual: β·M·stage - γ·h·f(eval_point, drivers) == 0
        eval_point := base_state + a_ii * stage
        - stage: current guess for the stage increment (length n)
        - drivers: external inputs; passed through to dxdt
        - h: step size
        - linear_rhs: scratch (length n); used for eval_point and M·stage
        - residual: output mismatch (length n); also holds f(eval_point) transiently
        """
        n = stage.shape[0]

        # 1) build eval_point in linear_rhs
        for i in range(n):
            linear_rhs[i] = base_state[i] + a_ii * stage[i]

        # 2) rate at eval_point into residual
        dxdt(linear_rhs, parameters, drivers, residual)  # residual := f(eval_point)

        # 3) linear_rhs := β·stage, then apply mass if present
        for i in range(n):
            linear_rhs[i] = beta * stage[i]
        if USE_M:
            # Signature: mass_apply(state_like, parameters, drivers, in_vec, out_vec)
            mass_apply(linear_rhs, parameters, drivers, linear_rhs, linear_rhs)

        # 4) residual := β·M·stage - γ·h·f(eval_point)
        for i in range(n):
            residual[i] = linear_rhs[i] - gamma * h * residual[i]

    return residual_function

