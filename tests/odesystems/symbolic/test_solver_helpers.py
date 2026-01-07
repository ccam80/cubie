import numpy as np
import pytest
import sympy as sp
from numba import cuda, from_dtype

from cubie.odesystems.symbolic.codegen import (
    generate_cached_jvp_code,
    generate_cached_operator_apply_code,
    generate_neumann_preconditioner_cached_code,
    generate_neumann_preconditioner_code,
    generate_operator_apply_code,
    generate_prepare_jac_code,
    generate_stage_residual_code,
)
from cubie.odesystems.symbolic.parsing import JVPEquations
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system


@pytest.fixture(scope="session")
def operator_system(precision):
    """Build a linear system with a constant Jacobian."""

    dxdt = [
        "dx0 = a*x0 + b*x1",
        "dx1 = c*x0 + d*x1",
    ]
    constants = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    system = create_ODE_system(dxdt, states=["x0", "x1"],
                               constants=constants, precision=precision)
    return system


def _build_operator_factory(system, precision):
    def factory(beta, gamma, M):
        fname = (
            "operator_apply_factory_"
            f"{abs(hash((beta, gamma, M.tobytes())))}"
        )
        code = generate_operator_apply_code(
            system.equations,
            system.indices,
            M=M,
            func_name=fname,
        )
        op_fac = system.gen_file.import_function(fname, code)
        return op_fac(
            system.constants.values_dict,
            from_dtype(system.precision),
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="session")
def operator_factory(operator_system, precision):
    """Return a factory producing operator_apply device functions."""

    return _build_operator_factory(operator_system, precision)


@pytest.fixture(scope="session")
def operator_kernel(precision):
    """Kernel applying operator_apply to a vector."""

    n = 2

    def make_kernel(op):
        @cuda.jit
        def kernel(t, h, a_ij, vec, base_state, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            # base_state is provided by caller (can be empty placeholder)
            op(state, parameters, drivers, base_state, t, h, a_ij, vec, out)

        return kernel

    return make_kernel


@pytest.fixture(scope="session")
def cached_system():
    """Build a nonlinear system with state-dependent Jacobian."""

    dxdt = [
        "dx0 = a*x0*x1 + b*sin(x0)",
        "dx1 = c*x0*x1 + d*cos(x1)",
    ]
    constants = {"a": 0.5, "b": 1.3, "c": -0.7, "d": 0.9}
    system = create_ODE_system(dxdt, states=["x0", "x1"], constants=constants)
    return system


@pytest.fixture(scope="session")
def prepare_jac_factory(cached_system, precision):
    """Return a factory producing prepare_jac device functions."""

    def factory():
        fname = "prepare_jac_factory"
        code, aux_count = generate_prepare_jac_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        prep_fac = cached_system.gen_file.import_function(fname, code)
        prepare = prep_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
        )
        return prepare, aux_count

    return factory


@pytest.fixture(scope="session")
def cached_jvp_factory(cached_system, precision):
    """Return a factory producing calculate_cached_jvp device functions."""

    def factory():
        fname = "cached_jvp_factory"
        code = generate_cached_jvp_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        jvp_fac = cached_system.gen_file.import_function(fname, code)
        return jvp_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
        )

    return factory


@pytest.fixture(scope="session")
def cached_jvp_kernel(cached_system, precision):
    """Apply cached JVP outputs for comparison with analytic Jacobian."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, cached_jvp, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            vec,
            out_cached,
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, cached_aux)
            cached_jvp(
                state,
                parameters,
                drivers,
                cached_aux,
                t,
                vec,
                out_cached,
            )

        return kernel

    return make_kernel


@pytest.fixture(scope="session")
def cached_operator_factory(cached_system, precision):
    """Return a factory producing cached operator device functions."""

    def factory(beta, gamma, M):
        fname = (
            "cached_operator_factory_"
            f"{abs(hash((beta, gamma, M.tobytes())))}"
        )
        code = generate_cached_operator_apply_code(
            cached_system.equations,
            cached_system.indices,
            M=M,
            func_name=fname,
        )
        op_fac = cached_system.gen_file.import_function(fname, code)
        return op_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="session")
def cached_operator_kernel(cached_system, precision):
    """Kernel applying cached operator to a vector."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, op, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            h,
            a_ij,
            vec,
            base_state,
            out,
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, cached_aux)
            op(state, parameters, drivers, cached_aux, base_state, t, h, a_ij, vec, out)

        return kernel

    return make_kernel


def test_split_jvp_expressions_caches_high_cost_terms():
    """Cache the expression removing the largest runtime operation count."""

    x0, x1 = sp.symbols("x0 x1")
    dep0 = sp.Symbol("dep0")
    heavy = sp.Symbol("aux_heavy")
    simple = sp.Symbol("simple")
    j_00 = sp.Symbol("j_00")

    exprs = [
        (
            dep0,
            sp.sin(x0)
            + sp.cos(x1),
        ),
        (
            heavy,
            dep0**3
            + sp.exp(dep0)
            + sp.tan(dep0)
            + sp.log(dep0 + 2)
            + dep0 * sp.sinh(dep0),
        ),
        (simple, x0 + x1),
        (j_00, heavy + simple),
        (j_01 := sp.Symbol("j_01"), simple),
        (
            sp.Symbol("jvp[0]"),
            j_00 * sp.Symbol("v[0]")
            + j_01 * sp.Symbol("v[1]")
        ),
    ]

    equations = JVPEquations(exprs)
    cached_aux, runtime_aux, prepare_assigns = equations.cached_partition()
    selection = equations.cache_selection

    cached_symbols = [lhs for lhs, _ in cached_aux]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]
    prepare_symbols = [lhs for lhs, _ in prepare_assigns]

    assert cached_symbols == [j_00]
    assert list(selection.cached_leaf_order) == [j_00]
    assert heavy not in runtime_symbols
    assert heavy in prepare_symbols
    assert dep0 not in runtime_symbols
    assert dep0 in prepare_symbols
    assert equations.jvp_terms[0] == exprs[-1][1]


def test_split_jvp_expressions_limits_cache_size():
    """Limit cached expressions to twice the output dimension."""

    x = sp.symbols("x")
    heavy_symbols = [sp.Symbol(f"aux_heavy{i}") for i in range(3)]
    heavy_exprs = [
        (
            sp.sin(x)
            + sp.cos(x)
            + sp.exp(x)
            + sp.log(x + 2)
            + sp.tan(x)
            + sp.sinh(x)
        ),
        (
            sp.sin(2 * x)
            + sp.cos(2 * x)
            + sp.exp(2 * x)
            + sp.log(x + 3)
            + sp.tan(2 * x)
            + sp.sinh(2 * x)
            + x**2
        ),
        (
            sp.sin(3 * x)
            + sp.cos(3 * x)
            + sp.exp(3 * x)
            + sp.log(x + 4)
            + sp.tan(3 * x)
            + sp.sinh(3 * x)
            + x**3
            + sp.sqrt(x + 1)
        ),
    ]

    exprs = list(zip(heavy_symbols, heavy_exprs))
    j_00 = sp.Symbol("j_00")
    exprs.append((j_00, sum(heavy_symbols)))
    exprs.append((sp.Symbol("jvp[0]"), j_00 * sp.Symbol("v[0]")))

    equations = JVPEquations(exprs)
    cached_aux, runtime_aux, _ = equations.cached_partition()
    selection = equations.cache_selection

    cached_symbols = [lhs for lhs, _ in cached_aux]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]

    assert cached_symbols == [j_00]
    assert list(selection.cached_leaf_order) == [j_00]
    assert all(sym not in runtime_symbols for sym in heavy_symbols)


def test_split_jvp_expressions_groups_cse_dependents():
    """Cache dependents sharing a CSE prerequisite as a single group."""

    x0, x1 = sp.symbols("x0 x1")
    cse_sym = sp.Symbol("_cse0")
    aux_a = sp.Symbol("aux_a")
    aux_b = sp.Symbol("aux_b")
    jac = sp.Symbol("j_00")

    exprs = [
        (
            cse_sym,
            sp.sin(x0)
            + sp.cos(x1)
            + sp.exp(x0 + x1)
            + sp.log(x0 + 3),
        ),
        (
            aux_a,
            cse_sym**2
            + sp.exp(cse_sym)
            + sp.sin(cse_sym)
            + sp.tan(cse_sym)
            + sp.log(cse_sym + 2),
        ),
        (
            aux_b,
            cse_sym**3
            + sp.cos(cse_sym)
            + sp.sinh(cse_sym)
            + sp.acos(sp.tanh(x0))
            + sp.atan(cse_sym + 1),
        ),
        (jac, aux_a + aux_b),
        (sp.Symbol("jvp[0]"), jac * sp.Symbol("v[0]")),
    ]

    equations = JVPEquations(exprs, min_ops_threshold=5)
    cached_aux, runtime_aux, prepare_assigns = equations.cached_partition()
    selection = equations.cache_selection

    cached_symbols = [lhs for lhs, _ in cached_aux]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]
    prepare_symbols = [lhs for lhs, _ in prepare_assigns]

    assert cached_symbols == [jac]
    assert list(selection.cached_leaf_order) == [jac]
    assert cse_sym in runtime_symbols
    assert aux_a in prepare_symbols
    assert aux_b in prepare_symbols
    assert runtime_symbols == [cse_sym]
    assert equations.jvp_terms[0] == exprs[-1][1]


def test_split_jvp_expressions_limits_cse_depth_for_slots():
    """Restrict CSE traversal when grouping exceeds the cache budget."""

    x0, x1 = sp.symbols("x0 x1")
    cse_root = sp.Symbol("_cse0")
    cse_mid = sp.Symbol("_cse1")
    aux_a = sp.Symbol("aux_a")
    aux_b = sp.Symbol("aux_b")
    aux_c = sp.Symbol("aux_c")
    jac = sp.Symbol("j_00")

    exprs = [
        (
            cse_root,
            sp.sin(x0)
            + sp.cos(x1)
            + sp.exp(x0 + x1),
        ),
        (
            cse_mid,
            cse_root**2
            + sp.exp(cse_root)
            + sp.tan(cse_root),
        ),
        (
            aux_a,
            cse_mid**2
            + sp.sin(cse_mid)
            + sp.log(cse_mid + 2),
        ),
        (
            aux_b,
            cse_mid**3
            + sp.exp(cse_mid)
            + sp.sinh(cse_mid)
            + sp.atan(cse_mid + 1)
            + sp.sqrt(cse_mid + 3),
        ),
        (aux_c, x0 + x1),
        (jac, aux_a + aux_b + aux_c),
        (sp.Symbol("jvp[0]"), jac * sp.Symbol("v[0]")),
    ]

    equations = JVPEquations(
        exprs,
        max_cached_terms=1,
        min_ops_threshold=1,
    )
    cached_aux, runtime_aux, prepare_assigns = equations.cached_partition()
    selection = equations.cache_selection

    cached_symbols = [lhs for lhs, _ in cached_aux]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]
    prepare_symbols = {lhs for lhs, _ in prepare_assigns}

    assert cached_symbols == [jac]
    assert list(selection.cached_leaf_order) == [jac]
    assert aux_a not in runtime_symbols
    assert aux_b not in runtime_symbols
    assert aux_c not in runtime_symbols
    assert cse_mid in prepare_symbols
    assert cse_root in runtime_symbols


def test_cache_plan_shared_cse_with_slot_limit():
    """Ensure shared CSE branches remain available with cache limits."""

    x0, x1 = sp.symbols("x0 x1")
    cse_sym = sp.Symbol("_cse_shared")
    aux_a = sp.Symbol("aux_a")
    aux_b = sp.Symbol("aux_b")
    jac_a = sp.Symbol("j_00")
    jac_b = sp.Symbol("j_01")

    exprs = [
        (
            cse_sym,
            sp.sin(x0)
            + sp.cos(x1)
            + sp.exp(x0 + x1)
            + sp.log(x0 + 2),
        ),
        (
            aux_a,
            cse_sym**2
            + sp.sin(cse_sym)
            + sp.tan(cse_sym)
            + sp.log(cse_sym + 3),
        ),
        (
            aux_b,
            cse_sym**3
            + sp.cos(cse_sym)
            + sp.sinh(cse_sym)
            + sp.log(cse_sym + 4),
        ),
        (
            jac_a,
            aux_a + sp.exp(cse_sym) + sp.sin(aux_a),
        ),
        (
            jac_b,
            aux_b + sp.tanh(cse_sym) + sp.cos(aux_b),
        ),
        (sp.Symbol("jvp[0]"), jac_a * sp.Symbol("v[0]")),
        (sp.Symbol("jvp[1]"), jac_b * sp.Symbol("v[1]")),
    ]

    equations = JVPEquations(
        exprs,
        max_cached_terms=1,
        min_ops_threshold=1,
    )
    cached_aux, runtime_aux, prepare_assigns = equations.cached_partition()
    selection = equations.cache_selection

    assert len(selection.cached_leaf_order) == 1
    cached_leaf = selection.cached_leaf_order[0]
    runtime_symbols = [lhs for lhs, _ in runtime_aux]
    prepare_symbols = [lhs for lhs, _ in prepare_assigns]

    assert cse_sym in runtime_symbols
    assert cse_sym in prepare_symbols

    if cached_leaf == jac_a:
        assert jac_b in runtime_symbols
        assert aux_b in runtime_symbols
        assert aux_a in prepare_symbols
    else:
        assert jac_a in runtime_symbols
        assert aux_a in runtime_symbols
        assert aux_b in prepare_symbols

    remaining_leaf = jac_b if cached_leaf == jac_a else jac_a
    assert remaining_leaf in runtime_symbols


def test_build_expression_costs_tracks_jvp_dependencies():
    """Propagate JVP usage counts through dependency closures."""

    x0, x1 = sp.symbols("x0 x1")
    dep0 = sp.Symbol("dep0")
    heavy = sp.Symbol("aux_heavy")
    simple = sp.Symbol("simple")
    j_00 = sp.Symbol("j_00")

    non_jvp_order = [dep0, heavy, simple, j_00]
    non_jvp_exprs = {
        dep0: sp.sin(x0) + sp.cos(x1),
        heavy: dep0**2 + sp.exp(dep0),
        simple: x0 + x1,
        j_00: heavy + simple,
    }
    jvp_terms = {0: j_00 * sp.Symbol("v[0]")}

    exprs = [(sym, non_jvp_exprs[sym]) for sym in non_jvp_order]
    exprs.append((sp.Symbol("jvp[0]"), jvp_terms[0]))
    equations = JVPEquations(exprs)

    assert equations.jvp_usage == {j_00: 1}
    assert equations.jvp_closure_usage[j_00] == 1
    assert equations.jvp_closure_usage[heavy] == 1
    assert equations.jvp_closure_usage[dep0] == 1
    assert equations.jvp_closure_usage[simple] == 1


def test_equations_track_dependency_levels_and_costs():
    """Collect dependent levels and cumulative costs for auxiliaries."""

    x0, x1 = sp.symbols("x0 x1")
    seed = sp.Symbol("cse1")
    branch_a = sp.Symbol("cse7")
    branch_b = sp.Symbol("cse10")
    j_00 = sp.Symbol("j_00")
    j_20 = sp.Symbol("j_20")
    j_22 = sp.Symbol("j_22")
    j_02 = sp.Symbol("j_02")
    assignments = [
        (seed, x0 + x1),
        (branch_a, seed + x0),
        (branch_b, seed * x1),
        (j_00, branch_a + x0),
        (j_20, branch_a + x1),
        (j_22, branch_b + x0),
        (j_02, branch_b + x1),
        (sp.Symbol("jvp[0]"), j_00 * sp.Symbol("v[0]")),
        (sp.Symbol("jvp[1]"), j_20 * sp.Symbol("v[1]")),
        (sp.Symbol("jvp[2]"), j_22 * sp.Symbol("v[0]")),
        (sp.Symbol("jvp[3]"), j_02 * sp.Symbol("v[1]")),
    ]

    equations = JVPEquations(assignments)

    levels = equations.dependency_levels[seed]
    assert len(levels) == 2
    assert set(levels[0]) == {branch_a, branch_b}
    assert set(levels[1]) == {j_00, j_20, j_22, j_02}

    assert equations.order_index[seed] == 0
    assert equations.total_ops_cost[branch_a] == 2
    assert equations.total_ops_cost[j_00] == 3
    assert equations.total_ops_cost[sp.Symbol("jvp[0]")] == 4


@pytest.mark.parametrize(
    "beta,gamma,h,M",
    [
        (1.0, 1.0, 1.0, np.eye(2)),
        (1.0, 1.0, 1.0, np.diag([2.0, 3.0])),
        (0.5, 2.0, 1.0, np.array([[1.0, 0.5], [0.5, 2.0]])),
    ],
)
def test_operator_apply_dense(
    beta,
    gamma,
    h,
    M,
    operator_factory,
    operator_kernel,
    precision,
    tolerance,
):
    """Evaluate operator_apply for specific scalings and mass matrices."""

    op = operator_factory(beta, gamma, M)
    kernel = operator_kernel(op)
    v = np.array([1.0, -1.0], dtype=precision)
    out = np.zeros(2, dtype=precision)
    empty_base = np.empty(0, dtype=precision)
    kernel[1, 1](precision(0.0), precision(h), precision(1.0), v, empty_base, out)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    expected = beta * M @ v - gamma * h * J @ v
    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def test_operator_apply_constant_unpacking(operator_system):
    """Ensure constants are defined as individual variables."""
    code = generate_operator_apply_code(
        operator_system.equations, operator_system.indices
    )
    assert "a = precision(constants['a'])" in code

def test_cached_jvp_matches_jacobian(
    cached_system,
    prepare_jac_factory,
    cached_jvp_factory,
    cached_jvp_kernel,
    precision,
    tolerance,
):
    """Ensure cached JVP equals the analytic Jacobian-vector product."""

    prepare, aux_count = prepare_jac_factory()
    cached_jvp = cached_jvp_factory()
    kernel = cached_jvp_kernel(prepare, cached_jvp, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.8, -1.1], dtype=precision)
    vec = vec[:state_len]
    out_cached = np.zeros(state_len, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        vec,
        out_cached,
    )

    a = precision(cached_system.constants.values_dict["a"])
    b = precision(cached_system.constants.values_dict["b"])
    c = precision(cached_system.constants.values_dict["c"])
    d = precision(cached_system.constants.values_dict["d"])

    x0, x1 = state_values
    jacobian = np.array(
        [
            [a * x1 + b * np.cos(x0), a * x0],
            [c * x1, c * x0 - d * np.sin(x1)],
        ],
        dtype=precision,
    )
    expected = jacobian @ vec

    assert np.allclose(
        out_cached,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.mark.parametrize(
    "beta,gamma,h,M",
    [
        (1.0, 1.0, 0.25, np.eye(2)),
        (1.0, 1.0, 0.25, np.diag([1.2, 0.8])),
        (0.5, 1.7, 0.15, np.array([[1.0, 0.3], [0.4, 1.5]])),
    ],
)
def test_cached_operator_apply_dense(
    beta,
    gamma,
    h,
    M,
    cached_operator_factory,
    cached_operator_kernel,
    cached_system,
    prepare_jac_factory,
    precision,
    tolerance,
):
    """Evaluate cached operator using precomputed auxiliaries."""

    prepare, aux_count = prepare_jac_factory()
    op = cached_operator_factory(beta, gamma, M)
    kernel = cached_operator_kernel(prepare, op, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)

    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.8, -1.1], dtype=precision)
    vec = vec[:state_len]
    out = np.zeros(state_len, dtype=precision)

    empty_base = np.empty(0, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        precision(h),
        precision(1.0),
        vec,
        empty_base,
        out,
    )

    a = precision(cached_system.constants.values_dict["a"])
    b = precision(cached_system.constants.values_dict["b"])
    c = precision(cached_system.constants.values_dict["c"])
    d = precision(cached_system.constants.values_dict["d"])

    x0, x1 = state_values
    jacobian = np.array(
        [
            [a * x1 + b * np.cos(x0), a * x0],
            [c * x1, c * x0 - d * np.sin(x1)],
        ],
        dtype=precision,
    )
    beta_val = precision(beta)
    gamma_val = precision(gamma)
    h_val = precision(h)
    mass = np.array(M, dtype=precision)
    expected = (
        beta_val * mass @ vec - gamma_val * h_val * jacobian @ vec
    )

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


# ---------------------------------------------------------------------------
# Neumann preconditioner expression tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def neumann_factory(operator_system, precision):
    """Return a factory producing Neumann preconditioner device functions."""

    def factory(beta, gamma, order):
        fname = (f"neumann_preconditioner_factory_{int(beta)}_{int(gamma)}"
                 f"_{order}")
        code = generate_neumann_preconditioner_code(
            operator_system.equations,
            operator_system.indices,
            func_name=fname,
        )
        pre_fac = operator_system.gen_file.import_function(fname, code)
        return pre_fac(
            operator_system.constants.values_dict,
            from_dtype(operator_system.precision),
            beta=beta,
            gamma=gamma,
            order=order,
        )

    return factory


@pytest.fixture(scope="session")
def neumann_kernel(precision):
    """Apply the Neumann preconditioner to a vector, passing scratch."""

    n = 2

    def make_kernel(pre):
        @cuda.jit
        def kernel(t, h, a_ij, vec, base_state, out):
            state = cuda.local.array(n, precision)
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            scratch = cuda.local.array(n, precision)
            pre(state, parameters, drivers, base_state, t, h, a_ij, vec, out, scratch)

        return kernel

    return make_kernel


@pytest.fixture(scope="session")
def neumann_cached_factory(cached_system, precision):
    """Return a factory producing cached Neumann preconditioners."""

    def factory(beta, gamma, order):
        fname = (
            "neumann_cached_factory_"
            f"{int(beta * 10)}_{int(gamma * 10)}_{order}"
        )
        code = generate_neumann_preconditioner_cached_code(
            cached_system.equations,
            cached_system.indices,
            func_name=fname,
        )
        pre_fac = cached_system.gen_file.import_function(fname, code)
        return pre_fac(
            cached_system.constants.values_dict,
            from_dtype(cached_system.precision),
            beta=beta,
            gamma=gamma,
            order=order,
        )

    return factory


@pytest.fixture(scope="session")
def neumann_cached_kernel(cached_system, precision):
    """Apply cached Neumann preconditioner to a vector."""

    n_state = len(cached_system.indices.states.index_map)
    n_params = len(cached_system.indices.parameters.index_map)
    n_drivers = len(cached_system.indices.drivers.index_map)

    def make_kernel(prepare, pre, aux_count):
        aux_len = max(aux_count, 1)
        param_len = max(n_params, 1)
        driver_len = max(n_drivers, 1)

        @cuda.jit
        def kernel(
            state_values,
            parameter_values,
            driver_values,
            t,
            h,
            a_ij,
            vec,
            base_state,
            out,
        ):
            state = cuda.local.array(n_state, precision)
            parameters = cuda.local.array(param_len, precision)
            drivers = cuda.local.array(driver_len, precision)
            cached_aux = cuda.local.array(aux_len, precision)
            jvp = cuda.local.array(n_state, precision)

            for idx in range(n_state):
                state[idx] = state_values[idx]
            for idx in range(n_params):
                parameters[idx] = parameter_values[idx]
            for idx in range(n_drivers):
                drivers[idx] = driver_values[idx]

            prepare(state, parameters, drivers, t, cached_aux)
            pre(
                state,
                parameters,
                drivers,
                cached_aux,
                base_state,
                t,
                h,
                a_ij,
                vec,
                out,
                jvp,
            )

        return kernel

    return make_kernel

@pytest.mark.parametrize(
        "solver_settings_override",
        [{"precision": np.float64}],
        ids=[""],
        indirect=True,
)
@pytest.mark.parametrize(
    "beta,gamma,h,order",
    [
        (1.0, 1.0, 0.25, 0),
        (1.0, 1.0, 0.25, 1),
        (1.0, 1.0, 0.25, 2),
        (0.5, 2.0, 0.1, 3),
    ],
)
def test_neumann_preconditioner_expression(
    beta,
    gamma,
    h,
    order,
    neumann_factory,
    neumann_kernel,
    precision,
    tolerance,
):
    """Validate Neumann preconditioner against a truncated series.

    System: dx/dt = J x with J = [[a, b], [c, d]] = [[1, 2], [3, 4]].
    Preconditioner approximates (beta*I - gamma*h*J)^{-1} via truncated series.
    """
    pre = neumann_factory(beta, gamma, order)
    kernel = neumann_kernel(pre)

    v = np.array([0.7, -1.3], dtype=precision)
    out = np.zeros(2, dtype=precision)
    empty_base = np.empty(0, dtype=precision)

    kernel[1, 1](precision(0.0), precision(h), precision(1.0), v, empty_base, out)

    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    beta_inv = 1.0 / beta
    T = (gamma * beta_inv) * h * J

    # Truncated Neumann series: beta^{-1} sum_{k=0}^{order} (T^k) v
    expected = np.zeros_like(v)
    Tk_v = v.copy()
    expected += Tk_v
    for _ in range(order):
        Tk_v = T @ Tk_v
        expected += Tk_v
    expected = beta_inv * expected

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


@pytest.mark.parametrize(
    "beta,gamma,h,order",
    [
        (1.0, 1.0, 0.25, 0),
        (1.0, 1.0, 0.25, 1),
        (1.0, 1.0, 0.25, 2),
        (0.5, 1.5, 0.1, 3),
    ],
)
def test_neumann_preconditioner_cached_expression(
    beta,
    gamma,
    h,
    order,
    neumann_cached_factory,
    neumann_cached_kernel,
    cached_system,
    prepare_jac_factory,
    precision,
    tolerance,
    ):
    """Validate cached Neumann preconditioner with stored auxiliaries."""

    prepare, aux_count = prepare_jac_factory()
    pre = neumann_cached_factory(beta, gamma, order)
    kernel = neumann_cached_kernel(prepare, pre, aux_count)

    state_len = len(cached_system.indices.states.index_map)
    param_len = max(len(cached_system.indices.parameters.index_map), 1)
    drv_len = max(len(cached_system.indices.drivers.index_map), 1)

    state_values = np.array([0.4, -0.6], dtype=precision)
    state_values = state_values[:state_len]
    parameter_values = np.zeros(param_len, dtype=precision)
    driver_values = np.zeros(drv_len, dtype=precision)
    vec = np.array([0.7, -1.3], dtype=precision)
    vec = vec[:state_len]
    out = np.zeros(state_len, dtype=precision)

    empty_base = np.empty(0, dtype=precision)

    kernel[1, 1](
        state_values,
        parameter_values,
        driver_values,
        precision(0.0),
        precision(h),
        precision(1.0),
        vec,
        empty_base,
        out,
    )

    a = precision(cached_system.constants.values_dict["a"])
    b = precision(cached_system.constants.values_dict["b"])
    c = precision(cached_system.constants.values_dict["c"])
    d = precision(cached_system.constants.values_dict["d"])

    x0, x1 = state_values
    jacobian = np.array(
        [
            [a * x1 + b * np.cos(x0), a * x0],
            [c * x1, c * x0 - d * np.sin(x1)],
        ],
        dtype=precision,
    )
    beta_val = precision(beta)
    gamma_val = precision(gamma)
    beta_inv = precision(1.0) / beta_val
    h_val = precision(h)
    T = (gamma_val * beta_inv) * h_val * jacobian

    expected = np.zeros(state_len, dtype=precision)
    Tk_v = vec.copy()
    expected += Tk_v
    for _ in range(order):
        Tk_v = T @ Tk_v
        expected += Tk_v
    expected = beta_inv * expected

    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_loose * 50,
        rtol=tolerance.rel_loose * 50,
    )


@pytest.fixture(scope="session")
def residual_system():
    """Linear system with constant Jacobian for residual tests."""

    dxdt = [
        "dx0 = a*x0 + b*x1",
        "dx1 = c*x0 + d*x1",
    ]
    constants = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    system = create_ODE_system(dxdt, states=["x0", "x1"], constants=constants)
    system.build()
    return system


@pytest.fixture(scope="session")
def stage_residual_factory(residual_system, precision):
    def factory(beta, gamma, a_ii, M):
        base = cuda.to_device(np.array([0.25, -0.25], dtype=precision))
        fname = f"stage_residual_factory_{abs(hash(M.tobytes()))}"
        code = generate_stage_residual_code(
            residual_system.equations,
            residual_system.indices,
            M=M,
            func_name=fname,
        )
        res_fac = residual_system.gen_file.import_function(fname, code)
        return res_fac(
            residual_system.constants.values_dict,
            from_dtype(residual_system.precision),
            beta=beta,
            gamma=gamma,
        )

    return factory


@pytest.fixture(scope="session")
def residual_kernel(precision):
    n = 2

    def make_kernel(residual):
        @cuda.jit
        def kernel(t, h, aij, vec, base_state, out):
            parameters = cuda.local.array(1, precision)
            drivers = cuda.local.array(1, precision)
            residual(vec, parameters, drivers, t, h, aij, base_state, out)

        return kernel

    return make_kernel


@pytest.mark.parametrize(
    "beta,gamma,h,a_ii,M",
    [
        (1.0, 1.0, 1.0, 1.0, np.eye(2)),
        (1.0, 1.0, 1.0, 0.5, np.diag([2.0, 3.0])),
        (0.5, 2.0, 1.0, 0.25, np.array([[1.0, 0.5], [0.5, 2.0]])),
    ],
)
def test_stage_residual(
    beta,
    gamma,
    h,
    a_ii,
    M,
    stage_residual_factory,
    residual_kernel,
    precision,
    tolerance,
):
    residual = stage_residual_factory(beta, gamma, a_ii, M)
    kernel = residual_kernel(residual)
    stage = np.array([0.5, -0.3], dtype=precision)
    base = np.array([0.25, -0.25], dtype=precision)
    out = np.zeros(2, dtype=precision)
    kernel[1, 1](precision(0.0), precision(h), precision(a_ii), stage, base, out)
    J = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
    eval_point = base + a_ii * stage
    expected = beta * (M @ stage) - gamma * h * (J @ eval_point)
    assert np.allclose(
        out,
        expected,
        atol=tolerance.abs_tight,
        rtol=tolerance.rel_tight,
    )


def test_user_beta_gamma_variables():
    """Validate that users can define beta and gamma as state variables.
    
    This test addresses issue #373 by confirming that the internal code
    generation variables (now prefixed with _cubie_codegen_) do not
    conflict with user-defined variables named beta and gamma.
    """
    from cubie import solve_ivp, SolveResult
    
    dxdt = [
        "dbeta = -alpha * beta + gamma",
        "dgamma = alpha * beta - gamma",
    ]
    
    system = create_ODE_system(
        dxdt,
        states=["beta", "gamma"],
        parameters=["alpha"],
        precision=np.float64,
    )
    
    initial_conditions = np.array([[1.0, 0.5]], dtype=np.float64)
    parameter_values = np.array([[0.5]], dtype=np.float64)
    
    result = solve_ivp(
        system,
        initial_conditions,
        parameter_values,
        max_steps=100,
    )
    
    assert isinstance(result, SolveResult), "Result should be a SolveResult"
    assert result.status_codes is not None, "Status codes should be present"
    assert np.all(result.status_codes == 0), (
        f"Solver failed with beta/gamma as state variables: "
        f"status_codes={result.status_codes}"
    )
    assert result.time_domain_array is not None, "Time domain array should be present"
    assert len(result.time_domain_array) > 0, "Time domain array should not be empty"
