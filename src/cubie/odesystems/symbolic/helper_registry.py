"""Registry of solver-helper source generators and binding contracts.

Each concrete :class:`~cubie.odesystems.solver_helpers.SolverHelperKind`
maps to one :class:`_RegistryEntry` describing how its source is
generated, which source dependencies the generator consumes, the exact
factory-binding arguments the generated factory accepts (declared,
never introspected), any metadata returned with the built member, and
an optional per-request validation hook.

Each concrete request produces two identities through the canonical
serializer:

- :func:`helper_source_hash` identifies the generated factory source.
  It contains only inputs that change the emitted source: the helper
  kind, the ODE equation/layout identity, the mass matrix for
  generators that bake it into source, and the canonical stage
  specification for stage-aware generators.
- :func:`helper_member_hash` identifies one bound helper product: the
  source identity plus the normalized factory arguments the entry
  declares.

One generated factory can legitimately bind multiple
beta/gamma/order/constant sets: different bindings that share source
reuse the generated factory and produce distinct members. Neither
identity uses Python function or closure identity.

See Also
--------
:mod:`cubie.odesystems.solver_helpers`
    Request and cache containers consumed alongside this registry.
:mod:`cubie.odesystems.symbolic.codegen`
    Source generators the registry entries dispatch to.
"""

from typing import Callable, Optional, Tuple

from attrs import field, frozen

from cubie._serialize import canonical_digest
from cubie.odesystems.solver_helpers import (
    STAGE_AWARE_KINDS,
    SolverHelperKind,
    SolverHelperRequest,
)
from cubie.odesystems.symbolic.codegen import (
    generate_cached_jvp_code,
    generate_cached_operator_apply_code,
    generate_jacobi_preconditioner_cached_code,
    generate_jacobi_preconditioner_code,
    generate_n_stage_jacobi_preconditioner_code,
    generate_n_stage_linear_operator_code,
    generate_n_stage_neumann_preconditioner_code,
    generate_n_stage_residual_code,
    generate_neumann_preconditioner_cached_code,
    generate_neumann_preconditioner_code,
    generate_operator_apply_code,
    generate_prepare_jac_code,
    generate_stage_residual_code,
)
from cubie.odesystems.symbolic.codegen.time_derivative import (
    generate_time_derivative_fac_code,
)
from cubie.odesystems.symbolic.codegen.neumann_convergence import (
    check_neumann_convergence,
)

__all__ = [
    "SOLVER_HELPER_REGISTRY",
    "helper_source_hash",
    "helper_member_hash",
]


def _neumann_validation(system, request: SolverHelperRequest) -> None:
    """Run the Neumann convergence diagnostic for a request.

    Runs on every request — including cache hits — so the warning
    surfaces for reused code as well as freshly generated code.
    """
    check_neumann_convergence(
        system.indices,
        evaluator=system._get_neumann_evaluator(),
        stage_coefficients=request.stage_coefficients,
        beta=request.beta,
        gamma=request.gamma,
    )


_SCALAR_ARGS = ("constants", "precision", "lineinfo")
_SCALED_ARGS = ("constants", "precision", "beta", "gamma", "lineinfo")
_ORDERED_ARGS = (
    "constants",
    "precision",
    "beta",
    "gamma",
    "order",
    "lineinfo",
)


@frozen
class _RegistryEntry:
    """One concrete helper kind's generation and binding contract.

    Attributes
    ----------
    kind
        The helper kind this entry serves.
    generate
        Callable ``(system, request, func_name)`` returning generated
        source, or ``(source, aux_count)`` when ``returns_aux_count``.
    factory_args
        Exact names of the factory-binding arguments the generated
        factory accepts. Declared, never introspected.
    uses_mass
        Whether the generator bakes the mass matrix into source.
    stage_aware
        Whether the generator consumes the stage specification.
    returns_aux_count
        Whether generation returns ``(source, aux_count)`` and the
        imported factory carries an ``aux_count`` attribute.
    validation_hook
        Optional diagnostic run on every request of this kind.
    """

    kind: SolverHelperKind
    generate: Callable = field(eq=False)
    factory_args: Tuple[str, ...] = _SCALED_ARGS
    uses_mass: bool = False
    stage_aware: bool = False
    returns_aux_count: bool = False
    validation_hook: Optional[Callable] = field(default=None, eq=False)


def _gen_linear_operator(system, request, func_name):
    return generate_operator_apply_code(
        system.equations,
        system.indices,
        M=system.compile_settings.mass,
        func_name=func_name,
        jvp_equations=system._get_jvp_exprs(),
    )


def _gen_linear_operator_cached(system, request, func_name):
    return generate_cached_operator_apply_code(
        system.equations,
        system.indices,
        M=system.compile_settings.mass,
        func_name=func_name,
        jvp_equations=system._get_jvp_exprs(),
    )


def _gen_prepare_jac(system, request, func_name):
    return generate_prepare_jac_code(
        system.equations,
        system.indices,
        func_name=func_name,
        jvp_equations=system._get_jvp_exprs(),
    )


def _gen_cached_jvp(system, request, func_name):
    return generate_cached_jvp_code(
        system.equations,
        system.indices,
        func_name=func_name,
        jvp_equations=system._get_jvp_exprs(),
    )


def _gen_neumann(system, request, func_name):
    return generate_neumann_preconditioner_code(
        system.equations,
        system.indices,
        func_name,
        jvp_equations=system._get_jvp_exprs(),
    )


def _gen_neumann_cached(system, request, func_name):
    return generate_neumann_preconditioner_cached_code(
        system.equations,
        system.indices,
        func_name,
        jvp_equations=system._get_jvp_exprs(),
    )


def _gen_jacobi(system, request, func_name):
    return generate_jacobi_preconditioner_code(
        system.equations,
        system.indices,
        func_name,
        M=system.compile_settings.mass,
    )


def _gen_jacobi_cached(system, request, func_name):
    return generate_jacobi_preconditioner_cached_code(
        system.equations,
        system.indices,
        func_name,
        M=system.compile_settings.mass,
    )


def _gen_stage_residual(system, request, func_name):
    return generate_stage_residual_code(
        system.equations,
        system.indices,
        M=system.compile_settings.mass,
        func_name=func_name,
    )


def _gen_time_derivative(system, request, func_name):
    return generate_time_derivative_fac_code(
        system.equations,
        system.indices,
        func_name=func_name,
    )


def _gen_n_stage_residual(system, request, func_name):
    return generate_n_stage_residual_code(
        equations=system.equations,
        index_map=system.indices,
        stage_coefficients=request.stage_coefficients,
        stage_nodes=request.stage_nodes,
        M=system.compile_settings.mass,
        func_name=func_name,
    )


def _gen_n_stage_linear_operator(system, request, func_name):
    return generate_n_stage_linear_operator_code(
        equations=system.equations,
        index_map=system.indices,
        stage_coefficients=request.stage_coefficients,
        stage_nodes=request.stage_nodes,
        M=system.compile_settings.mass,
        func_name=func_name,
        jvp_equations=system._get_jvp_exprs(),
    )


def _gen_n_stage_neumann(system, request, func_name):
    return generate_n_stage_neumann_preconditioner_code(
        equations=system.equations,
        index_map=system.indices,
        stage_coefficients=request.stage_coefficients,
        stage_nodes=request.stage_nodes,
        func_name=func_name,
        jvp_equations=system._get_jvp_exprs(),
    )


def _gen_n_stage_jacobi(system, request, func_name):
    return generate_n_stage_jacobi_preconditioner_code(
        equations=system.equations,
        index_map=system.indices,
        stage_coefficients=request.stage_coefficients,
        stage_nodes=request.stage_nodes,
        func_name=func_name,
        M=system.compile_settings.mass,
    )


SOLVER_HELPER_REGISTRY = {
    SolverHelperKind.LINEAR_OPERATOR: _RegistryEntry(
        kind=SolverHelperKind.LINEAR_OPERATOR,
        generate=_gen_linear_operator,
        uses_mass=True,
    ),
    SolverHelperKind.LINEAR_OPERATOR_CACHED: _RegistryEntry(
        kind=SolverHelperKind.LINEAR_OPERATOR_CACHED,
        generate=_gen_linear_operator_cached,
        uses_mass=True,
    ),
    SolverHelperKind.NEUMANN_PRECONDITIONER: _RegistryEntry(
        kind=SolverHelperKind.NEUMANN_PRECONDITIONER,
        generate=_gen_neumann,
        factory_args=_ORDERED_ARGS,
        validation_hook=_neumann_validation,
    ),
    SolverHelperKind.NEUMANN_PRECONDITIONER_CACHED: _RegistryEntry(
        kind=SolverHelperKind.NEUMANN_PRECONDITIONER_CACHED,
        generate=_gen_neumann_cached,
        factory_args=_ORDERED_ARGS,
        validation_hook=_neumann_validation,
    ),
    SolverHelperKind.JACOBI_PRECONDITIONER: _RegistryEntry(
        kind=SolverHelperKind.JACOBI_PRECONDITIONER,
        generate=_gen_jacobi,
        factory_args=_ORDERED_ARGS,
        uses_mass=True,
    ),
    SolverHelperKind.JACOBI_PRECONDITIONER_CACHED: _RegistryEntry(
        kind=SolverHelperKind.JACOBI_PRECONDITIONER_CACHED,
        generate=_gen_jacobi_cached,
        factory_args=_ORDERED_ARGS,
        uses_mass=True,
    ),
    SolverHelperKind.STAGE_RESIDUAL: _RegistryEntry(
        kind=SolverHelperKind.STAGE_RESIDUAL,
        generate=_gen_stage_residual,
        uses_mass=True,
    ),
    SolverHelperKind.N_STAGE_RESIDUAL: _RegistryEntry(
        kind=SolverHelperKind.N_STAGE_RESIDUAL,
        generate=_gen_n_stage_residual,
        uses_mass=True,
        stage_aware=True,
    ),
    SolverHelperKind.N_STAGE_LINEAR_OPERATOR: _RegistryEntry(
        kind=SolverHelperKind.N_STAGE_LINEAR_OPERATOR,
        generate=_gen_n_stage_linear_operator,
        uses_mass=True,
        stage_aware=True,
    ),
    SolverHelperKind.N_STAGE_NEUMANN_PRECONDITIONER: _RegistryEntry(
        kind=SolverHelperKind.N_STAGE_NEUMANN_PRECONDITIONER,
        generate=_gen_n_stage_neumann,
        factory_args=_ORDERED_ARGS,
        stage_aware=True,
        validation_hook=_neumann_validation,
    ),
    SolverHelperKind.N_STAGE_JACOBI_PRECONDITIONER: _RegistryEntry(
        kind=SolverHelperKind.N_STAGE_JACOBI_PRECONDITIONER,
        generate=_gen_n_stage_jacobi,
        factory_args=_ORDERED_ARGS,
        uses_mass=True,
        stage_aware=True,
    ),
    SolverHelperKind.PREPARE_JAC: _RegistryEntry(
        kind=SolverHelperKind.PREPARE_JAC,
        generate=_gen_prepare_jac,
        factory_args=_SCALAR_ARGS,
        returns_aux_count=True,
    ),
    SolverHelperKind.CALCULATE_CACHED_JVP: _RegistryEntry(
        kind=SolverHelperKind.CALCULATE_CACHED_JVP,
        generate=_gen_cached_jvp,
        factory_args=_SCALAR_ARGS,
    ),
    SolverHelperKind.TIME_DERIVATIVE_RHS: _RegistryEntry(
        kind=SolverHelperKind.TIME_DERIVATIVE_RHS,
        generate=_gen_time_derivative,
        factory_args=_SCALAR_ARGS,
    ),
}
"""Registry mapping each concrete kind to its generation contract."""

# The request container gates stage data by kind; the registry's
# per-entry flags must agree with it.
assert {
    entry.kind
    for entry in SOLVER_HELPER_REGISTRY.values()
    if entry.stage_aware
} == STAGE_AWARE_KINDS


def helper_source_hash(system, request: SolverHelperRequest) -> str:
    """Return the generated-source identity for a request.

    Contains only inputs that change the emitted source: helper kind,
    the ODE equation/layout identity, the mass matrix for generators
    that consume it, and the canonical stage specification for
    stage-aware generators. Binding values (beta, gamma, order,
    constants, precision, lineinfo) are deliberately absent.
    """
    entry = SOLVER_HELPER_REGISTRY[request.kind]
    mass = system.compile_settings.mass if entry.uses_mass else None
    return canonical_digest(
        (
            "cubie-helper-source",
            request.kind.value,
            system.fn_hash,
            mass,
            request.stage_identity if entry.stage_aware else None,
        )
    )


def helper_member_hash(source_hash: str, canonical_args: tuple) -> str:
    """Return the bound-member identity for one factory binding."""
    return canonical_digest(
        ("cubie-helper-member", source_hash, canonical_args)
    )
