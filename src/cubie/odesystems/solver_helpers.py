"""Solver-helper request and cache containers.

Implicit algorithms consume generated device helpers (operators,
residuals, preconditioners, Jacobian caches) from an ODE system. Each
lookup is described by an immutable :class:`SolverHelperRequest`
derived from the requesting algorithm's compile settings — helper
request state never mutates the ODE system's configuration, so an ODE
may serve multiple algorithms or beta/gamma/tableau bindings without
its identity depending on request order.

The generation side (source emitters and their binding contracts)
lives in :mod:`cubie.odesystems.symbolic.helper_registry`; this module
holds only the request/product containers so the abstract ODE base can
reference them without importing the symbolic pipeline.

Published Classes
-----------------
:class:`SolverHelperKind`
    Enumeration of concrete generated-helper kinds.
:class:`SolverHelperRequest`
    Frozen description of one helper lookup.
:class:`HelperResult`
    A bound helper member: device callable plus typed metadata.
:class:`SolverHelperCache`
    Memoized generated factories and bound members for one live ODE
    build.
"""

from enum import Enum
from typing import Any, Callable, Optional, Tuple

import sympy as sp
from attrs import Factory, define, field, frozen

__all__ = [
    "SolverHelperKind",
    "STAGE_AWARE_KINDS",
    "SolverHelperRequest",
    "HelperResult",
    "SolverHelperCache",
]


class SolverHelperKind(Enum):
    """Concrete generated-helper kinds.

    Composite preconditioner names are not helper kinds: the implicit
    algorithm layer resolves ``preconditioner_type`` into one or two
    concrete requests and owns the chain composition.
    """

    LINEAR_OPERATOR = "linear_operator"
    LINEAR_OPERATOR_CACHED = "linear_operator_cached"
    NEUMANN_PRECONDITIONER = "neumann_preconditioner"
    NEUMANN_PRECONDITIONER_CACHED = "neumann_preconditioner_cached"
    JACOBI_PRECONDITIONER = "jacobi_preconditioner"
    JACOBI_PRECONDITIONER_CACHED = "jacobi_preconditioner_cached"
    STAGE_RESIDUAL = "stage_residual"
    N_STAGE_RESIDUAL = "n_stage_residual"
    N_STAGE_LINEAR_OPERATOR = "n_stage_linear_operator"
    N_STAGE_NEUMANN_PRECONDITIONER = "n_stage_neumann_preconditioner"
    N_STAGE_JACOBI_PRECONDITIONER = "n_stage_jacobi_preconditioner"
    PREPARE_JAC = "prepare_jac"
    CALCULATE_CACHED_JVP = "calculate_cached_jvp"
    TIME_DERIVATIVE_RHS = "time_derivative_rhs"


STAGE_AWARE_KINDS = frozenset(
    (
        SolverHelperKind.N_STAGE_RESIDUAL,
        SolverHelperKind.N_STAGE_LINEAR_OPERATOR,
        SolverHelperKind.N_STAGE_NEUMANN_PRECONDITIONER,
        SolverHelperKind.N_STAGE_JACOBI_PRECONDITIONER,
    )
)
"""Kinds whose emitted source depends on the stage specification."""


def _kind_converter(value: Any) -> SolverHelperKind:
    """Accept a kind enum member or its string value."""
    if isinstance(value, SolverHelperKind):
        return value
    return SolverHelperKind(value)


def _stage_value_repr(value: Any) -> str:
    """Return the canonical text form of one stage entry."""
    return sp.srepr(sp.sympify(value))


def _stage_matrix_converter(value: Any) -> Optional[Tuple[tuple, ...]]:
    """Normalise stage coefficients to a tuple of row tuples."""
    if value is None:
        return None
    return tuple(tuple(row) for row in value)


def _stage_vector_converter(value: Any) -> Optional[tuple]:
    """Normalise stage nodes to a tuple."""
    if value is None:
        return None
    return tuple(value)


@frozen
class SolverHelperRequest:
    """Immutable description of one solver-helper lookup.

    Parameters
    ----------
    kind
        Concrete helper kind, as an enum member or its string value.
    beta
        Shift scaling applied to the mass-matrix term, where the
        helper consumes it.
    gamma
        Weight applied to the Jacobian term, where the helper
        consumes it.
    preconditioner_order
        Polynomial order of Neumann preconditioners, where the helper
        consumes it.
    stage_coefficients
        Stage coupling matrix for stage-aware helpers, row-major.
        Entries may be floats or exact SymPy numbers.
    stage_nodes
        Stage nodes expressed as timestep fractions for stage-aware
        helpers.

    Notes
    -----
    Stage entries participate in identity through their canonical
    SymPy text form, so exact and floating forms of the same tableau
    are distinguished deliberately — they emit different source.
    Unsupported combinations fail at construction: stage-aware kinds
    require stage data and other kinds reject it.
    """

    kind: SolverHelperKind = field(converter=_kind_converter)
    beta: float = field(default=1.0, converter=float)
    gamma: float = field(default=1.0, converter=float)
    preconditioner_order: int = field(default=2, converter=int)
    stage_coefficients: Optional[Tuple[tuple, ...]] = field(
        default=None, converter=_stage_matrix_converter, eq=False
    )
    stage_nodes: Optional[tuple] = field(
        default=None, converter=_stage_vector_converter, eq=False
    )
    _stage_identity: Optional[tuple] = field(
        default=None, init=False, repr=False
    )

    def __attrs_post_init__(self):
        if self.kind in STAGE_AWARE_KINDS:
            if self.stage_coefficients is None or self.stage_nodes is None:
                raise ValueError(
                    f"Helper kind '{self.kind.value}' requires stage "
                    "coefficients and stage nodes."
                )
            rows = tuple(
                tuple(_stage_value_repr(value) for value in row)
                for row in self.stage_coefficients
            )
            nodes = tuple(
                _stage_value_repr(value) for value in self.stage_nodes
            )
            object.__setattr__(self, "_stage_identity", (rows, nodes))
        elif (
            self.stage_coefficients is not None
            or self.stage_nodes is not None
        ):
            raise ValueError(
                f"Helper kind '{self.kind.value}' does not consume "
                "stage coefficients or stage nodes."
            )

    @property
    def stage_identity(self) -> Optional[tuple]:
        """Canonical identity of the stage specification, if any."""
        return self._stage_identity

    def _cubie_canonical_(self) -> tuple:
        """Return the canonical identity of this request."""
        return (
            "SolverHelperRequest",
            self.kind.value,
            self.beta,
            self.gamma,
            self.preconditioner_order,
            self._stage_identity,
        )


@define
class HelperResult:
    """One bound helper member.

    Attributes
    ----------
    device_function
        The compiled device callable.
    cached_auxiliary_count
        Number of precomputed auxiliary slots the helper populates or
        consumes. Set for ``prepare_jac``; ``None`` otherwise.
    """

    device_function: Callable
    cached_auxiliary_count: Optional[int] = None


@define
class SolverHelperCache:
    """Memoized helper products for one live ODE build.

    The maps are intentionally mutable: they memoize products derived
    from immutable requests and the immutable ODE snapshot. A true ODE
    compile-setting change rebuilds the ODE build product and starts a
    fresh member map.

    Attributes
    ----------
    factories
        Imported generated factories keyed by ``source_hash``.
    members
        Bound helper members keyed by ``member_hash``.
    """

    factories: dict = Factory(dict)
    members: dict = Factory(dict)
