"""CUDA factory for dense stage prediction of implicit RK methods.

After an implicit Runge--Kutta step converges, its stage increments
sample the state's derivative curve at the stage times. Reading that
curve ahead over the next step gives a far better Newton starting
guess than reusing the previous increments unchanged; for
collocation tableaus this is exactly the classical extrapolation of
the collocation polynomial.

:class:`DenseStagePredictor` compiles that read-ahead as an in-place
device function over a stage-major increment vector. The transform is
a matrix whose entries are polynomials in the step-size ratio
``next dt / previous dt``; the polynomial coefficients are fixed by
the tableau and precomputed on the host, so the device evaluates one
small polynomial per matrix entry and applies one matrix-vector
product per state. The only persistent storage is a single scalar
holding the previous step size.

The predictor does not decide when prediction is valid: the calling
algorithm judges first-step and rejected-step conditions and passes
the verdict as a flag. The predictor must be called on every step so
its stored step size tracks the increments the solver last wrote.

Published Functions
-------------------
:func:`dense_predictor_matrix`
    Return the dense extrapolation matrix for a tableau and ratio.

:func:`dense_predictor_ratio_coefficients`
    Return the per-power coefficient matrices of the ratio polynomial.

:func:`tableau_supports_dense_prediction`
    Whether a tableau satisfies the transform's preconditions.

Published Classes
-----------------
:class:`DenseStagePredictorConfig`
    Attrs compile settings for the predictor factory.

:class:`DenseStagePredictor`
    CUDAFactory building the in-place prediction device function.

See Also
--------
:class:`~cubie.integrators.algorithms.generic_firk.FIRKStep`
    Owns a predictor and applies it to its coupled stage vector.
:class:`~cubie.integrators.algorithms.generic_dirk.DIRKStep`
    Owns a predictor and applies it to its stage-increment history.
"""

from typing import Callable

from attrs import define, field, validators
from numpy import asarray as np_asarray
from numpy import ndarray as np_ndarray
from numpy import zeros as np_zeros
from numpy.polynomial import polynomial as np_poly
from numpy.polynomial.legendre import leggauss as np_leggauss

from cubie._utils import (
    PrecisionDType,
    build_config,
    getype_validator,
    is_device_validator,
)
from cubie.buffer_registry import buffer_registry
from cubie.CUDAFactory import (
    CUDADispatcherCache,
    CUDAFactory,
    CUDAFactoryConfig,
)
from cubie.cuda_simsafe import cuda, int32
from cubie.integrators.algorithms.base_algorithm_step import (
    ButcherTableau,
)


MAX_PREDICTION_STEP_RATIO = 8.0
"""Largest step-size ratio the transform is applied at.

Matches the largest single-step growth the default controllers allow
(RADAU5's ``facr``). Beyond it the read-ahead leaves the sampled
interval so far that amplification outweighs the guess — a save
boundary can truncate a step to a sliver, making the next ratio
arbitrarily large — so the increments carry unchanged instead.
"""

MAX_NODE_AMPLIFICATION_RATIO = 10.0
"""Largest amplification allowed relative to ideally spread nodes.

The read-ahead multiplies each sample's contribution by its
extrapolation weight, so the worst-row sum of absolute weights bounds
how much sample error (the solver converges only to tolerance) is
amplified into the guess. That sum grows with stage count even for
the best possible node spread, so the check compares against the
Gauss--Legendre nodes of the same stage count: clustered nodes push
the ratio into the tens, where the guess is no better than carrying
the increments unchanged, and such tableaus fail the precondition
check instead.
"""


def _distinct_stage_nodes(
    stage_nodes: np_ndarray,
) -> tuple:
    """Return the distinct nodes and the stage kept at each node.

    Two samples at one time cannot both pin a single curve, so a
    repeated node contributes the last sharing stage's sample only.
    """

    representatives = {}
    for stage_index, node in enumerate(stage_nodes.tolist()):
        representatives[node] = stage_index
    distinct_nodes = np_asarray(list(representatives.keys()))
    return distinct_nodes, tuple(representatives.values())


def _predictor_matrix_from_nodes(
    stage_nodes: np_ndarray,
    step_ratio: float,
) -> np_ndarray:
    """Return the read-ahead matrix for *stage_nodes*."""

    stage_count = len(stage_nodes)
    distinct_nodes, sample_indices = _distinct_stage_nodes(stage_nodes)
    # Weight of each kept derivative sample in the curve's value at
    # each of the next step's stage times, rescaled by the ratio
    # because increments carry a factor of the step size. Stages
    # sharing a node get equal rows; dropped samples zero columns.
    predictor = np_zeros((stage_count, stage_count))
    for target_index, target_node in enumerate(
        1.0 + step_ratio * stage_nodes
    ):
        for basis_index, basis_node in enumerate(distinct_nodes):
            weight = step_ratio
            # Scale by distance ratios to every other kept sample.
            for other_index, other_node in enumerate(distinct_nodes):
                if other_index != basis_index:
                    weight *= (target_node - other_node) / (
                        basis_node - other_node
                    )
            predictor[
                target_index, sample_indices[basis_index]
            ] = weight

    return predictor


def _validate_tableau(tableau: ButcherTableau) -> np_ndarray:
    """Check the transform's preconditions and return the nodes.

    Parameters
    ----------
    tableau
        Runge--Kutta tableau to validate.

    Returns
    -------
    numpy.ndarray
        The tableau's stage nodes in float64.

    Raises
    ------
    ValueError
        If any precondition fails.
    """

    stage_nodes = np_asarray(tableau.c, dtype=float)
    distinct_nodes, _ = _distinct_stage_nodes(stage_nodes)
    distinct_count = len(distinct_nodes)

    equal_step = _predictor_matrix_from_nodes(stage_nodes, 1.0)
    amplification = float(abs(equal_step).sum(axis=1).max())
    gauss_nodes = 0.5 * (np_leggauss(distinct_count)[0] + 1.0)
    reference = float(
        abs(_predictor_matrix_from_nodes(gauss_nodes, 1.0))
        .sum(axis=1)
        .max()
    )
    if amplification > MAX_NODE_AMPLIFICATION_RATIO * reference:
        raise ValueError(
            "Dense prediction amplifies stage-sample error "
            f"{amplification:.0f}-fold through this tableau's "
            f"nodes ({reference:.0f}-fold for ideally spread "
            "nodes); the guess would be no better than the "
            "carried increments."
        )
    return stage_nodes


def dense_predictor_matrix(
    tableau: ButcherTableau,
    step_ratio: float = 1.0,
) -> np_ndarray:
    """Return the dense extrapolation matrix for *tableau*.

    Each converged stage increment is the step size times the state
    derivative at that stage's time, so the previous step's
    increments sample the derivative curve at the stage nodes. The
    matrix reads the curve through those samples ahead at the next
    step's stage times and rescales by the step-size ratio, mapping
    previous increments to a starting guess for the next step's
    increments. For collocation tableaus this equals extrapolating
    the collocation polynomial itself. Repeated stage nodes each
    contribute one sample (the last stage sharing the node), and
    stages sharing a node share a predicted value.

    Parameters
    ----------
    tableau
        Runge--Kutta tableau meeting the transform's preconditions.
    step_ratio
        Size of the next step relative to the previous step.

    Returns
    -------
    numpy.ndarray
        Dense predictor matrix in float64.

    Raises
    ------
    ValueError
        If the tableau does not meet the transform's preconditions.
    """

    stage_nodes = _validate_tableau(tableau)
    return _predictor_matrix_from_nodes(stage_nodes, step_ratio)


def dense_predictor_ratio_coefficients(
    tableau: ButcherTableau,
) -> np_ndarray:
    """Return the per-power coefficient matrices of the transform.

    Each entry of the predictor matrix is a polynomial in the
    step-size ratio with no constant term and degree equal to the
    number of distinct stage nodes. Expanding each stage-value weight
    about the end of the previous step yields those polynomial
    coefficients directly.

    Parameters
    ----------
    tableau
        Runge--Kutta tableau meeting the transform's preconditions.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(distinct_count, stage_count, stage_count)``
        where index ``p`` holds the matrix multiplying
        ``step_ratio ** (p + 1)`` and ``distinct_count`` is the
        number of distinct stage nodes.

    Raises
    ------
    ValueError
        If the tableau does not meet the transform's preconditions.
    """

    stage_nodes = _validate_tableau(tableau)
    stage_count = tableau.stage_count
    distinct_nodes, sample_indices = _distinct_stage_nodes(stage_nodes)
    distinct_count = len(distinct_nodes)

    # Weight of each kept derivative sample in the curve's value at a
    # time (1 + y) step-lengths after the previous step's start,
    # expanded in powers of y.
    shift = np_poly.Polynomial([1.0, 1.0])
    shifted_weight_powers = []
    for basis_index in range(distinct_count):
        other_nodes = [
            distinct_nodes[node_index]
            for node_index in range(distinct_count)
            if node_index != basis_index
        ]
        coefficients = np_poly.polyfromroots(other_nodes)
        coefficients = coefficients / np_poly.polyval(
            distinct_nodes[basis_index], coefficients
        )
        shifted = np_poly.Polynomial(coefficients)(shift)
        shifted_weight_powers.append(shifted.coef)

    # The new step's stage times sit at y = ratio * node, so the
    # weight's power-p term contributes at ratio power p + 1 once
    # the increments' step-size factor is rescaled. Kept samples fill
    # their stage's column; dropped duplicates keep zero columns.
    ratio_coefficients = np_zeros(
        (distinct_count, stage_count, stage_count)
    )
    for power in range(distinct_count):
        for target_index in range(stage_count):
            node_power = stage_nodes[target_index] ** power
            for basis_index in range(distinct_count):
                ratio_coefficients[
                    power, target_index, sample_indices[basis_index]
                ] = shifted_weight_powers[basis_index][power] * node_power
    return ratio_coefficients


def tableau_supports_dense_prediction(
    tableau: ButcherTableau,
) -> bool:
    """Return whether *tableau* meets the transform preconditions."""

    try:
        _validate_tableau(tableau)
    except ValueError:
        return False
    return True


@define
class DenseStagePredictorConfig(CUDAFactoryConfig):
    """Compile settings for :class:`DenseStagePredictor`.

    Attributes
    ----------
    n : int
        Number of state variables per stage.
    tableau : ButcherTableau
        Tableau the prediction matrix derives from.
    predict_first_stage : bool
        Whether the transform predicts the first stage. Callers that
        never solve it pass ``False``; its entries then keep the
        stored sample. Must stay ``True`` for single-stage tableaus.
    previous_step_size_location : str
        Buffer location for the previous-step-size scalar.
    """

    n: int = field(default=1, validator=getype_validator(int, 1))
    tableau: ButcherTableau = field(
        default=None,
        validator=validators.instance_of(ButcherTableau),
    )
    predict_first_stage: bool = field(
        default=True, validator=validators.instance_of(bool)
    )
    previous_step_size_location: str = field(
        default="local",
        validator=validators.in_(["local", "shared"]),
    )

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""

        return self.tableau.stage_count


@define
class DenseStagePredictorCache(CUDADispatcherCache):
    """Hold the compiled in-place prediction device function."""

    predict: Callable = field(validator=is_device_validator)


class DenseStagePredictor(CUDAFactory):
    """Compile the in-place dense stage prediction transform.

    The compiled device function has signature::

        predict(stage_increment, dt_scalar, apply_flag, shared,
                persistent_local)

    ``stage_increment`` holds the previous step's converged stage
    increments in stage-major layout. When ``apply_flag`` is true the
    vector is transformed in place into the next step's Newton guess,
    unless the step-size ratio exceeds
    :data:`MAX_PREDICTION_STEP_RATIO`; otherwise it is left
    unchanged. With ``predict_first_stage`` false the first stage's
    entries are left untouched (its sample still feeds the read-ahead
    of the other stages). Either way the stored previous step size is
    refreshed, so the caller must invoke the function on every step
    and must pass ``apply_flag`` false on the first step and after a
    rejected step.
    """

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        tableau: ButcherTableau,
        **kwargs,
    ) -> None:
        """Initialise the predictor factory.

        Parameters
        ----------
        precision
            Floating-point precision for the transform.
        n
            Number of state variables per stage.
        tableau
            Tableau the prediction matrix derives from.
        **kwargs
            Optional ``predict_first_stage`` and
            ``previous_step_size_location`` settings. None values are
            ignored.
        """

        super().__init__()
        config = build_config(
            DenseStagePredictorConfig,
            required={
                "precision": precision,
                "n": n,
                "tableau": tableau,
            },
            **kwargs,
        )
        self.setup_compile_settings(config)
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register the predictor's buffers with the registry."""

        config = self.compile_settings
        stage_count = int(config.stage_count)
        # A skipped first stage drops its row from the transform.
        predicted_rows = stage_count - (
            0 if config.predict_first_stage else 1
        )
        buffer_registry.register(
            "previous_step_size",
            self,
            1,
            config.previous_step_size_location,
            persistent=True,
            precision=config.precision,
        )
        buffer_registry.register(
            "predictor_transform",
            self,
            predicted_rows * stage_count,
            "local",
            precision=config.precision,
        )
        buffer_registry.register(
            "predictor_previous_values",
            self,
            stage_count,
            "local",
            precision=config.precision,
        )

    def update(self, updates_dict=None, silent=False, **kwargs):
        """Update compile settings and re-register buffers.

        Returns
        -------
        set
            Set of recognized parameter names that were updated.
        """

        all_updates = {}
        if updates_dict:
            all_updates.update(updates_dict)
        all_updates.update(kwargs)
        if not all_updates:
            return set()

        recognised = self.update_compile_settings(
            updates_dict=all_updates, silent=silent
        )
        if recognised:
            self.register_buffers()
        return recognised

    @property
    def device_function(self) -> Callable:
        """Return the compiled prediction device function."""

        return self.get_cached_output("predict")

    def build(self) -> DenseStagePredictorCache:
        """Compile the in-place prediction device function."""

        config = self.compile_settings
        numba_precision = config.numba_precision
        typed_zero = numba_precision(0.0)
        n = int32(config.n)
        stage_count = int32(config.stage_count)
        # A skipped first stage drops its row from the transform.
        first_predicted_py = 0 if config.predict_first_stage else 1
        predicted_count_py = int(config.stage_count) - first_predicted_py
        transform_size = int32(
            predicted_count_py * int(config.stage_count)
        )
        first_predicted = int32(first_predicted_py)
        predicted_count = int32(predicted_count_py)

        # Flattened power-major coefficient stack for the predicted
        # rows: entry [power][row][column] multiplies
        # ratio ** (power + 1).
        coefficient_stack = dense_predictor_ratio_coefficients(
            config.tableau
        )[:, first_predicted_py:, :]
        power_count = int32(coefficient_stack.shape[0])
        ratio_coefficients = tuple(
            numba_precision(value) for value in coefficient_stack.flat
        )

        max_step_ratio = numba_precision(MAX_PREDICTION_STEP_RATIO)

        getalloc = buffer_registry.get_allocator
        alloc_previous_step_size = getalloc("previous_step_size", self)
        alloc_transform = getalloc("predictor_transform", self)
        alloc_previous_values = getalloc(
            "predictor_previous_values", self
        )

        # no cover: start
        @cuda.jit(
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def predict(
            stage_increment,
            dt_scalar,
            apply_flag,
            shared,
            persistent_local,
        ):
            previous_step_size = alloc_previous_step_size(
                shared, persistent_local
            )
            transform = alloc_transform(shared, persistent_local)
            previous_values = alloc_previous_values(
                shared, persistent_local
            )
            previous_dt = previous_step_size[0]
            previous_step_size[0] = dt_scalar

            # Keep the ratio finite on zeroed first-step storage.
            safe_previous_dt = (
                previous_dt if previous_dt > typed_zero else dt_scalar
            )
            ratio = dt_scalar / safe_previous_dt

            # Truncated slivers unbound the ratio; cap application.
            commit_prediction = apply_flag and (
                ratio <= max_step_ratio
            )

            # Evaluate each matrix entry's ratio polynomial, highest
            # power first.
            for entry_idx in range(transform_size):
                accumulator = typed_zero
                for power_idx in range(power_count):
                    flat_idx = (
                        (power_count - int32(1) - power_idx)
                        * transform_size
                        + entry_idx
                    )
                    accumulator = (
                        accumulator * ratio
                        + ratio_coefficients[flat_idx]
                    )
                transform[entry_idx] = accumulator * ratio

            # Multiply each state's stage vector by the matrix,
            # committing only when the caller's flag and the ratio
            # bound allow.
            for state_idx in range(n):
                for stage_idx in range(stage_count):
                    previous_values[stage_idx] = stage_increment[
                        stage_idx * n + state_idx
                    ]
                for row_idx in range(predicted_count):
                    accumulator = typed_zero
                    row_base = row_idx * stage_count
                    for source_idx in range(stage_count):
                        accumulator += (
                            transform[row_base + source_idx]
                            * previous_values[source_idx]
                        )
                    if commit_prediction:
                        stage_increment[
                            (first_predicted + row_idx) * n + state_idx
                        ] = accumulator

        # no cover: end
        return DenseStagePredictorCache(predict=predict)
