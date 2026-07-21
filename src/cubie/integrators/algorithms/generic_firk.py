"""Fully implicit Runge–Kutta integration step implementation.

Published Classes
-----------------
:class:`FIRKStepConfig`
    Configuration container for the FIRK step.

:class:`FIRKStep`
    Multi-stage fully implicit step solving all stages as a coupled
    nonlinear system. Uses Kahan summation for output accumulation.

Constants
---------
:data:`FIRK_ADAPTIVE_DEFAULTS`
    Default Gustafsson controller settings for adaptive tableaus.

:data:`FIRK_FIXED_DEFAULTS`
    Default fixed-step settings for errorless tableaus.

Notes
-----
The step controller defaults are selected dynamically based on whether
the tableau has an embedded error estimate. FIRK methods require
solving a coupled system of all stages simultaneously, which is more
expensive than DIRK methods but can achieve higher orders for stiff
systems.

See Also
--------
:class:`~cubie.integrators.algorithms.ode_implicitstep.ODEImplicitStep`
    Abstract parent managing the Newton–Krylov solver lifecycle.
:class:`~cubie.integrators.algorithms.generic_firk_tableaus.FIRKTableau`
    Tableau class describing FIRK coefficients.
:class:`FIRKStepConfig`
    Configuration for this step.
"""

from typing import Callable, Optional

from attrs import define, field, validators
from cubie.cuda_simsafe import cuda, int32, selp
from numpy import (
    asarray as np_asarray,
    linalg as np_linalg,
    ndarray as np_ndarray,
)

from cubie.result_codes import CUBIE_RESULT_CODES

from cubie._utils import PrecisionDType, build_config
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.generic_firk_tableaus import (
    DEFAULT_FIRK_TABLEAU,
    FIRKTableau,
)
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig,
    ODEImplicitStep,
)
from cubie.integrators.norms import FIRKCorrectionNorm, TiledScaledNorm
from cubie.buffer_registry import buffer_registry


FIRK_ADAPTIVE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "gustafsson",
        "deadband_min": 1.0,
        "deadband_max": 1.2,
        "min_gain": 0.2,
        "max_gain": 8.0,
        "safety": 0.9,
    }
)
"""Default step controller settings for adaptive FIRK tableaus.

This configuration is used when the FIRK tableau has an embedded error
estimate (``tableau.has_error_estimate == True``).

The Gustafsson predictive controller with these limits reproduces the
step control of Hairer & Wanner's RADAU5, the reference implementation
for fully implicit Runge--Kutta methods (``facl = 0.2``, ``facr = 8``,
``quot1 = 1.0``, ``quot2 = 1.2``, ``safe = 0.9``). The deadband keeps
the step unchanged for small gains so warp-coherent threads avoid
needless step-size churn.

Notes
-----
These defaults are applied automatically when creating a :class:`FIRKStep`
with an adaptive tableau. Users can override any of these settings by
explicitly specifying step controller parameters.
"""

FIRK_FIXED_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
    }
)
"""Default step controller settings for errorless FIRK tableaus.

This configuration is used when the FIRK tableau lacks an embedded error
estimate (``tableau.has_error_estimate == False``).

Fixed-step controllers maintain a constant step size throughout the
integration. This is the only valid choice for errorless tableaus since
adaptive stepping requires an error estimate to adjust the step size.
"""


def _dense_predictor_matrix(tableau: FIRKTableau) -> np_ndarray:
    """Return the one-step dense extrapolation matrix for *tableau*.

    The FIRK unknown contains derivative increments ``U`` while the
    collocation polynomial is defined by state increments ``A U``.  The
    returned matrix maps the previous step's ``U`` to the next equal-size
    step's stage guess.

    Parameters
    ----------
    tableau
        Fully implicit Runge--Kutta tableau.

    Returns
    -------
    numpy.ndarray
        Dense predictor matrix in host coefficient precision.
    """

    stage_count = tableau.stage_count
    nodes = np_asarray((0.0,) + tuple(tableau.c), dtype=float)
    stage_nodes = np_asarray(tableau.c, dtype=float)

    endpoint_weights = []
    for basis_index in range(1, stage_count + 1):
        weight = 1.0
        basis_node = nodes[basis_index]
        for other_index, other_node in enumerate(nodes):
            if other_index != basis_index:
                weight *= (1.0 - other_node) / (basis_node - other_node)
        endpoint_weights.append(weight)

    extrapolation = []
    for target_node in 1.0 + stage_nodes:
        target_weights = []
        for basis_index in range(1, stage_count + 1):
            weight = 1.0
            basis_node = nodes[basis_index]
            for other_index, other_node in enumerate(nodes):
                if other_index != basis_index:
                    weight *= (target_node - other_node) / (
                        basis_node - other_node
                    )
            target_weights.append(weight)
        extrapolation.append(
            [
                target_weights[index] - endpoint_weights[index]
                for index in range(stage_count)
            ]
        )

    stage_matrix = np_asarray(tableau.a, dtype=float)
    return np_linalg.solve(
        stage_matrix,
        np_asarray(extrapolation) @ stage_matrix,
    )


def _lu_factor(
    matrix: np_ndarray,
) -> tuple[np_ndarray, np_ndarray, tuple[tuple[int, int], ...]]:
    """Factor *matrix* for a buffer-free in-place device transform.

    Parameters
    ----------
    matrix
        Square dense matrix.

    Returns
    -------
    tuple
        Unit-lower and upper factors plus inverse-permutation swaps.
    """

    upper = np_asarray(matrix).copy()
    stage_count = upper.shape[0]
    lower = np_asarray(
        [
            [1.0 if row == column else 0.0 for column in range(stage_count)]
            for row in range(stage_count)
        ],
        dtype=upper.dtype,
    )
    pivot_swaps = []

    for column in range(stage_count):
        pivot_row = column
        pivot_magnitude = abs(upper[column, column])
        for candidate_row in range(column + 1, stage_count):
            candidate_magnitude = abs(upper[candidate_row, column])
            if candidate_magnitude > pivot_magnitude:
                pivot_row = candidate_row
                pivot_magnitude = candidate_magnitude
        if pivot_magnitude == 0.0:
            raise ValueError("Dense FIRK predictor matrix is singular")
        if pivot_row != column:
            upper[[column, pivot_row], :] = upper[[pivot_row, column], :]
            if column:
                lower[[column, pivot_row], :column] = lower[
                    [pivot_row, column], :column
                ]
            pivot_swaps.append((column, pivot_row))

        for row in range(column + 1, stage_count):
            multiplier = upper[row, column] / upper[column, column]
            lower[row, column] = multiplier
            upper[row, column:] -= multiplier * upper[column, column:]

    return lower, upper, tuple(reversed(pivot_swaps))


@define
class FIRKStepConfig(ImplicitStepConfig):
    """Configuration describing the FIRK integrator."""

    tableau: FIRKTableau = field(
        default=DEFAULT_FIRK_TABLEAU,
    )
    stage_increment_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    stage_driver_stack_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )
    stage_state_location: str = field(
        default="local", validator=validators.in_(["local", "shared"])
    )

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""

        return self.tableau.stage_count

    @property
    def all_stages_n(self) -> int:
        """Return the flattened dimension covering all stage increments."""

        return self.stage_count * self.n

    @property
    def solver_n(self) -> int:
        """Return the coupled solver dimension across all stages."""

        return self.all_stages_n


class FIRKStep(ODEImplicitStep):
    """Fully implicit Runge--Kutta step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        evaluate_f: Optional[Callable] = None,
        evaluate_observables: Optional[Callable] = None,
        evaluate_driver_at_t: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        tableau: FIRKTableau = DEFAULT_FIRK_TABLEAU,
        n_drivers: int = 0,
        **kwargs,
    ) -> None:
        """Initialise the FIRK step configuration.

        This constructor creates a FIRK step object and automatically selects
        appropriate default step controller settings based on whether the
        tableau has an embedded error estimate. Tableaus with error estimates
        default to adaptive stepping (Gustafsson controller), while
        errorless tableaus default to fixed stepping.

        Parameters
        ----------
        precision
            Floating-point precision for CUDA computations.
        n
            Number of state variables in the ODE system.
        evaluate_f
            Device function for evaluating f(t, y) right-hand side.
        evaluate_observables
            Device function computing system observables.
        evaluate_driver_at_t
            Optional device function evaluating drivers at arbitrary times.
        get_solver_helper_fn
            Factory function returning solver helper for Jacobian operations.
        tableau
            FIRK tableau describing the coefficients. Defaults to
            :data:`DEFAULT_FIRK_TABLEAU`.
        n_drivers
            Number of driver variables in the system.
        **kwargs
            Optional parameters passed to config classes. See
            FIRKStepConfig, ImplicitStepConfig, and solver config classes
            for available parameters. None values are ignored.

        Notes
        -----
        The step controller defaults are selected dynamically:

        - If ``tableau.has_error_estimate`` is ``True``:
          Uses :data:`FIRK_ADAPTIVE_DEFAULTS` (Gustafsson controller)
        - If ``tableau.has_error_estimate`` is ``False``:
          Uses :data:`FIRK_FIXED_DEFAULTS` (fixed-step controller)

        This automatic selection prevents incompatible configurations where
        an adaptive controller is paired with an errorless tableau.

        FIRK methods require solving a coupled system of all stages
        simultaneously, which is more computationally expensive than DIRK
        methods but can achieve higher orders of accuracy for stiff systems.
        """
        config = build_config(
            FIRKStepConfig,
            required={
                "precision": precision,
                "n": n,
                "n_drivers": n_drivers,
                "evaluate_f": evaluate_f,
                "evaluate_observables": evaluate_observables,
                "evaluate_driver_at_t": evaluate_driver_at_t,
                "get_solver_helper_fn": get_solver_helper_fn,
                "tableau": tableau,
                "beta": 1.0,
                "gamma": 1.0,
            },
            **kwargs,
        )

        # Select defaults based on error estimate
        if tableau.has_error_estimate:
            controller_defaults = FIRK_ADAPTIVE_DEFAULTS
        else:
            controller_defaults = FIRK_FIXED_DEFAULTS

        newton_norm = FIRKCorrectionNorm(
            precision=precision,
            n=config.all_stages_n,
            state_n=n,
            stage_coefficients=tableau.a_flat(float),
            instance_label="newton",
            **kwargs,
        )
        # The coupled solve stacks all stages, but callers pass the
        # single-stage base state; the tiled norm reuses it per stage.
        krylov_norm = TiledScaledNorm(
            precision=precision,
            n=config.all_stages_n,
            state_n=n,
            instance_label="krylov",
            **kwargs,
        )
        super().__init__(
            config,
            controller_defaults,
            newton_norm=newton_norm,
            krylov_norm=krylov_norm,
            **kwargs,
        )
        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers according to locations in compile settings."""
        config = self.compile_settings
        precision = config.precision
        n = config.n
        tableau = config.tableau

        # Calculate buffer sizes
        all_stages_n = tableau.stage_count * n
        stage_driver_stack_elements = tableau.stage_count * config.n_drivers

        buffer_registry.register_child(
            self, self.solver, name="solver"
        )
        buffer_registry.register(
            "stage_increment",
            self,
            all_stages_n,
            config.stage_increment_location,
            persistent=True,
            precision=precision,
        )
        buffer_registry.register(
            "stage_driver_stack",
            self,
            stage_driver_stack_elements,
            config.stage_driver_stack_location,
            precision=precision,
        )
        buffer_registry.register(
            "stage_state",
            self,
            n,
            config.stage_state_location,
            precision=precision,
        )

    def build_implicit_helpers(
        self,
    ) -> None:
        """Construct the nonlinear solver chain used by implicit methods."""

        config = self.compile_settings
        tableau = config.tableau
        beta = config.beta
        gamma = config.gamma

        get_fn = config.get_solver_helper_fn

        stage_coefficients = [list(row) for row in tableau.a]
        stage_nodes = list(tableau.c)

        residual = get_fn(
            "n_stage_residual",
            solver_beta=beta,
            solver_gamma=gamma,
            stage_coefficients=stage_coefficients,
            stage_nodes=stage_nodes,
        )

        operator = get_fn(
            "n_stage_linear_operator",
            solver_beta=beta,
            solver_gamma=gamma,
            stage_coefficients=stage_coefficients,
            stage_nodes=stage_nodes,
        )

        preconditioner = get_fn(
            "n_stage_preconditioner",
            preconditioner_type=config.preconditioner_type,
            solver_beta=beta,
            solver_gamma=gamma,
            preconditioner_order=config.preconditioner_order,
            stage_coefficients=stage_coefficients,
            stage_nodes=stage_nodes,
        )

        # Update solvers with device functions
        self.solver.update(
            operator_apply=operator,
            preconditioner=preconditioner,
            preconditioner_is_chained=(
                config.preconditioner_is_chained
            ),
            residual_function=residual,
            n=config.all_stages_n,
        )

        self.update_compile_settings(
            {"solver_function": self.solver.device_function}
        )

    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the FIRK device step."""

        config = self.compile_settings
        tableau = config.tableau
        enable_dense_prediction = bool(
            getattr(self, "is_controller_fixed", False)
        )
        buffer_registry.register(
            "previous_step_size",
            self,
            int(enable_dense_prediction),
            config.stage_increment_location,
            persistent=True,
            precision=config.precision,
        )

        nonlinear_solver = solver_function

        n = int32(n)
        n_drivers = int32(n_drivers)
        stage_count = int32(self.stage_count)

        has_evaluate_driver_at_t = evaluate_driver_at_t is not None
        has_error = self.is_adaptive

        stage_rhs_coeffs = tableau.a_flat(numba_precision)
        solution_weights = tableau.typed_vector(tableau.b, numba_precision)
        typed_zero = numba_precision(0.0)
        success = int32(CUBIE_RESULT_CODES.SUCCESS)
        error_weights = tableau.error_weights(numba_precision)
        if error_weights is None or not has_error:
            error_weights = tuple(typed_zero for _ in range(stage_count))
        stage_time_fractions = tableau.typed_vector(tableau.c, numba_precision)

        dense_predictor = np_asarray(
            _dense_predictor_matrix(tableau),
            dtype=config.precision,
        )
        dense_lower, dense_upper, dense_swaps = _lu_factor(dense_predictor)
        dense_coefficients_unpadded = tuple(
            numba_precision(value) for value in dense_predictor.flat
        )
        dense_coefficients = dense_coefficients_unpadded + tuple(
            typed_zero
            for _ in range(max(0, 9 - len(dense_coefficients_unpadded)))
        )
        dense_lower_coefficients = tuple(
            numba_precision(value) for value in dense_lower.flat
        )
        dense_upper_coefficients = tuple(
            numba_precision(value) for value in dense_upper.flat
        )
        dense_swap_rows_unpadded = tuple(
            int32(row) for swap in dense_swaps for row in swap
        )
        dense_swap_rows = dense_swap_rows_unpadded or (
            int32(0),
            int32(0),
        )
        dense_swap_count = int32(len(dense_swaps))
        # Replace streaming accumulation with direct assignment when
        # stage matches b or b_hat row in coupling matrix.
        accumulates_output = tableau.accumulates_output
        accumulates_error = tableau.accumulates_error
        b_row = tableau.b_matches_a_row
        b_hat_row = tableau.b_hat_matches_a_row
        if b_row is not None:
            b_row = int32(b_row)
        if b_hat_row is not None:
            b_hat_row = int32(b_hat_row)

        ends_at_one = stage_time_fractions[-1] == numba_precision(1.0)

        # Get allocators from buffer registry
        getalloc = buffer_registry.get_allocator
        alloc_stage_increment = getalloc("stage_increment", self)
        alloc_stage_driver_stack = getalloc("stage_driver_stack", self)
        alloc_stage_state = getalloc("stage_state", self)
        alloc_previous_step_size = getalloc("previous_step_size", self)

        # Re-register the solver child under the same name as
        # register_buffers so the size snapshot reflects the solver's
        # fully built buffer group (the instance owns the group; the
        # compiled device function does not).
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(
                self, self.solver, name="solver"
            )
        )

        # no cover: start
        @cuda.jit(
            # (
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     numba_precision,
            #     numba_precision,
            #     int32,
            #     int32,
            #     numba_precision[::1],
            #     numba_precision[::1],
            #     int32[::1],
            # ),
            device=True,
            inline=True,
            **self.jit_kwargs,
        )
        def step(
            state,
            proposed_state,
            parameters,
            driver_coeffs,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
            counters,
        ):
            # ----------------------------------------------------------- #
            # Selective allocation from local or shared memory
            # ----------------------------------------------------------- #
            stage_state = alloc_stage_state(shared, persistent_local)
            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(
                shared, persistent_local
            )
            stage_increment = alloc_stage_increment(shared, persistent_local)
            previous_step_size = alloc_previous_step_size(
                shared, persistent_local
            )
            stage_driver_stack = alloc_stage_driver_stack(
                shared, persistent_local
            )

            # ----------------------------------------------------------- #

            current_time = time_scalar
            end_time = current_time + dt_scalar
            status_code = success

            if enable_dense_prediction:
                previous_dt = previous_step_size[0]
                apply_dense_prediction = (
                    not first_step_flag
                    and accepted_flag != int32(0)
                    and dt_scalar == previous_dt
                )
                previous_step_size[0] = dt_scalar

                if stage_count == int32(2):
                    for state_idx in range(n):
                        old_stage_0 = stage_increment[state_idx]
                        old_stage_1 = stage_increment[n + state_idx]
                        predicted_stage_0 = (
                            dense_coefficients[0] * old_stage_0
                            + dense_coefficients[1] * old_stage_1
                        )
                        predicted_stage_1 = (
                            dense_coefficients[2] * old_stage_0
                            + dense_coefficients[3] * old_stage_1
                        )
                        stage_increment[state_idx] = selp(
                            first_step_flag,
                            typed_zero,
                            selp(
                                apply_dense_prediction,
                                predicted_stage_0,
                                old_stage_0,
                            ),
                        )
                        stage_increment[n + state_idx] = selp(
                            first_step_flag,
                            typed_zero,
                            selp(
                                apply_dense_prediction,
                                predicted_stage_1,
                                old_stage_1,
                            ),
                        )
                elif stage_count == int32(3):
                    for state_idx in range(n):
                        old_stage_0 = stage_increment[state_idx]
                        old_stage_1 = stage_increment[n + state_idx]
                        old_stage_2 = stage_increment[2 * n + state_idx]
                        predicted_stage_0 = (
                            dense_coefficients[0] * old_stage_0
                            + dense_coefficients[1] * old_stage_1
                            + dense_coefficients[2] * old_stage_2
                        )
                        predicted_stage_1 = (
                            dense_coefficients[3] * old_stage_0
                            + dense_coefficients[4] * old_stage_1
                            + dense_coefficients[5] * old_stage_2
                        )
                        predicted_stage_2 = (
                            dense_coefficients[6] * old_stage_0
                            + dense_coefficients[7] * old_stage_1
                            + dense_coefficients[8] * old_stage_2
                        )
                        stage_increment[state_idx] = selp(
                            first_step_flag,
                            typed_zero,
                            selp(
                                apply_dense_prediction,
                                predicted_stage_0,
                                old_stage_0,
                            ),
                        )
                        stage_increment[n + state_idx] = selp(
                            first_step_flag,
                            typed_zero,
                            selp(
                                apply_dense_prediction,
                                predicted_stage_1,
                                old_stage_1,
                            ),
                        )
                        stage_increment[2 * n + state_idx] = selp(
                            first_step_flag,
                            typed_zero,
                            selp(
                                apply_dense_prediction,
                                predicted_stage_2,
                                old_stage_2,
                            ),
                        )
                else:
                    for state_idx in range(n):
                        for stage_idx in range(stage_count):
                            predicted = typed_zero
                            coefficient_base = stage_idx * stage_count
                            for source_idx in range(stage_idx, stage_count):
                                predicted += (
                                    dense_upper_coefficients[
                                        coefficient_base + source_idx
                                    ]
                                    * stage_increment[
                                        source_idx * n + state_idx
                                    ]
                                )
                            target_index = stage_idx * n + state_idx
                            stage_increment[target_index] = selp(
                                apply_dense_prediction,
                                predicted,
                                stage_increment[target_index],
                            )

                        for reverse_idx in range(stage_count):
                            stage_idx = stage_count - int32(1) - reverse_idx
                            target_index = stage_idx * n + state_idx
                            predicted = stage_increment[target_index]
                            coefficient_base = stage_idx * stage_count
                            for source_idx in range(stage_idx):
                                predicted += (
                                    dense_lower_coefficients[
                                        coefficient_base + source_idx
                                    ]
                                    * stage_increment[
                                        source_idx * n + state_idx
                                    ]
                                )
                            stage_increment[target_index] = selp(
                                apply_dense_prediction,
                                predicted,
                                stage_increment[target_index],
                            )

                        for swap_idx in range(dense_swap_count):
                            swap_base = int32(2) * swap_idx
                            left_stage = dense_swap_rows[swap_base]
                            right_stage = dense_swap_rows[
                                swap_base + int32(1)
                            ]
                            left_index = left_stage * n + state_idx
                            right_index = right_stage * n + state_idx
                            left_value = stage_increment[left_index]
                            right_value = stage_increment[right_index]
                            stage_increment[left_index] = selp(
                                apply_dense_prediction,
                                right_value,
                                left_value,
                            )
                            stage_increment[right_index] = selp(
                                apply_dense_prediction,
                                left_value,
                                right_value,
                            )

                        for stage_idx in range(stage_count):
                            target_index = stage_idx * n + state_idx
                            stage_increment[target_index] = selp(
                                first_step_flag,
                                typed_zero,
                                stage_increment[target_index],
                            )

            for idx in range(n):
                if accumulates_output:
                    proposed_state[idx] = state[idx]
                if has_error and accumulates_error:
                    error[idx] = typed_zero

            # Fill stage_drivers_stack if driver arrays provided
            if has_evaluate_driver_at_t:
                for stage_idx in range(stage_count):
                    stage_time = (
                        current_time
                        + dt_scalar * stage_time_fractions[stage_idx]
                    )
                    driver_offset = stage_idx * n_drivers
                    driver_slice = stage_driver_stack[
                        driver_offset : driver_offset + n_drivers
                    ]
                    evaluate_driver_at_t(
                        stage_time, driver_coeffs, driver_slice
                    )

            # Solve n-stage nonlinear problem for all stages
            solver_status = nonlinear_solver(
                stage_increment,
                parameters,
                stage_driver_stack,
                current_time,
                dt_scalar,
                typed_zero,
                state,
                state,
                solver_shared,
                solver_persistent,
                counters,
            )
            status_code = int32(status_code | solver_status)

            for stage_idx in range(stage_count):
                if has_evaluate_driver_at_t:
                    stage_base = stage_idx * n_drivers
                    for idx in range(n_drivers):
                        proposed_drivers[idx] = stage_driver_stack[
                            stage_base + idx
                        ]

                for idx in range(n):
                    value = state[idx]
                    for contrib_idx in range(stage_count):
                        flat_idx = stage_idx * stage_count + contrib_idx
                        increment_idx = contrib_idx * n
                        coeff = stage_rhs_coeffs[flat_idx]
                        if coeff != typed_zero:
                            value += (
                                coeff * stage_increment[increment_idx + idx]
                            )
                    stage_state[idx] = value

                # Capture precalculated outputs if tableau allows
                if not accumulates_output:
                    if b_row == stage_idx:
                        for idx in range(n):
                            proposed_state[idx] = stage_state[idx]
                if not accumulates_error:
                    if b_hat_row == stage_idx:
                        for idx in range(n):
                            error[idx] = stage_state[idx]

            # Kahan summation to reduce floating point errors
            # see https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            if accumulates_output:
                for idx in range(n):
                    solution_acc = typed_zero
                    compensation = typed_zero
                    for stage_idx in range(stage_count):
                        increment_value = stage_increment[stage_idx * n + idx]
                        weighted = (
                            solution_weights[stage_idx] * increment_value
                        )
                        term = weighted - compensation
                        temp = solution_acc + term
                        compensation = (temp - solution_acc) - term
                        solution_acc = temp
                    proposed_state[idx] = state[idx] + solution_acc

            if has_error and accumulates_error:
                # Standard accumulation path for error
                for idx in range(n):
                    error_acc = typed_zero
                    compensation = typed_zero
                    for stage_idx in range(stage_count):
                        increment_value = stage_increment[stage_idx * n + idx]
                        weighted = error_weights[stage_idx] * increment_value
                        term = weighted - compensation
                        temp = error_acc + term
                        compensation = (temp - error_acc) - term
                        error_acc = temp
                    error[idx] = error_acc

            if not ends_at_one:
                if has_evaluate_driver_at_t:
                    evaluate_driver_at_t(
                        end_time,
                        driver_coeffs,
                        proposed_drivers,
                    )

            evaluate_observables(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

            if not accumulates_error:
                for idx in range(n):
                    error[idx] = proposed_state[idx] - error[idx]

            return status_code

        # no cover: end
        return StepCache(step=step, nonlinear_solver=nonlinear_solver)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` as the method has multiple stages."""

        return self.stage_count > 1

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` when the tableau supplies an error estimate."""

        return self.tableau.has_error_estimate

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""
        return self.compile_settings.stage_count

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` because the method solves nonlinear systems."""
        return True

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""
        return self.tableau.order

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""
        return 1
