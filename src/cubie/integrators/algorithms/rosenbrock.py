"""Rosenbrock-W integration step using a streamed accumulator layout."""

from typing import Callable, Optional, Sequence, Tuple

import attrs
import numpy as np
from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_implicitstep import (ImplicitStepConfig,
                                                 ODEImplicitStep)
from cubie.integrators.matrix_free_solvers import linear_solver_factory


@attrs.define(frozen=True)
class RosenbrockTableau:
    """Coefficient tableau describing a Rosenbrock-W method."""

    a: Tuple[Tuple[float, ...], ...]
    C: Tuple[Tuple[float, ...], ...]
    b: Tuple[float, ...]
    d: Tuple[float, ...]
    c: Tuple[float, ...]
    gamma: float

    @property
    def stage_count(self) -> int:
        """Return the number of stages described by the tableau."""

        return len(self.b)

    def typed_rows(
        self,
        rows: Sequence[Sequence[float]],
        numba_precision: type,
    ) -> Tuple[Tuple[float, ...], ...]:
        """Pad and convert tableau rows to the requested precision."""

        typed_rows = []
        for row in rows:
            padded = list(row)
            if len(padded) < self.stage_count:
                padded.extend([0.0] * (self.stage_count - len(padded)))
            typed_rows.append(
                tuple(numba_precision(value) for value in padded)
            )
        return tuple(typed_rows)

    def build_combined_successors(
        self,
        stage_rows: Tuple[Tuple[float, ...], ...],
        jacobian_rows: Tuple[Tuple[float, ...], ...],
    ) -> Tuple[Tuple[Tuple[int, float, float], ...], ...]:
        """Return successor contributions for state and Jacobian accumulators."""

        successors = []
        for stage_index in range(self.stage_count):
            stage_successors = []
            for successor_index in range(stage_index + 1, self.stage_count):
                stage_coeff = stage_rows[successor_index][stage_index]
                jacobian_coeff = jacobian_rows[successor_index][stage_index]
                if stage_coeff != 0.0 or jacobian_coeff != 0.0:
                    stage_successors.append(
                        (successor_index, stage_coeff, jacobian_coeff)
                    )
            successors.append(tuple(stage_successors))
        return tuple(successors)



ROSENBROCK_W6S4OS_TABLEAU = RosenbrockTableau(
    a=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            0.5812383407115008,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.9039624413714670,
            1.8615191555345010,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.0765797196750000,
            0.1884255381414796,
            1.8701589674910320,
            0.0,
            0.0,
            0.0,
        ),
        (
            4.4355506384843120,
            5.4571817986101890,
            4.6163507880689300,
            3.1181119524023610,
            0.0,
            0.0,
        ),
        (
            10.791701698483260,
            -10.056915225841310,
            14.995644854284190,
            5.2743399543909430,
            1.4297308712611900,
            0.0,
        ),
    ),
    C=(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            -2.661294105131369,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -3.128450202373838,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -6.920335474535658,
            -1.202675288266817,
            -9.733561811413620,
            0.0,
            0.0,
            0.0,
        ),
        (
            -28.095306291026950,
            20.371262954793770,
            -41.043752753028690,
            -19.663731756208950,
            0.0,
            0.0,
        ),
        (
            9.7998186780974000,
            11.935792886603180,
            3.6738749290132010,
            14.807828541095500,
            0.8318583998690680,
            0.0,
        ),
    ),
    b=(
        6.4562170746532350,
        -4.8531413177680530,
        9.7653183340692600,
        2.0810841772787230,
        0.6603936866352417,
        0.6000000000000000,
    ),
    d=(
        0.2500000000000000,
        0.0836691184292894,
        0.0544718623516351,
        -0.3402289722355864,
        0.0337651588339529,
        -0.0903074267618540,
    ),
    c=(
        0.0,
        0.1453095851778752,
        0.3817422770256738,
        0.6367813704374599,
        0.7560744496323561,
        0.9271047239875670,
    ),
    gamma=0.25,
)


ROSENBROCK_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pi",
        "order": 4,
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        "kp": 0.6,
        "kd": 0.4,
        "deadband_min": 1.0,
        "deadband_max": 1.1,
        "min_gain": 0.5,
        "max_gain": 2.0,
    }
)


@attrs.define
class RosenbrockStepConfig(ImplicitStepConfig):
    """Configuration describing the Rosenbrock-W integrator."""

    tableau: RosenbrockTableau = attrs.field(default=ROSENBROCK_W6S4OS_TABLEAU)

    @property
    def settings_dict(self) -> dict:
        """Return configuration values as a dictionary."""

        settings = super().settings_dict
        settings.update(
            {
                "tableau": self.tableau,
            }
        )
        return settings


class RosenbrockStep(ODEImplicitStep):
    """Rosenbrock-W step with an embedded error estimate."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: Optional[float],
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        preconditioner_order: int = 1,
        krylov_tolerance: float = 1e-6,
        max_linear_iters: int = 200,
        linear_correction_type: str = "minimal_residual",
        tableau: RosenbrockTableau = ROSENBROCK_W6S4OS_TABLEAU,
    ) -> None:
        """Initialise the Rosenbrock-W step configuration."""

        mass = np.eye(n, dtype=precision)
        config = RosenbrockStepConfig(
            precision=precision,
            n=n,
            dt=dt,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            get_solver_helper_fn=get_solver_helper_fn,
            preconditioner_order=preconditioner_order,
            krylov_tolerance=krylov_tolerance,
            max_linear_iters=max_linear_iters,
            linear_correction_type=linear_correction_type,
            tableau=tableau,
            beta=1.0,
            gamma=tableau.gamma,
            M=mass,
        )
        super().__init__(config, ROSENBROCK_DEFAULTS)

    def build_implicit_helpers(self) -> Tuple[Callable, Callable]:
        """Construct the nonlinear solver chain used by implicit methods.

        Returns
        -------
        tuple of Callables
            Linear solver function and jvp compiled for the Rosenbrock-W step.
        """
        precision = self.precision
        config = self.compile_settings
        beta = config.beta
        gamma = config.tableau.gamma
        mass = config.M
        preconditioner_order = config.preconditioner_order
        n = config.n

        get_fn = config.get_solver_helper_fn

        preconditioner = get_fn(
            "neumann_preconditioner",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        linear_operator = get_fn(
            "linear_operator",
            beta=beta,
            gamma=gamma,
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        jacobian_operator = get_fn(
            "linear_operator",
            beta=precision(0.0),
            gamma=precision(1.0),
            mass=mass,
            preconditioner_order=preconditioner_order,
        )

        krylov_tolerance = config.krylov_tolerance
        max_linear_iters = config.max_linear_iters
        correction_type = config.linear_correction_type

        linear_solver = linear_solver_factory(
            linear_operator,
            n=n,
            preconditioner=preconditioner,
            correction_type=correction_type,
            tolerance=krylov_tolerance,
            max_iters=max_linear_iters,
        )

        return linear_solver, jacobian_operator

    def build_step(
        self,
        solver_fn: Callable,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        dt: Optional[float],
    ) -> StepCache:  # pragma: no cover - device function
        """Compile the Rosenbrock-W device step."""

        config = self.compile_settings
        tableau = config.tableau
        linear_solver, jacobian_apply = self.build_implicit_helpers()

        stage_count = tableau.stage_count
        has_driver_function = driver_function is not None

        stage_rhs_coefficients = self.tableau.typed_rows(
            tableau.a, numba_precision
        )
        jacobian_update_coefficients = self.tableau.typed_rows(
            tableau.C, numba_precision
        )
        solution_weights = tuple(
            numba_precision(value) for value in tableau.b
        )
        error_weights = tuple(numba_precision(value) for value in tableau.d)
        stage_time_fractions = tuple(
            numba_precision(value) for value in tableau.c
        )
        stage_offsets = tuple(index * n for index in range(stage_count))

        # Keep coefficients in separate structures and iterate successors by index
        # to avoid unpacking mixed tuples inside device code.
        typed_zero = numba_precision(0.0)

        @cuda.jit(
            (
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:, :, :],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision[:],
                numba_precision,
                numba_precision,
                numba_precision[:],
                numba_precision[:],
            ),
            device=True,
            inline=True,
        )
        def step(
            state,
            proposed_state,
            parameters,
            driver_coefficients,
            drivers_buffer,
            proposed_drivers,
            observables,
            proposed_observables,
            error,
            dt_scalar,
            time_scalar,
            shared,
            persistent_local,
        ):
            stage_rhs = proposed_state # reuse state proposal buffer for rhs
            stage_state = cuda.local.array(n, numba_precision)
            stage_increment = cuda.local.array(n, numba_precision)
            jacobian_stage_product = cuda.local.array(n, numba_precision)

            dt_value = dt_scalar
            current_time = time_scalar
            end_time = current_time + dt_value

            stage_accumulator = shared[: stage_count * n]
            jacobian_product_accumulator = shared[
                stage_count * n : 2 * stage_count * n
            ]

            for idx in range(stage_count * n):
                stage_accumulator[idx] = typed_zero
                jacobian_product_accumulator[idx] = typed_zero

            for idx in range(n):
                error[idx] = typed_zero
                stage_state[idx] = state[idx]

            for idx in range(proposed_observables.size):
                proposed_observables[idx] = observables[idx]

            for idx in range(drivers_buffer.size):
                proposed_drivers[idx] = drivers_buffer[idx]

            status_code = int32(0)

            for stage_index in range(stage_count):

                stage_offset = stage_offsets[stage_index]
                stage_time = (
                    current_time + dt_value * stage_time_fractions[stage_index]
                )

                if stage_index > 0:
                    for idx in range(n):
                        stage_state[idx] = (
                            state[idx] + stage_accumulator[stage_offset + idx]
                        )
                        if has_driver_function:
                            driver_function(
                                    stage_time,
                                    driver_coefficients,
                                    proposed_drivers,
                            )
                        proposed_observables = observables_function(
                                stage_state,
                                parameters,
                                proposed_drivers,
                                proposed_observables,
                                stage_time,
                        )

                dxdt_fn(
                    stage_state,
                    parameters,
                    proposed_drivers,
                    proposed_observables,
                    stage_rhs,
                    stage_time,
                )

                for idx in range(n):
                    rhs_value = stage_rhs[idx]
                    jacobian_term = jacobian_product_accumulator[
                        stage_offset + idx
                    ]
                    stage_increment[idx] = typed_zero
                    stage_rhs[idx] = dt_value * (rhs_value - jacobian_term)

                status_code |= linear_solver(
                    state,
                    parameters,
                    drivers_buffer,
                    dt_value,
                    stage_rhs,
                    stage_increment,
                )

                solution_weight = solution_weights[stage_index]
                error_weight = error_weights[stage_index]
                for idx in range(n):
                    increment = stage_increment[idx]
                    proposed_state[idx] += solution_weight * increment
                    error[idx] += error_weight * increment

                # Determine if there are any non-zero successor coefficients for this stage
                has_successor = False
                for successor_index in range(stage_index + 1, stage_count):
                    if (
                        stage_rhs_coefficients[successor_index][stage_index] != 0.0
                        or jacobian_update_coefficients[successor_index][stage_index] != 0.0
                    ):
                        has_successor = True

                if has_successor:
                    # Cache the Jacobian action for this stage increment once
                    # and stream weighted contributions to future stages.
                    jacobian_apply(
                        state,
                        parameters,
                        drivers_buffer,
                        numba_precision(1.0),
                        stage_increment,
                        jacobian_stage_product,
                    )

                    for successor_index in range(stage_index + 1, stage_count):
                        state_coeff = stage_rhs_coefficients[successor_index][stage_index]
                        jac_coeff = jacobian_update_coefficients[successor_index][stage_index]
                        if state_coeff == 0.0 and jac_coeff == 0.0:
                            continue
                        base = stage_offsets[successor_index]
                        for idx in range(n):
                            stage_contribution = (
                                state_coeff * stage_increment[idx]
                            )
                            jacobian_contribution = (
                                jac_coeff * jacobian_stage_product[idx]
                            )
                            stage_accumulator[base + idx] += stage_contribution
                            jacobian_product_accumulator[
                                base + idx
                            ] += jacobian_contribution

            final_time = end_time

            if has_driver_function:
                driver_function(
                    final_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                final_time,
            )
            for idx in range(n):
                proposed_state[idx] = state[idx] + proposed_state[idx]
            return status_code

        return StepCache(step=step)

    @property
    def is_multistage(self) -> bool:
        """Return ``True`` as the method has multiple stages."""

        return True

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` because an embedded error estimate is produced."""

        return True

    @property
    def shared_memory_required(self) -> int:
        """Return the number of precision entries required in shared memory."""

        stage_count = self.compile_settings.tableau.stage_count
        return 2 * stage_count * self.compile_settings.n

    @property
    def local_scratch_required(self) -> int:
        """Return the number of local precision entries required."""

        return 4 * self.compile_settings.n

    @property
    def persistent_local_required(self) -> int:
        """Return the number of persistent local entries required."""

        return 0

    @property
    def is_implicit(self) -> bool:
        """Return ``True`` because the method solves linear systems."""

        return True

    @property
    def order(self) -> int:
        """Return the classical order of accuracy."""

        return 4

    @property
    def threads_per_step(self) -> int:
        """Return the number of CUDA threads that advance one state."""

        return 1

    @property
    def tableau(self) -> RosenbrockTableau:
        """Return the tableau used by the integrator."""

        return self.compile_settings.tableau

