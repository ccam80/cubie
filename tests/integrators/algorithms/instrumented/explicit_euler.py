"""Explicit Euler step implementation."""

from typing import Callable, Optional

from numba import cuda, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import (
    StepCache,
    StepControlDefaults,
)
from cubie.integrators.algorithms.ode_explicitstep import (
    ExplicitStepConfig,
    ODEExplicitStep,
)

EE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)

class InstrumentedExplicitEulerStep(ODEExplicitStep):
    """Forward Euler integration step for explicit ODE updates."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """Initialise the explicit Euler step configuration.

        Parameters
        ----------
        precision
            Precision applied to device buffers.
        n
            Number of state entries advanced per step.
        dxdt_function
            Device derivative function evaluating ``dx/dt``.
        observables_function
            Device function computing system observables.
        driver_function
            Optional device function evaluating spline drivers at arbitrary
            times.
        get_solver_helper_fn
            Present for interface parity with implicit steps and ignored here.
        """
        config = ExplicitStepConfig(
            precision=precision,
            dxdt_function=dxdt_function,
            observables_function=observables_function,
            driver_function=driver_function,
            n=n,
        )

        super().__init__(config, EE_DEFAULTS.copy())

    def build_step(
        self,
        dxdt_function: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:
        """Build the device function for an explicit Euler step.

        Parameters
        ----------
        dxdt_function
            Device derivative function used within the update.
        observables_function
            Device function computing system observables.
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        numba_precision
            Numba precision corresponding to the configured precision.
        n
            Dimension of the state vector.
        n_drivers
            Number of driver signals provided to the system.

        Returns
        -------
        StepCache
            Container holding the compiled step function.
        """

        has_driver_function = driver_function is not None
        n = int32(n)

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
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, :, ::1],
            #     numba_precision[:, ::1],
            #     numba_precision[:, :, ::1],
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
            error,  # Non-adaptive algorithms receive a zero-length slice.
            residuals,
            jacobian_updates,
            stage_states,
            stage_derivatives,
            stage_observables,
            stage_drivers,
            stage_increments,
            newton_initial_guesses,
            newton_iteration_guesses,
            newton_residuals,
            newton_squared_norms,
            newton_iteration_scale,
            linear_initial_guesses,
            linear_iteration_guesses,
            linear_residuals,
            linear_squared_norms,
            linear_preconditioned_vectors,
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
            counters,
        ):
            """Advance the state with a single explicit Euler update.

            Parameters
            ----------
            state
                Device array storing the current state.
            proposed_state
                Device array receiving the updated state.
            parameters
                Device array of static model parameters.
            driver_coefficients
                Device array containing spline driver coefficients.
            drivers_buffer
                Device array of time-dependent drivers.
            proposed_drivers
                Device array receiving proposed driver samples.
            observables
                Device array storing accepted observable outputs.
            proposed_observables
                Device array receiving proposed observable outputs.
            error
                Device array reserved for error estimates. Non-adaptive
                algorithms receive a zero-length slice that can be reused as
                scratch.
            dt_scalar
                Scalar containing the proposed step size.
            time_scalar
                Scalar containing the current simulation time.
            shared
                Device array providing shared scratch buffers.
            persistent_local
                Device array for persistent local storage (unused here).

            Returns
            -------
            int
                Status code indicating successful completion.
            """
            typed_zero = numba_precision(0.0)

            # LOGGING: Record stage 0 initial state
            for idx in range(n):
                residuals[0, idx] = typed_zero
                jacobian_updates[0, idx] = typed_zero
                stage_states[0, idx] = state[idx]
                stage_increments[0, idx] = typed_zero
            for obs_idx in range(proposed_observables.shape[0]):
                stage_observables[0, obs_idx] = observables[obs_idx]
            for driver_idx in range(stage_drivers.shape[1]):
                stage_drivers[0, driver_idx] = drivers_buffer[driver_idx]

            # error buffer unused; stage dx/dt in proposed_state instead.
            dxdt_buffer = proposed_state
            dxdt_function(
                state,
                parameters,
                drivers_buffer,
                observables,
                dxdt_buffer,
                time_scalar,
            )

            # LOGGING: Record stage 0 derivatives and increments
            for idx in range(n):
                stage_derivatives[0, idx] = dxdt_buffer[idx]
                stage_increments[0, idx] = dt_scalar * dxdt_buffer[idx]

            for i in range(n):
                proposed_state[i] = state[i] + dt_scalar * dxdt_buffer[i]

            next_time = time_scalar + dt_scalar
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )
            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )
            return int32(0)
        # no cover: end
        
        return StepCache(step=step, nonlinear_solver=None)

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""
        return 1

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because explicit Euler is a single-stage method."""
        return False

    @property
    def is_adaptive(self) -> bool:
        """Return ``False`` because explicit Euler has no error estimator."""
        return False

    @property
    def order(self) -> int:
        """Return the classical order of the explicit Euler method."""
        return 1
