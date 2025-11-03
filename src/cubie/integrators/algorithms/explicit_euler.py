"""Explicit Euler step implementation."""

from typing import Callable, Optional

from numba import cuda, int16, int32

from cubie._utils import PrecisionDType
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    StepControlDefaults
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

class ExplicitEulerStep(ODEExplicitStep):
    """Forward Euler integration step for explicit ODE updates."""

    def __init__(
        self,
        precision: PrecisionDType,
        n: int,
        dt: float,
        dxdt_function: Optional[Callable] = None,
        observables_function: Optional[Callable] = None,
        driver_function: Optional[Callable] = None,
        get_solver_helper_fn: Optional[Callable] = None,
    ) -> None:
        """Initialise the explicit Euler step configuration.

        Parameters
        ----------
        precision
            Precision applied to device buffers.
        n
            Number of state entries advanced per step.
        dt
            Fixed step size used by the explicit update. Defaults to the
            controller value when ``None`` is supplied.
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
        if dt is None:
            dt = EE_DEFAULTS.step_controller['dt']

        config = ExplicitStepConfig(
            dt=dt,
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
        dt: float,
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
        dt
            Step size used for integration.

        Returns
        -------
        StepCache
            Container holding the compiled step function.
        """

        step_size = numba_precision(dt)
        has_driver_function = driver_function is not None
        driver_function = driver_function

        # no cover: start
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
                int16,
                int16,
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
            error,  # Non-adaptive algorithms receive a zero-length slice.
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
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
            for i in range(n):
                proposed_state[i] = state[i] + step_size * dxdt_buffer[i]

            next_time = time_scalar + step_size
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
    def shared_memory_required(self) -> int:
        """Shared memory usage expressed in precision-sized entries."""

        return super().shared_memory_required

    @property
    def local_scratch_required(self) -> int:
        """Local scratch usage expressed in precision-sized entries."""

        return 0

    @property
    def algorithm_shared_elements(self) -> int:
        """Explicit Euler does not reserve shared scratch."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Explicit Euler does not reserve persistent local storage."""

        return 0

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
