"""Backward Euler step implementation using Newton–Krylov."""

from typing import Callable, Optional

import attrs
from numba import cuda, int32
import numpy as np

from cubie._utils import PrecisionDType, build_config
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    StepControlDefaults
from cubie.integrators.algorithms.ode_implicitstep import (
    ImplicitStepConfig, ODEImplicitStep
)


@attrs.define
class BackwardsEulerStepConfig(ImplicitStepConfig):
    """Configuration for Backwards Euler step with buffer location."""

    increment_cache_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )


ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0,
                  'M': np.eye}

BE_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "fixed",
        "dt": 1e-3,
    }
)

class BackwardsEulerStep(ODEImplicitStep):
    """Backward Euler step solved with matrix-free Newton–Krylov."""

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
        """Initialise the backward Euler step configuration.

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
            Optional device function evaluating drivers at arbitrary times.
        get_solver_helper_fn
            Callable returning device helpers used by the nonlinear solver.
        **kwargs
            Optional parameters passed to config classes. See
            BackwardsEulerStepConfig, ImplicitStepConfig, and solver config
            classes for available parameters. None values are ignored.
        """
        beta = ALGO_CONSTANTS['beta']
        gamma = ALGO_CONSTANTS['gamma']
        M = ALGO_CONSTANTS['M'](n, dtype=precision)

        config = build_config(
            BackwardsEulerStepConfig,
            required={
                'precision': precision,
                'n': n,
                'dxdt_function': dxdt_function,
                'observables_function': observables_function,
                'driver_function': driver_function,
                'get_solver_helper_fn': get_solver_helper_fn,
                'beta': beta,
                'gamma': gamma,
                'M': M,
            },
            **kwargs
        )

        super().__init__(config, BE_DEFAULTS.copy(), **kwargs)

        self.register_buffers()


    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
        config = self.compile_settings

        # Register solver child buffers
        _ = buffer_registry.get_child_allocators(
            self, self.solver, name='solver_scratch'
        )

        # Register increment cache buffer
        buffer_registry.register(
            'increment_cache',
            self,
            config.n,
            config.increment_cache_location,
            persistent=True,
            precision=config.precision
        )

    def build_step(
        self,
        dxdt_fn: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        solver_function: Callable,
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:  # pragma: no cover - cuda code
        """Build the device function for a backward Euler step.

        Parameters
        ----------
        dxdt_fn
            Device derivative function for the ODE system.
        observables_function
            Device observable computation helper.
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        solver_function
            Device function for the Newton-Krylov nonlinear solver.
        numba_precision
            Numba precision corresponding to the configured precision.
        n
            Dimension of the state vector.
        n_drivers
            Number of driver signals provided to the system.

        Returns
        -------
        StepCache
            Container holding the compiled step function and solver.
        """
        a_ij = numba_precision(1.0)
        has_driver_function = driver_function is not None
        driver_function = driver_function
        n = int32(n)

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver_scratch')
        )

        # Get increment cache allocator from buffer_registry
        alloc_increment_cache = buffer_registry.get_allocator(
            'increment_cache', self
        )

        solver_fn = solver_function

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
            counters,
        ):
            """Perform one backward Euler update.

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
                Device array capturing solver diagnostics. Fixed-step
                algorithms receive a zero-length slice that can be repurposed
                as scratch when available.
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
                Status code returned by the nonlinear solver.
            """
            solver_scratch = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared, persistent_local)
            increment_cache = alloc_increment_cache(shared, persistent_local)

            for i in range(n):
                proposed_state[i] = increment_cache[i]

            next_time = time_scalar + dt_scalar
            if has_driver_function:
                driver_function(
                    next_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            status = solver_fn(
                proposed_state,
                parameters,
                proposed_drivers,
                next_time,
                dt_scalar,
                a_ij,
                state,
                solver_scratch,
                solver_persistent,
                counters,
            )

            for i in range(n):
                increment_cache[i] = proposed_state[i]
                proposed_state[i] += state[i]

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                next_time,
            )

            return status

        return StepCache(step=step, nonlinear_solver=solver_fn)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because backward Euler is a single-stage method."""

        return False

    @property
    def is_adaptive(self) -> bool:
        """Return ``False`` because backward Euler is fixed step."""

        return False

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def settings_dict(self) -> dict:
        """Return the configuration dictionary for the step."""

        return self.compile_settings.settings_dict

    @property
    def order(self) -> int:
        """Return the classical order of the backward Euler method."""
        return 1

    @property
    def dxdt_function(self) -> Optional[Callable]:
        """Return the derivative device function."""

        return self.compile_settings.dxdt_function

    @property
    def identifier(self) -> str:
        return "backwards_euler"
