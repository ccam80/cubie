"""Crank–Nicolson step with embedded backward Euler error estimation."""

from typing import Callable, Optional

import attrs
from numba import cuda, int32
import numpy as np

from cubie._utils import PrecisionDType, build_config
from cubie.buffer_registry import buffer_registry
from cubie.integrators.algorithms import ImplicitStepConfig
from cubie.integrators.algorithms.base_algorithm_step import StepCache, \
    StepControlDefaults
from cubie.integrators.algorithms.ode_implicitstep import ODEImplicitStep

ALGO_CONSTANTS = {'beta': 1.0,
                  'gamma': 1.0,
                  'M': np.eye}

CN_DEFAULTS = StepControlDefaults(
    step_controller={
        "step_controller": "pid",
        "dt_min": 1e-6,
        "dt_max": 1e-1,
        "kp": 0.7,
        "ki": -0.4,
        "deadband_min": 1.0,
        "deadband_max": 1.1,
        "min_gain": 0.5,
        "max_gain": 2.0,
    }
)


@attrs.define
class CrankNicolsonStepConfig(ImplicitStepConfig):
    """Configuration for Crank-Nicolson step."""

    dxdt_location: str = attrs.field(
        default='local',
        validator=attrs.validators.in_(['local', 'shared'])
    )


class CrankNicolsonStep(ODEImplicitStep):
    """Crank–Nicolson step with embedded backward Euler error estimation."""

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
        """Initialise the Crank–Nicolson step configuration.

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
            CrankNicolsonStepConfig, ImplicitStepConfig, and solver config
            classes for available parameters. None values are ignored.
        """
        beta = ALGO_CONSTANTS['beta']
        gamma = ALGO_CONSTANTS['gamma']
        M = ALGO_CONSTANTS['M'](n, dtype=precision)

        config = build_config(
            CrankNicolsonStepConfig,
            required={
                'precision': precision,
                'n': n,
                'get_solver_helper_fn': get_solver_helper_fn,
                'beta': beta,
                'gamma': gamma,
                'M': M,
                'dxdt_function': dxdt_function,
                'observables_function': observables_function,
                'driver_function': driver_function,
            },
            **kwargs
        )

        super().__init__(config, CN_DEFAULTS.copy(), **kwargs)

        self.register_buffers()

    def register_buffers(self) -> None:
        """Register buffers with buffer_registry."""
        config = self.compile_settings
        # Register solver child buffers

        _ = buffer_registry.get_child_allocators(
            self, self.solver, name='solver'
        )

        buffer_registry.register(
            'cn_dxdt',
            self,
            config.n,
            config.dxdt_location,
            aliases='solver_shared',
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
        """Build the device function for the Crank–Nicolson step.

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
        stage_coefficient = numba_precision(0.5)
        be_coefficient = numba_precision(1.0)
        has_driver_function = driver_function is not None
        n = int32(n)

        # Get child allocators for Newton solver
        alloc_solver_shared, alloc_solver_persistent = (
            buffer_registry.get_child_allocators(self, self.solver,
                                                 name='solver')
        )
        alloc_dxdt = buffer_registry.get_allocator('cn_dxdt', self)

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
            error,
            dt_scalar,
            time_scalar,
            first_step_flag,
            accepted_flag,
            shared,
            persistent_local,
            counters,
        ):
            """Advance the state using Crank–Nicolson with embedded error check.

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
                Device array capturing embedded error estimates.
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

            solver_shared = alloc_solver_shared(shared, persistent_local)
            solver_persistent = alloc_solver_persistent(shared, persistent_local)
            dxdt = alloc_dxdt(shared, persistent_local)

            # base_state aliases error as their lifetimes are disjoint
            base_state = error

            # Evaluate f(state)
            dxdt_fn(
                state,
                parameters,
                drivers_buffer,
                observables,
                dxdt,
                time_scalar,
            )

            half_dt = dt_scalar * numba_precision(0.5)
            end_time = time_scalar + dt_scalar

            # Form the Crank-Nicolson stage base
            for i in range(n):
                base_state[i] = state[i] + half_dt * dxdt[i]

            # Solve Crank-Nicolson step (main solution)
            if has_driver_function:
                driver_function(
                    end_time,
                    driver_coefficients,
                    proposed_drivers,
                )

            status = solver_function(
                proposed_state,
                parameters,
                proposed_drivers,
                end_time,
                dt_scalar,
                stage_coefficient,
                base_state,
                solver_shared,
                solver_persistent,
                counters,
            )

            for i in range(n):
                increment = proposed_state[i]
                proposed_state[i] = base_state[i] + stage_coefficient * increment
                base_state[i] = increment

            status |= solver_function(
                base_state,
                parameters,
                proposed_drivers,
                end_time,
                dt_scalar,
                be_coefficient,
                state,
                solver_shared,
                solver_persistent,
                counters,
            )

            # Compute error as difference between Crank-Nicolson and Backward Euler
            for i in range(n):
                error[i] = proposed_state[i] - (state[i] + base_state[i])

            observables_function(
                proposed_state,
                parameters,
                proposed_drivers,
                proposed_observables,
                end_time,
            )

            return status

        return StepCache(step=step, nonlinear_solver=solver_function)

    @property
    def is_multistage(self) -> bool:
        """Return ``False`` because Crank–Nicolson is a single-stage method."""

        return False

    @property
    def is_adaptive(self) -> bool:
        """Return ``True`` because the embedded error estimate enables adaptivity."""

        return True

    @property
    def threads_per_step(self) -> int:
        """Return the number of threads used per step."""

        return 1

    @property
    def order(self) -> int:
        """Return the classical order of the Crank–Nicolson method."""

        return 2
