"""Infrastructure for explicit integration step implementations."""

from abc import abstractmethod
from typing import Callable, Optional

import attrs

from cubie._utils import gttype_validator
from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
)


@attrs.define
class ExplicitStepConfig(BaseStepConfig):
    """Configuration settings for explicit integration steps.

    Parameters
    ----------
    dt
        Fixed step size applied by the explicit integrator.
    """
    dt: float = attrs.field(
            default=1e-3,
            validator=gttype_validator(float, 0)
    )

    @property
    def settings_dict(self) -> dict:
        """Return configuration fields as a dictionary."""
        settings_dict = super().settings_dict
        settings_dict.update({'dt': self.dt})
        return settings_dict


class ODEExplicitStep(BaseAlgorithmStep):
    """Base helper for explicit integration algorithms."""

    def build(self) -> StepCache:
        """Create and cache the device function for the explicit algorithm.

        Returns
        -------
        StepCache
            Container with the compiled step device function.
        """

        config = self.compile_settings
        dxdt_function = config.dxdt_function
        numba_precision = config.numba_precision
        n = config.n
        observables_function = config.observables_function
        driver_function = config.driver_function
        dt = config.dt
        n_drivers = config.n_drivers
        return self.build_step(
            dxdt_function,
            observables_function,
            driver_function,
            numba_precision,
            n,
            dt,
            n_drivers,
        )

    @abstractmethod
    def build_step(
        self,
        dxdt_function: Callable,
        observables_function: Callable,
        driver_function: Optional[Callable],
        numba_precision: type,
        n: int,
        dt: Optional[float],
        n_drivers: int,
    ) -> StepCache:
        """Build and return the explicit step device function.

        Parameters
        ----------
        dxdt_function
            Device derivative function for the ODE system.
        observables_function
            Device helper that computes observables for the system.
        driver_function
            Optional device function evaluating drivers at arbitrary times.
        numba_precision
            Numba precision for compiled device buffers.
        n
            Dimension of the state vector.
        dt
            Fixed step size applied by the integrator.
        n_drivers
            Number of driver signals provided to the system.

        Returns
        -------
        StepCache
            Container holding the device step implementation.
        """
        raise NotImplementedError

    @property
    def is_implicit(self) -> bool:
        """Return ``False`` to indicate the algorithm is explicit."""
        return False

    @property
    def dt(self) -> float:
        """Return the configured explicit step size."""
        return self.compile_settings.dt

    @property
    def algorithm_shared_elements(self) -> int:
        """Explicit base class does not reserve shared scratch."""

        return 0

    @property
    def algorithm_local_elements(self) -> int:
        """Explicit base class does not reserve persistent locals."""

        return 0
