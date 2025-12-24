"""Infrastructure for explicit integration step implementations."""

from abc import abstractmethod
from typing import Callable, Optional

import attrs

from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
)


@attrs.define
class ExplicitStepConfig(BaseStepConfig):
    """Configuration settings for explicit ODE integration algorithms."""
    pass


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
        n_drivers = config.n_drivers
        return self.build_step(
            dxdt_function,
            observables_function,
            driver_function,
            numba_precision,
            n,
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
