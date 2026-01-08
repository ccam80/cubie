"""Infrastructure for explicit integration step implementations."""

from abc import abstractmethod
from typing import Callable, Optional

from attrs import define

from cubie.integrators.algorithms.base_algorithm_step import (
    BaseAlgorithmStep,
    BaseStepConfig,
    StepCache,
)


@define
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
        evaluate_f = config.evaluate_f
        numba_precision = config.numba_precision
        n = config.n
        evaluate_observables = config.evaluate_observables
        evaluate_driver_at_t = config.evaluate_driver_at_t
        n_drivers = config.n_drivers
        return self.build_step(
            evaluate_f,
            evaluate_observables,
            evaluate_driver_at_t,
            numba_precision,
            n,
            n_drivers,
        )

    @abstractmethod
    def build_step(
        self,
        evaluate_f: Callable,
        evaluate_observables: Callable,
        evaluate_driver_at_t: Optional[Callable],
        numba_precision: type,
        n: int,
        n_drivers: int,
    ) -> StepCache:
        """Build and return the explicit step device function.

        Parameters
        ----------
        evaluate_f
            Device function for evaluating the ODE right-hand side f(t, y).
        evaluate_observables
            Device helper that computes observables for the system.
        evaluate_driver_at_t
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
