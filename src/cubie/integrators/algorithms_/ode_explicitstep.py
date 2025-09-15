from abc import abstractmethod
from typing import Callable

from numba import from_dtype
import attrs

from cubie import getype_validator, gttype_validator
from cubie.integrators.algorithms_.base_algorithm_step import (
    BaseAlgorithmStep, BaseStepConfig, StepCache)


@attrs.define
class ExplicitStepConfig(BaseStepConfig):
    """Configuration settings for explicit integration steps.

    Explicit algorithms do not access the full range of fields.
    """
    fixed_step_size: float = attrs.field(
            default=1e-3,
            validator=gttype_validator(float, 0)
    )


class ODEExplicitStep(BaseAlgorithmStep):

    def build(self) -> StepCache:
        """Create the cached step function for explicit algorithms."""

        config = self.compile_settings
        dxdt_function = config.dxdt_function
        numba_precision = from_dtype(config.precision)
        n = config.n
        fixed_step_size = config.fixed_step_size
        return self.build_step(
            dxdt_function, numba_precision, n, fixed_step_size
        )

    @abstractmethod
    def build_step(
        self,
        dxdt_function: Callable,
        numba_precision: type,
        n: int,
        fixed_step_size: float,
    ) -> StepCache:
        raise NotImplementedError

    @property
    def is_implicit(self) -> bool:
        return False