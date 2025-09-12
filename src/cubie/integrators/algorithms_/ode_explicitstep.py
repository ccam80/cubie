from numba import cuda
import attrs

from cubie.integrators.algorithms_.base_algorithm_step import BaseAlgorithmStep
from cubie.integrators.algorithms_.base_step_config import BaseStepConfig

@attrs.define
class ExplicitStepConfig(BaseStepConfig):
    """Configuration settings for a single explicit integration step.

    Explicit algorithms do not access the full range of fields.
    """
    @property
    def is_implicit(self):
        return False

class ODEExplicitStep(BaseAlgorithmStep):
    def build_step(self):
        @cuda.jit
        def explicit_step():
            pass