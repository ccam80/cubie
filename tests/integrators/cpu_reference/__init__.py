"""Reference CPU implementations used across integrator tests.

I've let genAI agents run fairly free on this module, adding many of the
over-engineered and pointless checks and complicated chains that it loves to
add, as all we really want in here is a reference implementation of the
GPU integrator components."""

from .algorithms import (
    get_ref_step_factory,
    get_ref_stepper,
)
from .cpu_ode_system import CPUODESystem
from .cpu_utils import (
    Array,
    DriverEvaluator,
    InstrumentedStepResult,
    STATUS_MASK,
    StepResult,
    StepResultLike,
    make_step_result,
)
from .loops import _collect_saved_outputs, run_reference_loop
from .step_controllers import CPUAdaptiveController

__all__ = [
    "Array",
    "CPUAdaptiveController",
    "CPUODESystem",
    "DriverEvaluator",
    "InstrumentedStepResult",
    "STATUS_MASK",
    "StepResult",
    "StepResultLike",
    "make_step_result",
    "_collect_saved_outputs",
    "get_ref_step_factory",
    "get_ref_stepper",
    "run_reference_loop",
]
