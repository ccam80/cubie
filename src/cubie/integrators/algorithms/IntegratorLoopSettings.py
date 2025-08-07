"""
Integrator configuration management with validation and adapter patterns.
"""

from typing import Callable, Optional

from attrs import define, field, validators
from numpy import float32, float16, float64

from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.outputhandling.output_sizes import LoopBufferSizes
from cubie.integrators.algorithms.LoopStepConfig import \
    LoopStepConfig


@define
class IntegratorLoopSettings:
    """
    Compile-critical settings for the integrator loop, including timing and
    sizes. The integrator loop is not the
    source of truth for these settings, so minimal setters are provided. Instead, there are update_from methods which
    take in other updated objects and extract the relevant settings.
    """

    # Core system properties
    loop_step_config: LoopStepConfig = field(
            validator=validators.instance_of(LoopStepConfig))
    buffer_sizes: LoopBufferSizes = field(
            validator=validators.instance_of(LoopBufferSizes))
    precision: type = field(default=float32, validator=validators.and_(
            validators.instance_of(type),
            validators.in_([float32, float64, float16], ), ), )
    compile_flags: OutputCompileFlags = field(default=OutputCompileFlags(),
                                              validator=validators.instance_of(
                                                      OutputCompileFlags, ), )
    dxdt_function: Optional[Callable] = field(default=None)
    save_state_func: Optional[Callable] = field(default=None)
    update_summaries_func: Optional[Callable] = field(default=None)
    save_summaries_func: Optional[Callable] = field(default=None)

    @property
    def fixed_steps(self):
        """Return the fixed steps as a tuple of (save_every_samples, summarise_every_samples, step_size)."""
        return self.loop_step_config.fixed_steps

    @property
    def fixed_step_size(self) -> float:
        """Return the step size used in the loop."""
        return self.loop_step_config.fixed_steps[-1]

    @property
    def dt_min(self) -> float:
        return self.loop_step_config.dt_min

    @property
    def dt_max(self) -> float:
        return self.loop_step_config.dt_max

    @property
    def dt_save(self) -> float:
        return self.loop_step_config.dt_save

    @property
    def dt_summarise(self) -> float:
        return self.loop_step_config.dt_summarise

    @property
    def atol(self) -> float:
        return self.loop_step_config.atol

    @property
    def rtol(self) -> float:
        return self.loop_step_config.rtol

    @classmethod
    def from_integrator_run(cls, run_object):
        """Create an IntegratorLoopSettings instance from an SingleIntegratorRun object."""
        return cls(loop_step_config=run_object.loop_step_config,
                buffer_sizes=run_object.loop_buffer_sizes,
                precision=run_object.precision,
                dxdt_function=run_object.dxdt_function,
                save_state_func=run_object.save_state_func,
                update_summaries_func=run_object.update_summaries_func,
                save_summaries_func=run_object.save_summaries_func, )
