"""
Integrator configuration management with validation and adapter patterns.
"""

from attrs import define, field, validators, Factory
from typing import Callable, Optional
from numba import float32, float64
from numba.types import Float as _numba_float_type
from warnings import warn
from CuMC.ForwardSim.OutputHandling.output_sizes import LoopBufferSizes


@define
class LoopStepConfig:
    """
    Step timing and exit conditions for an integrator loop. Convenience class for grouping and passing around loop
    step information.
    """
    dt_min: float = field(default=1e-6, validator=validators.instance_of(float))
    dt_max: float = field(default=1.0, validator=validators.instance_of(float))
    dt_save: float = field(default=0.1, validator=validators.instance_of(float))
    dt_summarise: float = field(default=0.1, validator=validators.instance_of(float))
    atol: float = field(default=1e-6, validator=validators.instance_of(float))
    rtol: float = field(default=1e-6, validator=validators.instance_of(float))


@define
class IntegratorLoopSettings:
    """
    Compile-critical settings for the integrator loop, including timing and sizes. The integrator loop is not the
    source of truth for these settings, so minimal setters are provided. Instead, there are update_from methods which
    take in other updated objects and extract the relevant settings.
    """

    # Core system properties
    loop_step_config: LoopStepConfig = field(validator=validators.instance_of(LoopStepConfig))
    buffer_sizes: LoopBufferSizes = field(validator=validators.instance_of(LoopBufferSizes))
    precision: _numba_float_type = field(default=float32, validator=validators.in_([float32, float64]))

    dxdt_func: Optional[Callable] = field(default=None)
    save_state_func: Optional[Callable] = field(default=None)
    update_summary_func: Optional[Callable] = field(default=None)
    save_summary_func: Optional[Callable] = field(default=None)

    @property
    def fixed_steps(self):
        """Fixed-step helper function: Convert time-based requests to integer numbers of steps at step_size (dt_min
        used by default in fixed-step loops). Sanity-check values and warn the user if they are adjusted.

        Returns:
            save_every_samples (int): The number of internal loop steps between saves.
            summarise_every_samples (int): The number of output samples between summary metric calculations.
            step_size (float): The internal time step size used in the loop (dt_min, by default).

        Raises:
            ValueError: If the user tries to save more often than they step, or summarise more often than they save.
            UserWarning: If the output rate or summary rate aren't an integer divisor of the internal loop frequency,
                update these values to be the actual time interval caused by stepping an integer number of steps. Warn
                the user that results aren't what they asked for.
        """

        step_size = self.dt_min
        dt_save = self.dt_save
        dt_summarise = self.dt_summarise

        n_steps_save = int(dt_save / step_size)
        n_steps_summarise = int(dt_summarise / dt_save)

        return n_steps_save, n_steps_summarise, step_size

    # @property
    # def n_states(self) -> int:
    #     return self._system_sizes.states
    #
    # @property
    # def n_observables(self) -> int:
    #     return self._system_sizes.observables
    #
    # @property
    # def n_parameters(self) -> int:
    #     return self._system_sizes.parameters
    #
    # @property
    # def n_drivers(self) -> int:
    #     return self._system_sizes.drivers

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