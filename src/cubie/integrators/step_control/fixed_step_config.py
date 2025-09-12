"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""

from attrs import define, field

from cubie._utils import getype_validator
from cubie.integrators.step_control.base_step_controller import (
    BaseStepControllerConfig
)

@define
class FixedStepControlConfig(BaseStepControllerConfig):
    """ Configuration for fixed-step integrator loops.

    There's not much to it.

    Parameters
    ----------
    precision:
        numpy datatype to work in: select from (float32, float64, float16)
    dt:
        fixed step size

    Attributes
    ----------
    precision:
        numpy datatype to work in: select from (float32, float64, float16)
    dt:
        fixed step size

    Methods
    -------
    dt_min: self.precision
        alias for dt
    dt_max: self.precision
        alias for dt
    dt0: self.precision
        alias for dt
    is_adaptive: bool
        False

    """
    dt: float = field(default=1e-3, validator=getype_validator(float, 0))

    def __attrs_post_init__(self):
        self._validate_config()

    def _validate_config(self):
        return True

    @property
    def dt_min(self) -> float:
        """Returns the minimum time step size."""
        return self.dt

    @property
    def dt_max(self) -> float:
        """Returns the maximum step size."""
        return self.dt
    @property
    def dt0(self) -> float:
        """Returns initial step size at start of loop."""
        return self.dt

    @property
    def is_adaptive(self) -> bool:
        """Returns whether the step controller is adaptive."""
        return False


# Backwards compatibility alias
IntegratorLoopSettings = FixedStepControlConfig



    # def _discretize_steps(self):
    #     """
    #     Discretize the step sizes for saving and summarising.
    #
    #     Adjusts dt_save and dt_summarise to be integer multiples of the
    #     minimum step size (dt_min) and issues warnings if the requested
    #     values are not achievable in a fixed-step algorithm.
    #     """
    #     step_size = self.dt_min
    #     dt_save = self.dt_save
    #     dt_summarise = self.dt_summarise
    #
    #     n_steps_save = int(dt_save / step_size)
    #     actual_dt_save = n_steps_save * step_size
    #
    #     n_steps_summarise = int(dt_summarise / actual_dt_save)
    #     actual_dt_summarise = n_steps_summarise * actual_dt_save
    #
    #     # Update parameters if they differ from requested values and warn the user
    #     if actual_dt_save != dt_save:
    #         self.dt_save = actual_dt_save
    #         warn(
    #             f"dt_save({dt_save}s) is not an integer multiple of loop step size ({step_size}s), "
    #             f"so is unachievable in a fixed-step algorithm. The actual time between output samples is "
    #             f"({actual_dt_save}s)",
    #             UserWarning,
    #         )
    #
    #     if actual_dt_summarise != dt_summarise:
    #         self.dt_summarise = actual_dt_summarise
    #         warn(
    #             f"dt_summarise({dt_summarise}s) is not an integer multiple of dt_save ({actual_dt_save}s), "
    #             f"so is unachievable in a fixed-step algorithm. The actual time between summary values is "
    #             f"({actual_dt_summarise}s)",
    #             UserWarning,
    #         )
    #
    #
    # @classmethod
    # def from_integrator_run(cls, run_object):
    #     """
    #     Create an IntegratorLoopSettings instance from a SingleIntegratorRun object.
    #
    #     Parameters
    #     ----------
    #     run_object : SingleIntegratorRun
    #         The SingleIntegratorRun object containing configuration parameters.
    #
    #     Returns
    #     -------
    #     IntegratorLoopSettings
    #         New instance configured with parameters from the run object.
    #     """
    #     # return cls(
    #     #     loop_step_config=run_object.loop_step_config,
    #     #     buffer_sizes=run_object.loop_buffer_sizes,
    #     #     precision=run_object.precision,
    #     #     dxdt_function=run_object.dxdt_function,
    #     #     save_state_func=run_object.save_state_func,
    #     #     update_summaries_func=run_object.update_summaries_func,
    #     #     save_summaries_func=run_object.save_summaries_func,
    #     # )
    #     pass
