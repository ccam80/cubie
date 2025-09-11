"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""
from typing import Optional

from attrs import define, field, validators

from cubie.integrators.step_control.base_step_controller_config import \
    BaseStepControllerConfig


@define
class AdaptiveStepControlConfig(BaseStepControllerConfig):
    """
    Configuration parameters for a general adaptive step controller.

    This class can be used as-is for simple adaptive step controllers,
    or subclassed for more complex controllers. BaseStepControllerConfig
    defines the interface with the integrator loop; This class provides
    default logic for API properties.

     """

    _dt_min: float = field(
            validator=validators.instance_of(float)
    )
    _dt_max: Optional[float] = field(
            default=None,
            validator=validators.instance_of(float)
    )
    atol: float = field(
            default=1e-6,
            validator=validators.instance_of(float)
    )
    rtol: float = field(
            default=1e-6,
            validator=validators.instance_of(float)
    )

    def __attrs_post_init__(self):
        if self._dt_max is None:
            _dt_max = self.dt_min * 100

    @property
    def dt_min(self) -> float:
        """
        Returns the minimum time step size.

        Returns
        -------
        float
            Minimum time step size from the loop step configuration.
        """
        return self._dt_min

    @property
    def dt_max(self) -> float:
        """Returns the maximum time step size."""
        return self._dt_max

    @property
    def dt0(self) -> float:
        """Returns initial step size at start of loop."""
        return (self.dt_min + self.dt_max) / 2

    @property
    def is_adaptive(self) -> bool:
        """Returns whether the step controller is adaptive."""
        return True


    @classmethod
    def from_integrator_run(cls, run_object):
        """
        Create an IntegratorLoopSettings instance from a SingleIntegratorRun object.

        Parameters
        ----------
        run_object : SingleIntegratorRun
            The SingleIntegratorRun object containing configuration parameters.

        Returns
        -------
        IntegratorLoopSettings
            New instance configured with parameters from the run object.
        """
        return cls(
            loop_step_config=run_object.loop_step_config,
            buffer_sizes=run_object.loop_buffer_sizes,
            precision=run_object.precision,
            dxdt_function=run_object.dxdt_function,
            save_state_func=run_object.save_state_func,
            update_summaries_func=run_object.update_summaries_func,
            save_summaries_func=run_object.save_summaries_func,
        )
