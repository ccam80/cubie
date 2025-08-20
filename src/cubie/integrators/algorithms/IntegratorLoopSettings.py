"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""

from typing import Callable, Optional

from attrs import define, field, validators
from numpy import float32, float16, float64

from cubie.outputhandling.output_config import OutputCompileFlags
from cubie.outputhandling.output_sizes import LoopBufferSizes
from cubie.integrators.algorithms.LoopStepConfig import LoopStepConfig


@define
class IntegratorLoopSettings:
    """
    Compile-critical settings for the integrator loop.

    This class manages configuration settings that are critical for compiling
    integrator loops, including timing parameters, buffer sizes, precision,
    and function references. The integrator loop is not the source of truth
    for these settings, so minimal setters are provided. Instead, there are
    update_from methods which extract relevant settings from other objects.

    Parameters
    ----------
    loop_step_config : LoopStepConfig
        Configuration object for loop step timing parameters.
    buffer_sizes : LoopBufferSizes
        Configuration object specifying buffer sizes for integration.
    precision : type, default=float32
        Numerical precision type (float32, float64, or float16).
    compile_flags : OutputCompileFlags, default=OutputCompileFlags()
        Compilation flags for output handling.
    dxdt_function : callable, optional
        Function that computes time derivatives of the state.
    save_state_func : callable, optional
        Function for saving state values during integration.
    update_summaries_func : callable, optional
        Function for updating summary statistics.
    save_summaries_func : callable, optional
        Function for saving summary statistics.

    Notes
    -----
    This class serves as a data container for compile-time settings and
    provides convenient property access to nested configuration values.
    It validates input parameters to ensure consistency.

    See Also
    --------
    LoopStepConfig : Step timing configuration
    LoopBufferSizes : Buffer size configuration
    OutputCompileFlags : Output compilation flags
    """

    # Core system properties
    loop_step_config: LoopStepConfig = field(
        validator=validators.instance_of(LoopStepConfig)
    )
    buffer_sizes: LoopBufferSizes = field(
        validator=validators.instance_of(LoopBufferSizes)
    )
    precision: type = field(
        default=float32,
        validator=validators.and_(
            validators.instance_of(type),
            validators.in_(
                [float32, float64, float16],
            ),
        ),
    )
    compile_flags: OutputCompileFlags = field(
        default=OutputCompileFlags(),
        validator=validators.instance_of(
            OutputCompileFlags,
        ),
    )
    dxdt_function: Optional[Callable] = field(default=None)
    save_state_func: Optional[Callable] = field(default=None)
    update_summaries_func: Optional[Callable] = field(default=None)
    save_summaries_func: Optional[Callable] = field(default=None)

    @property
    def fixed_steps(self):
        """
        Get the fixed steps configuration.

        Returns
        -------
        tuple
            Tuple of (save_every_samples, summarise_every_samples, step_size).
        """
        return self.loop_step_config.fixed_steps

    @property
    def fixed_step_size(self) -> float:
        """
        Get the step size used in the loop.

        Returns
        -------
        float
            The fixed step size for integration.
        """
        return self.loop_step_config.fixed_steps[-1]

    @property
    def dt_min(self) -> float:
        """
        Get the minimum time step size.

        Returns
        -------
        float
            Minimum time step size from the loop step configuration.
        """
        return self.loop_step_config.dt_min

    @property
    def dt_max(self) -> float:
        """
        Get the maximum time step size.

        Returns
        -------
        float
            Maximum time step size from the loop step configuration.
        """
        return self.loop_step_config.dt_max

    @property
    def dt_save(self) -> float:
        """
        Get the time interval between saved outputs.

        Returns
        -------
        float
            Time interval between saved outputs from the loop step configuration.
        """
        return self.loop_step_config.dt_save

    @property
    def dt_summarise(self) -> float:
        """
        Get the time interval between summary calculations.

        Returns
        -------
        float
            Time interval between summary calculations from the loop step configuration.
        """
        return self.loop_step_config.dt_summarise

    @property
    def atol(self) -> float:
        """
        Get the absolute tolerance for integration.

        Returns
        -------
        float
            Absolute tolerance from the loop step configuration.
        """
        return self.loop_step_config.atol

    @property
    def rtol(self) -> float:
        """
        Get the relative tolerance for integration.

        Returns
        -------
        float
            Relative tolerance from the loop step configuration.
        """
        return self.loop_step_config.rtol

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
