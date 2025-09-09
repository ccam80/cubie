"""
Integrator configuration management with validation and adapter patterns.

This module provides the IntegratorLoopSettings class for managing compile-critical
settings for integrator loops, including timing parameters, buffer sizes, and
function references. It uses validation and adapter patterns to ensure
configuration consistency.
"""
from warnings import warn

from attrs import define, field, validators


valid_float = validators.instance_of(float)
valid_int = validators.instance_of(int)



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
    _step_size = field(default=1e-3, validator=valid_float)
    _steps_per_save = field(default=1, validator=valid_int)
    _saves_per_summary = field(default=1, validator=valid_int)

    @property
    def step_size(self) -> float:
        """
        Get the step size used in the loop.

        Returns
        -------
        float
            The fixed step size for integration.
        """
        return self.step_size

    @property
    def steps_per_save(self) -> int:
        """
        Get the number of steps between saving states and observables.

        Returns
        -------
        int
            Interior steps between saves
        """
        return self._steps_per_save

    @property
    def saves_per_summary(self) -> int:
        """
        Get the number of steps between saving summary variables.

        Returns
        -------
        int
            Interior steps between saves
        """
        return self._steps_per_summary

    def steps_from_time(self, dt: float, dt_save: float, dt_summarise: float):
        """
        Convert time-based requests to integer numbers of steps for fixed-step loops.

        This helper function converts time-based timing requests to integer
        numbers of steps at the minimum step size (dt_min). It performs
        sanity checks and may adjust values for fixed-step algorithm compatibility.

        Parameters
        ----------
        dt
            step size for simulation in seconds
        dt_save:
            time between output samples
        dt_summarise:
            time between evaluations of summary metrics

        Notes
        -----
        For fixed-step algorithms
        The number of steps between saves and summaries are computed as integer
        divisions, which may result in slight adjustments to the requested timing.
        """
        self.step_size = dt
        self.steps_per_save = int(dt_save / dt)
        self.saves_per_summary = int(dt_summarise / dt_save)

    def _validate_timing(self):
        """
        Check for impossible or inconsistent timing settings.

        Validates timing relationships such as ensuring dt_max >= dt_min,
        dt_save >= dt_min, and dt_summarise >= dt_save. Raises errors for
        impossibilities and warnings if parameters are ignored.

        Raises
        ------
        ValueError
            If timing parameters have impossible relationships (e.g., dt_max < dt_min).

        Warns
        -----
        UserWarning
            If dt_max > dt_save, making dt_max redundant.
        """

        dt_min = self.dt_min
        dt_max = self.dt_max
        dt_save = self.dt_save
        dt_summarise = self.dt_summarise

        if self.dt_max < self.dt_min:
            raise ValueError(
                f"dt_max ({dt_max}s) must be >= dt_min ({dt_min}s).",
            )
        if dt_save < dt_min:
            raise ValueError(
                f"dt_save ({dt_save}s) must be >= dt_min ({dt_min}s). ",
            )
        if dt_summarise < dt_save:
            raise ValueError(
                f"dt_summarise ({dt_summarise}s) must be >= to dt_save ({dt_save}s)",
            )

        if dt_max > dt_save:
            warn(f"dt_max ({dt_max}s) > dt_save ({dt_save}s). The loop will never be able to step"
                f"that far before stopping to save, so dt_max is redundant.",
                UserWarning,
            )

    def _discretize_steps(self):
        """
        Discretize the step sizes for saving and summarising.

        Adjusts dt_save and dt_summarise to be integer multiples of the
        minimum step size (dt_min) and issues warnings if the requested
        values are not achievable in a fixed-step algorithm.
        """
        step_size = self.dt_min
        dt_save = self.dt_save
        dt_summarise = self.dt_summarise

        n_steps_save = int(dt_save / step_size)
        actual_dt_save = n_steps_save * step_size

        n_steps_summarise = int(dt_summarise / actual_dt_save)
        actual_dt_summarise = n_steps_summarise * actual_dt_save

        # Update parameters if they differ from requested values and warn the user
        if actual_dt_save != dt_save:
            self.dt_save = actual_dt_save
            warn(
                f"dt_save({dt_save}s) is not an integer multiple of loop step size ({step_size}s), "
                f"so is unachievable in a fixed-step algorithm. The actual time between output samples is "
                f"({actual_dt_save}s)",
                UserWarning,
            )

        if actual_dt_summarise != dt_summarise:
            self.dt_summarise = actual_dt_summarise
            warn(
                f"dt_summarise({dt_summarise}s) is not an integer multiple of dt_save ({actual_dt_save}s), "
                f"so is unachievable in a fixed-step algorithm. The actual time between summary values is "
                f"({actual_dt_summarise}s)",
                UserWarning,
            )


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
        # return cls(
        #     loop_step_config=run_object.loop_step_config,
        #     buffer_sizes=run_object.loop_buffer_sizes,
        #     precision=run_object.precision,
        #     dxdt_function=run_object.dxdt_function,
        #     save_state_func=run_object.save_state_func,
        #     update_summaries_func=run_object.update_summaries_func,
        #     save_summaries_func=run_object.save_summaries_func,
        # )
        pass
