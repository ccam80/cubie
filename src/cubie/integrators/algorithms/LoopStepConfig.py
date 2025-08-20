"""
Loop step configuration for integrator timing and tolerances.

This module provides the LoopStepConfig class, which serves as a convenience
class for grouping and passing around loop step timing information, including
step sizes, tolerances, and summary intervals.
"""

from attrs import define, field, validators


@define
class LoopStepConfig:
    """
    Step timing and exit conditions for an integrator loop.

    This class serves as a convenience container for grouping and passing
    around loop step information, including minimum and maximum step sizes,
    output intervals, and tolerance settings for integration algorithms.

    Parameters
    ----------
    dt_min : float, default=1e-6
        Minimum time step size for integration.
    dt_max : float, default=1.0
        Maximum time step size for integration.
    dt_save : float, default=0.1
        Time interval between saved output samples.
    dt_summarise : float, default=0.1
        Time interval between summary calculations.
    atol : float, default=1e-6
        Absolute tolerance for integration error control.
    rtol : float, default=1e-6
        Relative tolerance for integration error control.

    Notes
    -----
    This class provides convenient grouping of timing parameters and
    conversion utilities for fixed-step algorithms. The `fixed_steps`
    property converts time-based requests to integer numbers of steps.

    See Also
    --------
    IntegratorLoopSettings : Higher-level loop configuration
    IntegratorRunSettings : Runtime configuration settings
    """

    dt_min: float = field(
        default=1e-6, validator=validators.instance_of(float)
    )
    dt_max: float = field(default=1.0, validator=validators.instance_of(float))
    dt_save: float = field(
        default=0.1, validator=validators.instance_of(float)
    )
    dt_summarise: float = field(
        default=0.1, validator=validators.instance_of(float)
    )
    atol: float = field(default=1e-6, validator=validators.instance_of(float))
    rtol: float = field(default=1e-6, validator=validators.instance_of(float))

    @property
    def fixed_steps(self):
        """
        Convert time-based requests to integer numbers of steps for fixed-step loops.

        This helper function converts time-based timing requests to integer
        numbers of steps at the minimum step size (dt_min). It performs
        sanity checks and may adjust values for fixed-step algorithm compatibility.

        Returns
        -------
        tuple
            A tuple containing:
            - save_every_samples (int): Number of internal loop steps between saves
            - summarise_every_samples (int): Number of output samples between summary calculations
            - step_size (float): Internal time step size used in the loop (dt_min by default)

        Notes
        -----
        For fixed-step algorithms, dt_min is used as the internal step size.
        The number of steps between saves and summaries are computed as integer
        divisions, which may result in slight adjustments to the requested timing.
        """
        step_size = self.dt_min
        dt_save = self.dt_save
        dt_summarise = self.dt_summarise

        n_steps_save = int(dt_save / step_size)
        n_steps_summarise = int(dt_summarise / dt_save)

        return n_steps_save, n_steps_summarise, step_size
