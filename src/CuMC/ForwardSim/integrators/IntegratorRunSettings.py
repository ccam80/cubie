import attrs
from CuMC.ForwardSim.integrators.algorithms.IntegratorLoopSettings import LoopStepConfig
from numpy import ceil
from warnings import warn

@attrs.define
class IntegratorRunSettings:
    """Container for runtime/timing settings that are commonly grouped together."""
    duration: float = attrs.field(default=1.0, validator=attrs.validators.instance_of(float))
    warmup: float = attrs.field(default=0.0, validator=attrs.validators.instance_of(float))
    dt_min: float = attrs.field(default=1e-6, validator=attrs.validators.instance_of(float))
    dt_max: float = attrs.field(default=1.0, validator=attrs.validators.instance_of(float))
    dt_save: float = attrs.field(default=0.1, validator=attrs.validators.instance_of(float))
    dt_summarise: float = attrs.field(default=0.1, validator=attrs.validators.instance_of(float))
    atol: float = attrs.field(default=1e-6, validator=attrs.validators.instance_of(float))
    rtol: float = attrs.field(default=1e-6, validator=attrs.validators.instance_of(float))

    def __attrs_post_init__(self):
        """Validate timing relationships."""
        #This is covered in singleintegrator
        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must be <= dt_max")
        if self.dt_save < self.dt_min:
            raise ValueError("dt_save must be >= dt_min")
        if self.dt_summarise < self.dt_save:
            raise ValueError("dt_summarise must be >= dt_save")
        if self.dt_summarise <= 0: #Potentially gate this based on whether summaries are requested?:
            raise ValueError("dt_summarise must be greater than 0")
        if self.dt_save <= 0: #Potentially gate this based on whether summaries are requested?:
            raise ValueError("dt_save must be greater than 0")

    def _validate_timing(self):
        """Check for impossible or inconsistent timing settings, like saving more frequently than stepping or
        summarising more frequently than saving. Raise errors for impossibilities, or warnings if parameters are
        ignored."""

        dt_min = self.dt_min
        dt_max = self.dt_max
        dt_save = self.dt_save
        dt_summarise = self.dt_summarise

        if self.dt_max < self.dt_min:
            raise ValueError(f"dt_max ({dt_max}s) must be >= dt_min ({dt_min}s)."
                             )
        if dt_save < dt_min:
            raise ValueError(f"dt_save ({dt_save}s) must be >= dt_min ({dt_min}s). "
                             )
        if dt_summarise < dt_save:
            raise ValueError(
                f"dt_summarise ({dt_summarise}s) must be >= to dt_save ({dt_save}s)"
                )

        if dt_max > dt_save:
            warn(f"dt_max ({dt_max}s) > dt_save ({dt_save}s). The loop will never be able to step"
                 f"that far before stopping to save, so dt_max is redundant.", UserWarning
                 )

    def _discretize_steps(self):
        step_size = self.dt_min
        dt_save = self.dt_save
        dt_summarise = self.dt_summarise

        n_steps_save = int(dt_save / step_size)
        n_steps_summarise = int(dt_summarise / dt_save)

        # Calculate actual time intervals after discretisation into steps
        actual_dt_save = n_steps_save * step_size
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

    @property
    def loop_step_config(self):
        """Return a dictionary of the fixed step configuration."""
        return LoopStepConfig(dt_min=self.dt_min,
                              dt_max=self.dt_max,
                              dt_save=self.dt_save,
                              dt_summarise=self.dt_summarise,
                              atol=self.atol,
                              rtol=self.rtol
                              )

    @property
    def output_samples(self):
        """Calculate the number of output samples based on the duration and dt_save."""
        return int(ceil(self.duration) / self.dt_save)

    @property
    def summary_samples(self):
        """Calculate the number of summary samples based on the duration and dt_summarise."""
        return int(ceil(self.duration) / self.dt_summarise)