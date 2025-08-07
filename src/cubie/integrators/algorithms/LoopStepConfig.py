from attrs import define, field, validators


@define
class LoopStepConfig:
    """
    Step timing and exit conditions for an integrator loop. Convenience
    class for grouping and passing around loop
    step information.
    """
    dt_min: float = field(default=1e-6,
                          validator=validators.instance_of(float))
    dt_max: float = field(default=1.0, validator=validators.instance_of(float))
    dt_save: float = field(default=0.1,
                           validator=validators.instance_of(float))
    dt_summarise: float = field(default=0.1,
                                validator=validators.instance_of(float))
    atol: float = field(default=1e-6, validator=validators.instance_of(float))
    rtol: float = field(default=1e-6, validator=validators.instance_of(float))

    @property
    def fixed_steps(self):
        """Fixed-step helper function: Convert time-based requests to integer numbers of steps at step_size (dt_min
        used by default in fixed-step loops). Sanity-check values and warn the user if they are adjusted.

        Returns:
            save_every_samples (int): The number of internal loop steps between saves.
            summarise_every_samples (int): The number of output samples between summary metric calculations.
            step_size (float): The internal time step size used in the loop (dt_min, by default).
        """

        step_size = self.dt_min
        dt_save = self.dt_save
        dt_summarise = self.dt_summarise

        n_steps_save = int(dt_save / step_size)
        n_steps_summarise = int(dt_summarise / dt_save)

        return n_steps_save, n_steps_summarise, step_size
