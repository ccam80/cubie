"""
Integrator configuration management with validation and adapter patterns.
"""

from attrs import define, field, validators
from typing import Callable, Optional
from numba import float32, float64
from numba.types import Float as _numba_float_type
from warnings import warn
import numpy as np


@define
class IntegratorLoopSettings:
    """
    Compile-critical settings for the integrator loop, including timing and sizes. The integrator loop is not the
    source of truth for these settings, so minimal setters are provided. Instead, there are update_from methods which
    take in other updated objects and extract the relevant settings.
    """

    # Core system properties
    precision: _numba_float_type = field(default=float32, validator=validators.in_([float32, float64]))
    n_states: int = field(default=0, validator=validators.instance_of(int)) #getter
    n_observables: int = field(default=0, validator=validators.instance_of(int)) #getter
    n_parameters: int = field(default=0, validator=validators.instance_of(int)) # getter
    n_drivers: int = field(default=0, validator=validators.instance_of(int)) #getter

    dt_min: float = field(default=1e-6, validator=validators.instance_of(float))
    dt_max: float = field(default=1.0, validator=validators.instance_of(float))
    dt_save: float = field(default=0.1, validator=validators.instance_of(float))
    dt_summarise: float = field(default=0.1, validator=validators.instance_of(float))
    atol: float = field(default=1e-6, validator=validators.instance_of(float))
    rtol: float = field(default=1e-6, validator=validators.instance_of(float))

    # Output configuration
    save_time: bool = field(default=False, validator=validators.instance_of(bool)) # getter from
    # summarymetrics
    n_saved_states: int = field(default=0, validator=validators.instance_of(int)) #source of truth:
    # outputfunctions
    n_saved_observables: int = field(default=0, validator=validators.instance_of(int)) #source_of_truth:
    # output_functions
    summary_temp_memory: int = field(default=0, validator=validators.instance_of(int)) # getter from
    # summary_metrics

    # Function references (set by algorithm, not user)
    dxdt_func: Optional[Callable] = field(default=None)
    save_state_func: Optional[Callable] = field(default=None)
    update_summary_func: Optional[Callable] = field(default=None)
    save_summary_func: Optional[Callable] = field(default=None)

    def __attrs_post_init__(self):
        self._validate_timing()

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

        #Calculate actual time intervals after discretisation into steps
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

        return n_steps_save, n_steps_summarise, step_size


    @classmethod
    def from_run_settings(cls,
                         precision,
                         n_states,
                         n_observables,
                         n_parameters,
                         n_drivers,
                         run_settings: IntegatorRunSettings,
                         save_time,
                         n_saved_states,
                         n_saved_observables,
                         summary_temp_memory,
                         dxdt_func,
                         save_state_func,
                         update_summary_func,
                         save_summary_func):
        """
        Adapter: Create config from IntegratorRunSettings object.
        Breaks down the run_settings object into individual arguments.
        """
        return cls(
            precision=precision,
            n_states=n_states,
            n_observables=n_observables,
            n_parameters=n_parameters,
            n_drivers=n_drivers,
            duration=run_settings.duration,
            warmup=run_settings.warmup,
            dt_min=run_settings.dt_min,
            dt_max=run_settings.dt_max,
            dt_save=run_settings.dt_save,
            dt_summarise=run_settings.dt_summarise,
            atol=run_settings.atol,
            rtol=run_settings.rtol,
            save_time=save_time,
            n_saved_states=n_saved_states,
            n_saved_observables=n_saved_observables,
            summary_temp_memory=summary_temp_memory,
            dxdt_func=dxdt_func,
            save_state_func=save_state_func,
            update_summary_func=update_summary_func,
            save_summary_func=save_summary_func,
        )

    @classmethod
    def from_single_integrator_run(cls, single_integrator_run):
        """
        Adapter: Create config from SingleIntegratorRun instance.
        Extracts relevant settings from the existing architecture.
        """
        system_sizes = single_integrator_run._system.sizes()
        output_config = single_integrator_run._output_functions.compile_settings
        params = single_integrator_run._algorithm_params

        return cls(
            precision=single_integrator_run._system.precision(),
            n_states=system_sizes['n_states'],
            n_observables=system_sizes['max_observables'],
            n_parameters=system_sizes['n_parameters'],
            n_drivers=system_sizes['n_drivers'],
            duration=params.get('duration', 1.0),
            warmup=params.get('warmup', 0.0),
            dt_min=params['dt_min'],
            dt_max=params['dt_max'],
            dt_save=params['dt_save'],
            dt_summarise=params['dt_summarise'],
            atol=params['atol'],
            rtol=params['rtol'],
            save_time=output_config.save_time,
            n_saved_states=output_config.n_saved_states,
            n_saved_observables=output_config.n_saved_observables,
            summary_temp_memory=single_integrator_run._output_functions.memory_per_summarised_variable['temporary'],
            dxdt_func=single_integrator_run._system.device_function,
            save_state_func=single_integrator_run._output_functions.save_state_func,
            update_summary_func=single_integrator_run._output_functions.update_summaries_func,
            save_summary_func=single_integrator_run._output_functions.save_summary_metrics_func,
        )

    @classmethod
    def from_components(cls, system, output_functions, **timing_kwargs):
        """
        Adapter: Create config from system and output_functions components.
        More modular approach for new code.
        """
        system_sizes = system.sizes()
        output_config = output_functions.compile_settings

        # Extract timing parameters with defaults
        duration = timing_kwargs.get('duration', 1.0)
        warmup = timing_kwargs.get('warmup', 0.0)
        dt_min = timing_kwargs.get('dt_min', 1e-6)
        dt_max = timing_kwargs.get('dt_max', 1.0)
        dt_save = timing_kwargs.get('dt_save', 0.1)
        dt_summarise = timing_kwargs.get('dt_summarise', 0.1)
        atol = timing_kwargs.get('atol', 1e-6)
        rtol = timing_kwargs.get('rtol', 1e-6)

        return cls(
            precision=system.precision(),
            n_states=system_sizes['n_states'],
            n_observables=system_sizes['max_observables'],
            n_parameters=system_sizes['n_parameters'],
            n_drivers=system_sizes['n_drivers'],
            duration=duration,
            warmup=warmup,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_save=dt_save,
            dt_summarise=dt_summarise,
            atol=atol,
            rtol=rtol,
            save_time=output_config.save_time,
            n_saved_states=output_config.n_saved_states,
            n_saved_observables=output_config.n_saved_observables,
            summary_temp_memory=output_functions.memory_per_summarised_variable['temporary'],
            dxdt_func=system.device_function,
            save_state_func=output_functions.save_state_func,
            update_summary_func=output_functions.update_summaries_func,
            save_summary_func=output_functions.save_summary_metrics_func,
        )
