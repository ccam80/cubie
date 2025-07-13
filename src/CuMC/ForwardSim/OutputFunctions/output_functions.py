from dataclasses import dataclass
from numpy import asarray
from numpy.typing import ArrayLike
from typing import Sequence, Dict, Any
from CuMC.CUDAFactory import CUDAFactory
from CuMC.ForwardSim.OutputFunctions.save_state import save_state_factory
from CuMC.ForwardSim.OutputFunctions.save_summaries import save_summary_factory
from CuMC.ForwardSim.OutputFunctions.update_summaries import update_summary_factory

#feature: max absolute
#feature: running std deviation
#feature: min
#feature: neg_peak
#feature: both_extrema
#feature: dxdt_extrema
#feature: d2xdt2 extrema
#feature: dxdt_max_peaks
#feature: raw dxdt

#TODO: Implement a "terminate" flag to communicate that a condition has been met - e.g. we have found a peak,
# so stop integrating.

@dataclass
class OutputCompileFlags:
    save_state: bool = False
    save_observables: bool = False
    save_time: bool = False
    summarise_mean: bool = False
    summarise_max: bool = False
    summarise_peaks: bool = False
    summarise_rms: bool = False
    summarise: bool = False  # True if any of the summarise_* flags are set


_TOKENS_TO_COMPILE_FLAGS = {
    "state":       "save_state",
    "observables": "save_observables",
    "time":        "save_time",
    "peaks":       "summarise_peaks",
    "mean":        "summarise_mean",
    "rms":         "summarise_rms",
    "max":         "summarise_max",
    }


class _TempMemoryRequirements(dict):
    """  Just a dictionary that you can instantiate with an integer, setting the only variable value."""

    def __init__(self, num_peaks: int):
        super().__init__({
            "state":       0,
            "observables": 0,
            "time":        0,
            "mean":        1,
            "peaks":       3 + num_peaks,  # prev + prev_prev + peak_counter
            "rms":         1,
            "max":         1,
            },
                )


class _OutputMemoryRequirements(dict):
    """ Just a dictionary that you can instantiate with an integer, setting the only variable value."""

    def __init__(self, num_peaks: int):
        super().__init__({
            "state":       0,
            "observables": 0,
            "time":        0,
            "mean":        1,
            "peaks":       num_peaks,
            "rms":         1,
            "max":         1,
            },
                )


class OutputFunctions(CUDAFactory):
    """Class to hold output functions and associated data, with automatic caching of built functions provided by the
    CUDAFactory base class.
    ."""

    def __init__(self,
                 outputs_list: list[str] = None,
                 saved_states: Sequence[int] | ArrayLike = None,
                 saved_observables: Sequence[int] | ArrayLike = None,
                 n_peaks: int = None,
                 ):
        super().__init__()

        # Initialize with default settings
        compile_settings = {
            'outputs_list':      outputs_list,
            'saved_states':      saved_states,
            'saved_observables': saved_observables,
            'n_peaks':           n_peaks
            }

        self.setup_compile_settings(compile_settings)
        self._flags = self._output_list_to_compile_flags(compile_settings['outputs_list'])

    def _output_list_to_compile_flags(self, tokens: list[str]) -> "OutputCompileFlags":
        f = OutputCompileFlags()
        if tokens is None:
            return f

        for tok in tokens:
            try:
                attr = _TOKENS_TO_COMPILE_FLAGS[tok]
                setattr(f, attr, True)
            except KeyError:
                raise ValueError(f"Unknown option: {tok}")
        if f.summarise_mean or f.summarise_max or f.summarise_peaks or f.summarise_rms:
            f.summarise = True
        return f

    def update(self, **kwargs):
        """Update the configuration of the output functions with new parameters."""
        self.update_compile_settings(**kwargs)
        self._flags = self._output_list_to_compile_flags(self.compile_settings['outputs_list'])

    def build(self) -> Dict[str, Any]:
        """Compile three functions: Save state, update summary metrics, and save summaries.
        Calculate memory requirements for temporary and output arrays.

        Returns:
            A dictionary containing all compiled functions and memory requirements
        """
        outputs_list = self.compile_settings['outputs_list']
        saved_states = asarray(self.compile_settings['saved_states'])
        saved_observables = asarray(self.compile_settings['saved_observables'])
        n_peaks = self.compile_settings['n_peaks']

        save_state = self._flags.save_state
        save_observables = self._flags.save_observables
        save_time = self._flags.save_time
        summarise = self._flags.summarise
        summarise_mean = self._flags.summarise_mean
        summarise_max = self._flags.summarise_max
        summarise_peaks = self._flags.summarise_peaks
        summarise_rms = self._flags.summarise_rms

        nstates = len(saved_states)
        nobs = len(saved_observables)
        n_peaks = n_peaks if n_peaks is not None else 0

        #Return memory per-state so that the number can be used to allocate separate arrays for each of state, observables.
        temporary_requirements = sum([_TempMemoryRequirements(n_peaks)[output_type] for output_type in outputs_list])
        output_requirements = sum([_OutputMemoryRequirements(n_peaks)[output_type] for output_type in outputs_list])

        save_state_func = save_state_factory(
                nstates,
                nobs,
                saved_states,
                saved_observables,
                save_state,
                save_observables,
                save_time,
                )

        update_summary_metrics_func = update_summary_factory(
                nstates,
                nobs,
                saved_states,
                saved_observables,
                summarise,
                summarise_mean,
                summarise_peaks,
                summarise_max,
                summarise_rms,
                save_observables,
                n_peaks,
                )

        save_summary_metrics_func = save_summary_factory(
                nstates,
                nobs,
                summarise,
                summarise_mean,
                summarise_peaks,
                summarise_max,
                summarise_rms,
                save_observables,
                n_peaks,
                )

        # Return all built functions and memory requirements in a dictionary
        return {
            'save_state_function':             save_state_func,
            'update_summary_metrics_function': update_summary_metrics_func,
            'save_summary_metrics_function':   save_summary_metrics_func,
            'temp_memory_requirements':        temporary_requirements,
            'summary_output_length':           output_requirements,
            'save_time':                       save_time
            }

    @property
    def save_state_func(self):
        """Return the save_state function. Will rebuild if necessary."""
        return self.get_cached_output('save_state_function')

    @property
    def update_summary_metrics_func(self):
        """Return the update_summary_metrics function. Will rebuild if necessary."""
        return self.get_cached_output('update_summary_metrics_function')

    @property
    def save_summary_metrics_func(self):
        """Return the save_summary_metrics function. Will rebuild if necessary."""
        return self.get_cached_output('save_summary_metrics_function')

    @property
    def memory_per_summarised_variable(self):
        """Return the memory requirements for temporary and output arrays."""
        return {
            'temporary': self.get_cached_output('temp_memory_requirements'),
            'output':    self.get_cached_output('summary_output_length')
            }

    @property
    def save_time(self):
        """Return whether time is being saved."""
        return self.get_cached_output('save_time')