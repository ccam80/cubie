"""
Single integrator run coordination for CUDA-based ODE solving.

This module provides the SingleIntegratorRun class which coordinates the
low-level CUDA machinery to create device functions for running individual
ODE integration runs. It handles dependency injection to integrator loop
algorithms and manages light-weight caching to ensure changes in subfunctions
are properly communicated.
"""

from typing import Optional

from numpy.typing import ArrayLike

from cubie._utils import in_attr
from cubie.integrators.algorithms import ImplementedAlgorithms
from cubie.integrators.IntegratorRunSettings import IntegratorRunSettings
from cubie.outputhandling.output_functions import OutputFunctions
from cubie.outputhandling.output_sizes import LoopBufferSizes
from cubie.odesystems.ODEData import SystemSizes


class SingleIntegratorRun:
    """
    Coordinates low-level CUDA machinery for single ODE integration runs.

    This class presents the interface to lower-level CUDA code and performs
    dependency injection to integrator loop algorithms. It contains light-weight
    cache management to ensure that changes in one subfunction are communicated
    to others, but does not inherit from CUDAFactory as it performs a different
    role than other factory classes.

    Parameters
    ----------
    system : BaseODE object
        The ODE system to integrate.
    algorithm : str, default='euler'
        Name of the integration algorithm to use.
    dt_min : float, default=0.01
        Minimum time step size.
    dt_max : float, default=0.1
        Maximum time step size.
    dt_save : float, default=0.1
        Time interval between saved outputs.
    dt_summarise : float, default=1.0
        Time interval between summary calculations.
    atol : float, default=1e-6
        Absolute tolerance for integration.
    rtol : float, default=1e-6
        Relative tolerance for integration.
    saved_state_indices : ArrayLike, optional
        Indices of states to save during integration.
    saved_observable_indices : ArrayLike, optional
        Indices of observables to save during integration.
    summarised_state_indices : ArrayLike, optional
        Indices of states to summarise during integration.
    summarised_observable_indices : ArrayLike, optional
        Indices of observables to summarise during integration.
    output_types : list of str or tuple of str, optional
        Types of outputs to generate.

    Notes
    -----
    This class handles modifications that invalidate the currently compiled
    loop, including:

    - Changes to system constants (compiled-in parameters)
    - Changes to loop outputs (adding/removing output types, summary types)
    - Changes to algorithm parameters (step size, tolerances, algorithm type)

    The class maintains a list of currently implemented algorithms accessible
    through the ImplementedAlgorithms registry. Additional algorithms can be
    added by subclassing GenericIntegratorAlgorithm.

    This class is not typically exposed to users directly - the user-facing
    API is through the Solver class which handles batching and memory management.

    All device functions maintain a local cache of their output functions
    and compile-sensitive attributes, and will invalidate and rebuild if any
    of these are updated.

    See Also
    --------
    IntegratorRunSettings : Runtime configuration settings
    ImplementedAlgorithms : Registry of available integration algorithms
    """

    def __init__(
        self,
        system,
        algorithm: str = "euler",
        dt_min: float = 0.01,
        dt_max: float = 0.1,
        dt_save: float = 0.1,
        dt_summarise: float = 1.0,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        saved_state_indices: Optional[ArrayLike] = None,
        saved_observable_indices: Optional[ArrayLike] = None,
        summarised_state_indices: Optional[ArrayLike] = None,
        summarised_observable_indices: Optional[ArrayLike] = None,
        output_types: list[str] = None,
    ):
        # Store the system
        self._system = system
        system_sizes = system.sizes

        # Initialize output functions with shapes from system
        self._output_functions = OutputFunctions(
            max_states=system_sizes.states,
            max_observables=system_sizes.observables,
            output_types=output_types,
            saved_state_indices=saved_state_indices,
            saved_observable_indices=saved_observable_indices,
            summarised_state_indices=summarised_state_indices,
            summarised_observable_indices=summarised_observable_indices,
        )

        compile_settings = IntegratorRunSettings(
            dt_min=dt_min,
            dt_max=dt_max,
            dt_save=dt_save,
            dt_summarise=dt_summarise,
            atol=atol,
            rtol=rtol,
            output_types=output_types,
        )

        self.config = compile_settings

        # Instantiate algorithm with info from system and output functions
        self.algorithm_key = algorithm.lower()
        algorithm = ImplementedAlgorithms[self.algorithm_key]
        self._integrator_instance = algorithm.from_single_integrator_run(self)

        self._compiled_loop = None
        self._loop_cache_valid = False

    @property
    def loop_buffer_sizes(self):
        """
        Get buffer sizes required for the integration loop.

        Returns
        -------
        LoopBufferSizes
            Buffer size configuration for the integration loop.
        """
        return LoopBufferSizes.from_system_and_output_fns(
            self._system, self._output_functions
        )

    @property
    def output_array_heights(self):
        """
        Get the heights of output arrays.

        Returns
        -------
        OutputArrayHeights
            Output array height configuration from the OutputFunctions object.
        """
        return self._output_functions.output_array_heights

    @property
    def summaries_buffer_sizes(self):
        """
        Get buffer sizes for summary calculations.

        Returns
        -------
        SummaryBufferSizes
            Summary buffer size configuration from the OutputFunctions object.
        """
        return self._output_functions.summaries_buffer_sizes

    def update(self, updates_dict=None, silent=False, **kwargs):
        """
        Update parameters across all components.

        This method sends all parameters to all child components with
        silent=True to avoid spurious warnings, then checks if any parameters
        were not recognized by any component.

        Parameters
        ----------
        updates_dict : dict, optional
            Dictionary of parameters to update.
        silent : bool, default=False
            If True, suppress warnings about unrecognized parameters.
        **kwargs
            Parameter updates to apply as keyword arguments.

        Returns
        -------
        set
            Set of parameter names that were recognized and updated.

        Raises
        ------
        KeyError
            If parameters are not recognized by any component and silent=False.
        """
        if updates_dict == None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        all_unrecognized = set(updates_dict.keys())
        recognized = set()

        # Update anything held in the config object (step sizes, etc)
        for key, value in updates_dict.items():
            if in_attr(key, self.config):
                setattr(self.config, key, value)
                recognized.add(key)

        if "algorithm" in updates_dict.keys():
            # If the algorithm is being updated, we need to reset the
            # integrator instance
            self.algorithm_key = updates_dict["algorithm"].lower()
            algorithm = ImplementedAlgorithms[self.algorithm_key]
            self._integrator_instance = algorithm.from_single_integrator_run(
                self
            )
            recognized.add("algorithm")

        recognized |= self._system.update(updates_dict, silent=True)
        recognized |= self._output_functions.update(updates_dict, silent=True)

        cached_loop_updates = {
            "dxdt_function": self.dxdt_function,
            "save_state_func": self.save_state_func,
            "update_summaries_func": self.update_summaries_func,
            "save_summaries_func": self.save_summaries_func,
            "buffer_sizes": self.loop_buffer_sizes,
            "loop_step_config": self.loop_step_config,
            "precision": self.precision,
            "compile_flags": self.compile_flags,
        }

        recognized |= self._integrator_instance.update(
            cached_loop_updates, silent=True
        )
        all_unrecognized -= recognized

        if all_unrecognized:
            if not silent:
                raise KeyError(
                    f"The following updates were not recognized by any "
                    f"component:"
                    f" {all_unrecognized}",
                )

        self.config.validate_settings()
        self._invalidate_cache()
        return recognized

    def _invalidate_cache(self):
        """
        Invalidate the compiled loop cache.

        Marks the current compiled loop as invalid, forcing a rebuild
        on the next access.
        """
        self._loop_cache_valid = False
        self._compiled_loop = None

    def build(self):
        """
        Build the complete integrator loop.

        Compiles the integrator loop device function and updates the cache.

        Returns
        -------
        CUDA device function
            The compiled loop device function.
        """

        # Update with latest function references

        self._compiled_loop = self._integrator_instance.device_function
        self._loop_cache_valid = True

        return self._compiled_loop

    @property
    def device_function(self):
        """
        Get the compiled loop function, building if necessary.

        Returns
        -------
        CUDA device function
            The compiled CUDA device function for the integration loop.
        """
        if not self._loop_cache_valid or self._compiled_loop is None:
            self.build()
        return self._compiled_loop

    @property
    def cache_valid(self):
        """
        Check if the compiled loop is current.

        Returns
        -------
        bool
            True if the cached loop is valid, False otherwise.
        """
        return self._loop_cache_valid

    @property
    def shared_memory_elements(self):
        """
        Get the number of shared memory elements required for integration.

        Returns
        -------
        int
            Number of elements of shared memory required for a single
            integrator run.
        """
        if not self.cache_valid:
            self.build()
        loop_memory = self._integrator_instance.shared_memory_required
        summary_buffers = self._output_functions.total_summary_buffer_size
        total_elements = loop_memory + summary_buffers

        return total_elements

    @property
    def shared_memory_bytes(self):
        """
        Get the number of bytes of dynamic shared memory required.

        Returns
        -------
        int
            Number of bytes of dynamic shared memory required for a single
            integrator run.
        """
        if not self.cache_valid:
            self.build()
        datasize = self.precision(0.0).nbytes
        return self.shared_memory_elements * datasize

    # Reach through this interface class to get lower level features:
    @property
    def precision(self):
        """
        Get the numerical precision type from the system.

        Returns
        -------
        type
            Numerical precision type (e.g., float32, float64) from the
            child BaseODE object.
        """
        return self._system.precision

    @property
    def threads_per_loop(self):
        """
        Get the number of threads required by the loop algorithm.

        Returns
        -------
        int
            Number of threads per loop from the child GenericIntegratorAlgorithm
            object.
        """
        return self._integrator_instance.threads_per_loop

    @property
    def dxdt_function(self):
        """
        Get the derivative function from the system.

        Returns
        -------
        callable
            The dxdt function from the child BaseODE object.
        """
        return self._system.dxdt_function

    @property
    def save_state_func(self):
        """
        Get the state saving function.

        Returns
        -------
        callable
            State saving function from the child OutputFunctions object.
        """
        return self._output_functions.save_state_func

    @property
    def update_summaries_func(self):
        """
        Get the summary update function.

        Returns
        -------
        callable
            Summary update function from the child OutputFunctions object.
        """
        return self._output_functions.update_summaries_func

    @property
    def save_summaries_func(self):
        """
        Get the summary saving function.

        Returns
        -------
        callable
            Summary saving function from the child OutputFunctions object.
        """
        return self._output_functions.save_summary_metrics_func

    @property
    def loop_step_config(self):
        """
        Get the loop step configuration.

        Returns
        -------
        LoopStepConfig
            Step configuration object for the integration loop.
        """
        return self.config.loop_step_config

    @property
    def fixed_step_size(self):
        """
        Get the fixed step size for integration.

        Returns
        -------
        float
            Fixed step size from the child GenericIntegratorAlgorithm object.
        """
        return self._integrator_instance.fixed_step_size

    @property
    def dt_save(self):
        """
        Get the time step size for saving states and observables.

        Returns
        -------
        float
            Time interval between saved outputs.
        """
        return self.config.dt_save

    @property
    def dt_summarise(self):
        """
        Get the time step size for summarising states and observables.

        Returns
        -------
        float
            Time interval between summary calculations.
        """
        return self.config.dt_summarise

    @property
    def system_sizes(self) -> SystemSizes:
        """
        Get the system size information.

        Returns
        -------
        SystemSizes
            Size information from the child BaseODE object.
        """
        return self._system.sizes

    @property
    def compile_flags(self):
        """
        Get the compilation flags for output functions.

        Returns
        -------
        object
            Compilation flags from the child OutputFunctions object.
        """
        return self._output_functions.compile_flags

    @property
    def output_types(self):
        """
        Get the configured output types.

        Returns
        -------
        list
            List of output types from the child OutputFunctions object.
        """
        return self._output_functions.output_types

    @property
    def summary_legend_per_variable(self):
        """
        Get the summary legend for each variable.

        Returns
        -------
        object
            Summary legend mapping from the child OutputFunctions object.
        """
        return self._output_functions.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """
        Get indices of states to save.

        Returns
        -------
        ArrayLike
            Indices of states to save from the child OutputFunctions object.
        """
        return self._output_functions.saved_state_indices

    @property
    def saved_observable_indices(self):
        """
        Get indices of observables to save.

        Returns
        -------
        ArrayLike
            Indices of observables to save from the child OutputFunctions object.
        """
        return self._output_functions.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """
        Get indices of states to summarise.

        Returns
        -------
        ArrayLike
            Indices of states to summarise from the child OutputFunctions object.
        """
        return self._output_functions.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """
        Get indices of observables to summarise.

        Returns
        -------
        ArrayLike
            Indices of observables to summarise from the child OutputFunctions object.
        """
        return self._output_functions.summarised_observable_indices

    @property
    def save_time(self):
        """
        Get whether time values should be saved.

        Returns
        -------
        bool
            True if time should be saved, from the child OutputFunctions object.
        """
        return self._output_functions.save_time

    @property
    def system(self):
        """
        Get the ODE system object.

        Returns
        -------
        object
            The child BaseODE system object.
        """
        return self._system
