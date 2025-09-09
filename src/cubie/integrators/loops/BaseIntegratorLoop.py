"""
Base class for integration algorithm Loops.

This module provides the BaseIntegratorLoop class, which serves as
the base class for all ODE integration loops. This class provides default
update behaviour and properties for a unified interface and inherits build
and cache logic from CUDAFactory.

Integration loops handle the "outer" logic of an ODE integration, organising
steps and saving output, and call an algorithm-specific step function to do the
mathy end of the integration.
"""
from abc import abstractmethod
from typing import Optional, Callable


from cubie.CUDAFactory import CUDAFactory
from cubie._utils import in_attr


class BaseIntegratorLoop(CUDAFactory):
    """
    Base class for the stepping loop for ODE solving algorithms.

    This class handles building and caching of the loop device function, which
    is incorporated into a CUDA kernel for GPU execution.

    Parameters
    ----------
    precision : type
        Numerical precision type for computations.
    buffer_sizes : LoopBufferSizes
        Configuration object specifying buffer sizes.
    loop_step_config : LoopStepConfig
        Configuration object for loop step parameters.
    save_state_func : CUDA device function
        Function for saving state values during integration.
    update_summaries_func : CUDA device function
        Function for updating summary statistics.
    save_summaries_func : CUDA device function
        Function for saving summary statistics.
    compile_flags : OutputCompileFlags
        Compilation flags for device function generation.

    Notes
    -----
    Subclasses must override:

    - `_threads_per_loop` : Number of threads the algorithm uses
    - `build_loop()` : Factory method that builds the CUDA device function
    - `shared_memory_required` : Amount of shared memory the device allocates

    Data used in compiling and controlling the loop is handled by the
    IntegratorLoopSettings class. This class presents relevant attributes
    of the data class to higher-level components as properties.

    See Also
    --------
    IntegratorLoopSettings : Configuration data for loop compilation
    CUDAFactory : Base factory class for CUDA device functions
    """

    def __init__(
        self,
        save_state_func: Callable,
        update_summaries_func: Callable,
        save_summaries_func: Callable,
        step_fn: Callable,
    ):
        super().__init__()

        self.save_state_func = save_state_func
        self.update_summaries_func = update_summaries_func
        self.save_summaries_func = save_summaries_func
        self.step_fn = step_fn
        self.loop_fn = None

    @abstractmethod
    def build(self):
        """
        Build the integrator loop, unpacking config for local scope.

        Returns
        -------
        callable
            The compiled integrator loop device function.
        """
        return NotImplementedError

    @property
    @abstractmethod
    def shared_memory_elements(self):
        """
        Get the number of threads required by loop algorithm.

        Returns
        -------
        int
            Number of threads required per integration loop.
        """
        raise NotImplementedError

    def update(self,
               updates_dict : Optional[dict] = None,
               silent: bool = False,
               **kwargs):
        """
        Pass updates to compile settings through the CUDAFactory interface.

        This method will invalidate the cache if an update is successful.
        Use silent=True when doing bulk updates with other component parameters
        to suppress warnings about unrecognized keys.

        Parameters
        ----------
        updates_dict
            Dictionary of parameters to update.
        silent
            If True, suppress warnings about unrecognized parameters.
        **kwargs
            Parameter updates to apply as keyword arguments.

        Returns
        -------
        set
            Set of parameter names that were recognized and updated.
        """
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        recognised = self.update_compile_settings(updates_dict, silent=True)
        for key, value in updates_dict.items():
            if in_attr(key, self.compile_settings.loop_step_config):
                setattr(self.compile_settings, key, value)
                recognised.add(key)
            if hasattr(self, key):
                # Update output and step functions
                setattr(self, key, value)
                recognised.add(key)
                self._invalidate_cache()

        unrecognised = set(updates_dict.keys()) - recognised
        if not silent and unrecognised:
            raise KeyError(
                f"Unrecognized parameters in update: {unrecognised}. "
                "These parameters were not updated.",
            )
        return recognised

    @classmethod
    @abstractmethod
    def from_single_integrator_run(cls, run_object):
        """
        Create an instance of the integrator algorithm from a SingleIntegratorRun object.

        Parameters
        ----------
        run_object : SingleIntegratorRun
            The SingleIntegratorRun object containing configuration parameters.

        Returns
        -------
        BaseIntegratorLoop
            New instance of the integrator algorithm configured with parameters
            from the run object.
        """
        raise NotImplementedError

    @property
    def shared_memory_indices(self):
        return self.compile_settings.shared_memory_indices

    @property
    def constant_memory_indices(self):
        return self.compile_settings.constant_memory_indices

    @property
    def local_memory_indices(self):
        return self.compile_settings.local_memory_indices
