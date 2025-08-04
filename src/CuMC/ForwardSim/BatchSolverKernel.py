# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:45:03 2025

@author: cca79
"""

import os
from typing import Optional

import numpy as np
from numba import cuda
from numba import int32, int16, from_dtype
from numpy.typing import NDArray, ArrayLike

from CuMC.CUDAFactory import CUDAFactory
from CuMC.ForwardSim.BatchInputArrays import InputArrays
from CuMC.ForwardSim.BatchOutputArrays import OutputArrays
from CuMC.ForwardSim.BatchSolverConfig import BatchSolverConfig
from CuMC.ForwardSim.OutputHandling.output_sizes import BatchOutputSizes, SingleRunOutputSizes
from CuMC.ForwardSim.integrators.SingleIntegratorRun import SingleIntegratorRun


class BatchSolverKernel(CUDAFactory):
    """Class which builds and holds the integrating kernel and interfaces with lower-level modules: loop
    algorithms, ODE systems, and output functions
    The kernel function accepts single or batched sets of inputs, and distributes those amongst the threads on the
    GPU. It runs the loop device function on a given slice of it's allocated memory, and serves as the distributor
    of work amongst the individual runs of the integrators.
    This class is one level down from the user, managing sanitised inputs and handling the machinery of batching and
    running integrators. It does not handle:
     - Integration logic/algorithms - these are handled in SingleIntegratorRun, and below
     - Input sanitisation / batch construction - this is handled in the solver api.
     - System equations - these are handled in the system model classes.
    """

    def __init__(self,
                 system,
                 algorithm: str = 'euler',
                 duration: float = 1.0,
                 warmup: float = 0.0,
                 dt_min: float = 0.01,
                 dt_max: float = 0.1,
                 dt_save: float = 0.1,
                 dt_summarise: float = 1.0,
                 atol: float = 1e-6,
                 rtol: float = 1e-6,
                 saved_state_indices: NDArray[np.int_] = None,
                 saved_observable_indices: NDArray[np.int_] = None,
                 summarised_state_indices: Optional[ArrayLike] = None,
                 summarised_observable_indices: Optional[ArrayLike] = None,
                 output_types: list[str] = None,
                 precision: type = np.float64,
                 profileCUDA: bool = False,
                 ):
        super().__init__()

        config = BatchSolverConfig(precision=precision,
                                   algorithm=algorithm,
                                   duration=duration,
                                   warmup=warmup,
                                   profileCUDA=profileCUDA,
                                   )

        # Setup compile settings for the kernel
        self.setup_compile_settings(config)

        if output_types is None:
            output_types = ["state"]

        self.single_integrator = SingleIntegratorRun(
                system,
                algorithm=algorithm,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_save=dt_save,
                dt_summarise=dt_summarise,
                atol=atol,
                rtol=rtol,
                saved_state_indices=saved_state_indices,
                saved_observable_indices=saved_observable_indices,
                summarised_state_indices=summarised_state_indices,
                summarised_observable_indices=summarised_observable_indices,
                output_types=output_types,
                )

        self.input_arrays = InputArrays.from_solver(self)
        self.output_arrays = OutputArrays.from_solver(self)

    @property
    def output_heights(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.output_array_heights` from the child SingleIntegratorRun object."""
        return self.single_integrator.output_array_heights

    @property
    def kernel(self):
        return self.device_function

    def build(self):
        return self.build_kernel()

    def run(self,
            duration,
            params,
            inits,
            forcing_vectors,
            output_arrays=None,
            blocksize=128,
            stream=0,
            warmup=0.0,
            ):
        """Run the solver kernel."""
        # Order currently VERY IMPORTANT - num_runs is updated in input_arrays, which is used in output_arrays.
        self.input_arrays(inits, params, forcing_vectors)
        numruns = self.input_arrays.num_runs

        self.output_arrays(self)

        numruns = self.input_arrays.num_runs

        threads_per_loop = self.single_integrator.threads_per_loop
        runsperblock = int(blocksize / self.single_integrator.threads_per_loop)
        BLOCKSPERGRID = int(max(1, np.ceil(numruns / blocksize)))
        dynamic_sharedmem = self.shared_memory_bytes_per_run * numruns

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self.compile_settings.profileCUDA:
            cuda.profile_start()

        self.device_function[BLOCKSPERGRID, (threads_per_loop, runsperblock), stream, dynamic_sharedmem](
                self.input_arrays.device_initial_values,
                self.input_arrays.device_parameters,
                self.input_arrays.device_forcing_vectors,
                self.output_arrays.state,
                self.output_arrays.observables,
                self.output_arrays.state_summaries,
                self.output_arrays.observable_summaries,
                self.compile_settings.duration,
                self.compile_settings.warmup,
                numruns,
                )
        cuda.synchronize()

        if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1" and self.profileCUDA:
            cuda.profile_stop()

        return self.output_arrays

    def build_kernel(self):
        """Build the integration kernel."""
        precision = from_dtype(self.precision)
        loopfunction = self.single_integrator.device_function
        shared_elements_per_run = self.shared_memory_elements_per_run

        @cuda.jit((precision[:, :],
                   precision[:, :],
                   precision[:, :],
                   precision[:, :, :],
                   precision[:, :, :],
                   precision[:, :, :],
                   precision[:, :, :],
                   int32,
                   int32,
                   int32
                   ),
                  )
        def integration_kernel(inits,
                               params,
                               forcing_vector,
                               state_output,
                               observables_output,
                               state_summaries_output,
                               observables_summaries_output,
                               duration_samples,
                               warmup_samples=0,
                               n_runs=1,
                               ):
            """Master integration kernel - calls integratorLoop and dxdt device functions."""
            tx = int16(cuda.threadIdx.x)
            ty = int16(cuda.threadIdx.y)

            block_index = int32(cuda.blockIdx.x)
            runs_per_block = cuda.blockDim.y
            run_index = int32(runs_per_block * block_index + tx)

            if run_index >= n_runs:
                return None

            n_init_sets = inits.shape[0]
            init_index = run_index % n_init_sets
            param_index = run_index // n_init_sets

            shared_memory = cuda.shared.array(0, dtype=precision)
            c_forcing_vector = cuda.const.array_like(forcing_vector)

            # Run-indexed slices of shared and output memory
            rx_shared_memory = shared_memory[ty * shared_elements_per_run:(ty + 1) * shared_elements_per_run]
            rx_inits = inits[init_index, :]
            rx_params = params[param_index, :]
            rx_state = state_output[:, run_index, :]
            rx_observables = observables_output[:, run_index, :]
            rx_state_summaries = state_summaries_output[:, run_index, :]
            rx_observables_summaries = observables_summaries_output[:, run_index, :]

            loopfunction(
                    rx_inits,
                    rx_params,
                    c_forcing_vector,
                    rx_shared_memory,
                    rx_state,
                    rx_observables,
                    rx_state_summaries,
                    rx_observables_summaries,
                    duration_samples,
                    warmup_samples,
                    )

            return None

        return integration_kernel

    def update(self, updates_dict=None, silent=False, **kwargs):
        if updates_dict is None:
            updates_dict = {}
        if kwargs:
            updates_dict.update(kwargs)
        if updates_dict == {}:
            return set()

        all_unrecognized = set(updates_dict.keys())
        all_unrecognized -= self.update_compile_settings(updates_dict, silent=True)
        all_unrecognized -= self.single_integrator.update(updates_dict, silent=True)
        recognised = set(updates_dict.keys()) - all_unrecognized

        if all_unrecognized:
            if not silent:
                raise KeyError(f"Unrecognized parameters: {all_unrecognized}")
        return recognised

    @property
    def shared_memory_bytes_per_run(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.shared_memory_bytes` from the child SingleIntegratorRun object."""
        return self.single_integrator.shared_memory_bytes

    @property
    def shared_memory_elements_per_run(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.shared_memory_elements` from the child SingleIntegratorRun object."""
        return self.single_integrator.shared_memory_elements

    @property
    def precision(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.precision` from the child SingleIntegratorRun object."""
        return self.single_integrator.precision

    @property
    def threads_per_loop(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.threads_per_loop` from the child SingleIntegratorRun object."""
        return self.single_integrator.threads_per_loop

    @property
    def output_length(self):
        """Returns the number of output samples per run."""
        return int(np.ceil(self.compile_settings.duration / self.single_integrator.dt_save))

    @property
    def summaries_length(self):
        """Returns the number of summary samples per run."""
        return int(np.ceil(self.compile_settings.duration / self.single_integrator.dt_summarise))

    @property
    def num_runs(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchInputArrays.InputArrays.num_runs` from the child InputArrays object."""
        return self.input_arrays.num_runs

    @property
    def system(self):
        """Exposes the child system object from the SingleIntegratorRun instance."""
        return self.single_integrator._system

    @property
    def duration(self):
        """Returns the duration of the simulation."""
        return self.compile_settings.duration

    @property
    def warmup(self):
        """Returns the warmup time of the simulation."""
        return self.compile_settings.warmup

    @property
    def dt_save(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.dt_save` from the child SingleIntegratorRun object."""
        return self.single_integrator.dt_save

    @property
    def dt_summarise(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.dt_summarise` from the child SingleIntegratorRun object."""
        return self.single_integrator.dt_summarise

    @property
    def system_sizes(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.system_sizes` from the child SingleIntegratorRun object."""
        return self.single_integrator.system_sizes

    @property
    def ouput_array_sizes_2d(self):
        """Returns the 2D output array sizes for a single run."""
        return SingleRunOutputSizes.from_solver(self)

    @property
    def output_array_sizes_3d(self):
        """Returns the 3D output array sizes for a batch of runs."""
        return BatchOutputSizes.from_solver(self)

    @property
    def summary_legend_per_variable(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.summary_legend_per_variable` from the child SingleIntegratorRun object."""
        return self.single_integrator.summary_legend_per_variable

    @property
    def saved_state_indices(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.saved_state_indices` from the child SingleIntegratorRun object."""
        return self.single_integrator.saved_state_indices

    @property
    def saved_observable_indices(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.saved_observable_indices` from the child SingleIntegratorRun object."""
        return self.single_integrator.saved_observable_indices

    @property
    def summarised_state_indices(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.summarised_state_indices` from the child SingleIntegratorRun object."""
        return self.single_integrator.summarised_state_indices

    @property
    def summarised_observable_indices(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.summarised_observable_indices` from the child SingleIntegratorRun object."""
        return self.single_integrator.summarised_observable_indices

    @property
    def active_output_arrays(self) -> "ActiveOutputs": # noqa: F821
        """Exposes :attr:`~CuMC.ForwardSim.BatchOutputArrays.OutputArrays.active_outputs` from the child OutputArrays object."""
        self.output_arrays.allocate()
        return self.output_arrays.active_outputs

    @property
    def state_dev_array(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchOutputArrays.OutputArrays.state` from the child OutputArrays object."""
        return self.output_arrays.state

    @property
    def observables_dev_array(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchOutputArrays.OutputArrays.observables` from the child OutputArrays object."""
        return self.output_arrays.observables

    @property
    def state_summaries_dev_array(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchOutputArrays.OutputArrays.state_summaries` from the child OutputArrays object."""
        return self.output_arrays.state_summaries

    @property
    def observable_summaries_dev_array(self):
        """Exposes :attr:`~CuMC.ForwardSim.BatchOutputArrays.OutputArrays.observable_summaries` from the child OutputArrays object."""
        return self.output_arrays.observable_summaries

    @property
    def save_time(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.save_time` from the child SingleIntegratorRun object."""
        return self.single_integrator.save_time

    def enable_profiling(self):
        """
        Enable CUDA profiling for the solver. This will allow you to profile the performance of the solver on the
        GPU, but will slow things down.
        """
        # Consider disabling optimisation and enabling debug and line info for profiling
        self.compile_settings.profileCUDA = True

    def disable_profiling(self):
        """
        Disable CUDA profiling for the solver. This will stop profiling the performance of the solver on the GPU,
        but will speed things up.
        """
        self.compile_settings.profileCUDA  = False

    @property
    def output_types(self):
        """Exposes :attr:`~CuMC.ForwardSim.integrators.SingleIntegratorRun.output_types` from the child SingleIntegratorRun object."""
        return self.single_integrator.output_types