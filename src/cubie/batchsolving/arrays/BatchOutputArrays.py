"""
Batch Output Arrays Module.

This module provides classes for managing output arrays in batch integration
operations, including containers for storing results and managers for handling
memory allocation and data transfer between host and device.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel

import attrs
import attrs.validators as val
import numpy as np

from cubie.outputhandling.output_sizes import BatchOutputSizes
from cubie.batchsolving.arrays.BaseArrayManager import (BaseArrayManager,
                                                        ArrayContainer)
from cubie.batchsolving import ArrayTypes

@attrs.define(slots=False)
class OutputArrayContainer(ArrayContainer):
    """
    Container for batch output arrays.

    This container holds the output arrays generated during batch integration,
    including state trajectories, observables, and their summaries.

    Parameters
    ----------
    state : ArrayTypes, optional
        State variable trajectories over time.
    observables : ArrayTypes, optional
        Observable variable trajectories over time.
    state_summaries : ArrayTypes, optional
        Summary statistics for state variables.
    observable_summaries : ArrayTypes, optional
        Summary statistics for observable variables.
    stride_order : tuple[str, ...], default=("time", "run", "variable")
        Order of array dimensions.
    _memory_type : str, default="device"
        Type of memory allocation.

    Notes
    -----
    This class uses attrs for automatic initialization and validation.
    The stride_order specifies how the 3D arrays are organized in memory.
    """
    state: ArrayTypes = attrs.field(
            default=None)
    observables: ArrayTypes = attrs.field(
            default=None)
    state_summaries: ArrayTypes = attrs.field(
            default=None)
    observable_summaries: ArrayTypes = attrs.field(
            default=None)
    stride_order: tuple[str,...] = attrs.field(
            default=("time", "run","variable"),
            init=False)
    _memory_type: str = attrs.field(
            default="device",
            validator=val.in_(["device", "mapped", "pinned", "managed", "host"]))

    @classmethod
    def host_factory(cls):
        """
        Create a new host memory container.

        Returns
        -------
        OutputArrayContainer
            A new container configured for host memory.
        """
        return cls(memory_type="host")

    @classmethod
    def device_factory(cls):
        """
        Create a new device memory container.

        Returns
        -------
        OutputArrayContainer
            A new container configured for mapped memory.
        """
        return cls(memory_type="mapped")

@attrs.define
class ActiveOutputs:
    """
    Track which output arrays are actively being used.

    This class provides boolean flags indicating which output types are
    currently active based on array sizes and allocation status.

    Parameters
    ----------
    state : bool, default=False
        Whether state output is active.
    observables : bool, default=False
        Whether observables output is active.
    state_summaries : bool, default=False
        Whether state summaries output is active.
    observable_summaries : bool, default=False
        Whether observable summaries output is active.
    """
    state: bool = attrs.field(
            default=False,
            validator=val.instance_of(bool))
    observables: bool = attrs.field(
            default=False,
            validator=val.instance_of(bool))
    state_summaries: bool = attrs.field(
            default=False,
            validator=val.instance_of(bool))
    observable_summaries: bool = attrs.field(
            default=False,
            validator=val.instance_of(bool))

    def update_from_outputarrays(self, output_arrays: "OutputArrays"):
        """
        Update active outputs based on OutputArrays instance.

        Parameters
        ----------
        output_arrays : OutputArrays
            The OutputArrays instance to check for active outputs.

        Notes
        -----
        An output is considered active if the corresponding array exists
        and has more than one element (size > 1).
        """
        self.state = (output_arrays.host.state is not None and
                      output_arrays.host.state.size > 1)
        self.observables = (output_arrays.host.observables is not None and
                            output_arrays.host.observables.size > 1)
        self.state_summaries = (output_arrays.host.state_summaries is not None and
                                output_arrays.host.state_summaries.size > 1)
        self.observable_summaries = (output_arrays.host.observable_summaries is
                                     not None and
                                     output_arrays.host.observable_summaries.size
                                     > 1)

@attrs.define
class OutputArrays(BaseArrayManager):
    """
    Manage batch integration output arrays between host and device.

    This class manages the allocation, transfer, and synchronization of output
    arrays generated during batch integration operations. It handles state
    trajectories, observables, and summary statistics.

    Parameters
    ----------
    _sizes : BatchOutputSizes
        Size specifications for the output arrays.
    host : OutputArrayContainer
        Container for host-side arrays.
    device : OutputArrayContainer
        Container for device-side arrays.
    _active_outputs : ActiveOutputs
        Tracker for which outputs are currently active.

    Notes
    -----
    This class is initialized with a BatchOutputSizes instance (which is drawn
    from a solver instance using the from_solver factory method), which sets
    the allowable 3D array sizes from the ODE system's data and run settings.
    Once initialized, the object can be updated with a solver instance to
    update the expected sizes, check the cache, and allocate if required.
    """
    _sizes: BatchOutputSizes = attrs.field(
            factory=BatchOutputSizes,
            validator=val.instance_of(BatchOutputSizes))
    host: OutputArrayContainer = attrs.field(
            factory=OutputArrayContainer.host_factory,
            validator=val.instance_of(OutputArrayContainer),
            init=True)
    device: OutputArrayContainer = attrs.field(
            factory=OutputArrayContainer.device_factory,
            validator=val.instance_of(OutputArrayContainer),
            init=False)
    _active_outputs: ActiveOutputs = attrs.field(
            default=ActiveOutputs(),
            validator=val.instance_of(ActiveOutputs),
            init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.host._memory_type = "host"
        self.device._memory_type = "mapped"

    def update(self,
                 solver_instance) -> "OutputArrays":
        """
        Update output arrays from solver instance.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            The solver instance providing configuration and sizing information.

        Returns
        -------
        OutputArrays
            Self, for method chaining.
        """
        self.update_from_solver(solver_instance)
        self.allocate()

    @property
    def active_outputs(self) -> ActiveOutputs:
        """
        Get currently active output types.

        Returns
        -------
        ActiveOutputs
            Object indicating which output arrays are active.

        Notes
        -----
        Checks which outputs are requested, treating size-1 arrays as an
        artifact of the default allocation.
        """
        self._active_outputs.update_from_outputarrays(self)
        return self._active_outputs

    @property
    def state(self):
        """
        Get host state array.

        Returns
        -------
        ArrayTypes
            The state array from the host container.
        """
        return self.host.state

    @property
    def observables(self):
        """
        Get host observables array.

        Returns
        -------
        ArrayTypes
            The observables array from the host container.
        """
        return self.host.observables

    @property
    def state_summaries(self):
        """
        Get host state summaries array.

        Returns
        -------
        ArrayTypes
            The state summaries array from the host container.
        """
        return self.host.state_summaries

    @property
    def observable_summaries(self):
        """
        Get host observable summaries array.

        Returns
        -------
        ArrayTypes
            The observable summaries array from the host container.
        """
        return self.host.observable_summaries

    @property
    def device_state(self):
        """
        Get device state array.

        Returns
        -------
        ArrayTypes
            The state array from the device container.
        """
        return self.device.state

    @property
    def device_observables(self):
        """
        Get device observables array.

        Returns
        -------
        ArrayTypes
            The observables array from the device container.
        """
        return self.device.observables

    @property
    def device_state_summaries(self):
        """
        Get device state summaries array.

        Returns
        -------
        ArrayTypes
            The state summaries array from the device container.
        """
        return self.device.state_summaries

    @property
    def device_observable_summaries(self):
        """
        Get device observable summaries array.

        Returns
        -------
        ArrayTypes
            The observable summaries array from the device container.
        """
        return self.device.observable_summaries

    @classmethod
    def from_solver(cls,
                    solver_instance: "BatchSolverKernel") -> "OutputArrays":
        """
        Create an OutputArrays instance from a solver.

        Does not allocate arrays, just sets up size specifications.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            The solver instance to extract configuration from.

        Returns
        -------
        OutputArrays
            A new OutputArrays instance configured for the solver.
        """
        sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
        return cls(sizes=sizes,
                   precision=solver_instance.precision,
                   memory_manager=solver_instance.memory_manager,
                   stream_group=solver_instance.stream_group)

    def update_from_solver(self, solver_instance: "BatchSolverKernel"):
        """
        Update sizes and precision from solver instance.

        Parameters
        ----------
        solver_instance : BatchSolverKernel
            The solver instance to update from.

        Notes
        -----
        This method updates the array sizes and precision, and reallocates
        host arrays if their sizes don't match the expected dimensions.
        """
        self._sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
        host_dict = {k: v for k, v in self.host.__dict__.items()
                     if not k.startswith("_")}
        size_matches = self.check_sizes(host_dict, location="host")
        for array, match in size_matches.items():
            if not match:
                shape = getattr(self._sizes, array)
                setattr(self.host, array, np.zeros(shape=shape,
                                                     dtype=self._precision))
        self._precision = solver_instance.precision

    def finalise(self, host_indices):
        """
        Copy mapped arrays to host array slices.

        Parameters
        ----------
        host_indices : slice or array-like
            Indices for the chunk being finalized.

        Notes
        -----
        This method copies mapped device arrays over the specified slice
        of host arrays. The copy operation may trigger CUDA runtime
        synchronization.
        """
        chunk_index = self.host.stride_order.index(self._chunk_axis)
        slice_tuple = [slice(None)] * 3
        slice_tuple[chunk_index] = host_indices
        slice_tuple = tuple(slice_tuple)
        for array_name, array in self.host.__dict__.items():
            if not array_name.startswith("_"):
                array[slice_tuple] = getattr(self.device, array_name).copy()
                # I'm not sure that I can stream this? If I just overwrite,
                # that might jog the cuda runtime to synchronize.

    def initialise(self, host_indices):
        """
        Initialize device arrays before kernel execution.

        Parameters
        ----------
        host_indices : slice or array-like
            Indices for the chunk being initialized.

        Notes
        -----
        No initialization to zeros is needed unless chunk calculations in time
        leave a dangling sample at the end, which is possible but not expected.
        """
        pass
