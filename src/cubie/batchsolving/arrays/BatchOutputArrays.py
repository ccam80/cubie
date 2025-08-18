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
    """Container for output arrays"""
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
        """Factory method for creating a new array"""
        return cls(memory_type="host")

    @classmethod
    def device_factory(cls):
        """Factory method for creating a new array"""
        return cls(memory_type="mapped")

@attrs.define
class ActiveOutputs:
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
        """Update the active outputs based on the provided OutputArrays
        instance."""
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
    """ Manages batch integration output arrays between the host and device.
    This class is initialised with a BatchOutputSizes instance (which is drawn
    from a solver instance using the from_solver factory method), which sets
    the allowable 3d array sizes from the ODE system's data and run settings.
    Once initialised, the object can be updated with a solver instance to
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
        self.update_from_solver(solver_instance)
        self.allocate()

    @property
    def active_outputs(self) -> ActiveOutputs:
        """ Check which outputs are requested, treating size-1 arrays as an
        artefact of the default allocation."""
        self._active_outputs.update_from_outputarrays(self)
        return self._active_outputs

    @property
    def state(self):
        return self.host.state

    @property
    def observables(self):
        return self.host.observables

    @property
    def state_summaries(self):
        return self.host.state_summaries

    @property
    def observable_summaries(self):
        return self.host.observable_summaries

    @property
    def device_state(self):
        return self.device.state

    @property
    def device_observables(self):
        return self.device.observables

    @property
    def device_state_summaries(self):
        return self.device.state_summaries

    @property
    def device_observable_summaries(self):
        return self.device.observable_summaries

    @classmethod
    def from_solver(cls,
                    solver_instance: "BatchSolverKernel") -> "OutputArrays":
        """
        Create a OutputArrays instance from a solver instance. Does not
        allocate, just sets up sizes
        """
        sizes = BatchOutputSizes.from_solver(solver_instance).nonzero
        return cls(sizes=sizes,
                   precision=solver_instance.precision,
                   memory_manager=solver_instance.memory_manager,
                   stream_group=solver_instance.stream_group)

    def update_from_solver(self, solver_instance: "BatchSolverKernel"):
        """
        Update the sizes and precision of the OutputArrays instance from a
        solver instance.

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
        """ Copy mapped array over slice of host array """
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
        """ No need to initialise device to zeros.

        Unless chunk calculations in time leave a dangling sample at the
        end. Which I guess is possible, but not expected."""
        pass
