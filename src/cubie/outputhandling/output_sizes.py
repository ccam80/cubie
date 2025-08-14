from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
    from cubie.outputhandling.output_functions import OutputFunctions
    from cubie.systemmodels.systems.GenericODE import GenericODE

import attrs
from cubie.batchsolving._utils import ensure_nonzero_size
from numpy import ceil
from abc import ABC

@attrs.define
class ArraySizingClass(ABC):
    """Base class for all array sizing classes. Provides a nonzero method
    which returns a copy of the object where
    all sizes have a minimum of one element, useful for allocating memory.."""

    @property
    def nonzero(self) -> "ArraySizingClass":
        new_obj = attrs.evolve(self)
        for field in attrs.fields(self.__class__):
            value = getattr(new_obj, field.name)
            if isinstance(value, (int, tuple)):
                setattr(new_obj, field.name, ensure_nonzero_size(value))
        return new_obj


@attrs.define
class SummariesBufferSizes(ArraySizingClass):
    """Given heights of buffers, return them directly under state and
    observable aliases. Most useful when called
    with an adapter factory - for example, give it an output_functions
    object, and it returns sizes without awkward
    property names from a more cluttered namespace"""
    state: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    observables: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    per_variable: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))

    @classmethod
    def from_output_fns(cls,
                        output_fns: "OutputFunctions") -> \
            "SummariesBufferSizes":
        return cls(output_fns.state_summaries_buffer_height,
                   output_fns.observable_summaries_buffer_height,
                   output_fns.summaries_buffer_height_per_var, )


@attrs.define
class LoopBufferSizes(ArraySizingClass):
    """Dataclass which presents the sizes of all buffers used in the inner
    loop of an integrator - system-size based
    buffers like state, dxdt and summary buffers derived from output
    functions information."""
    state_summaries: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    observable_summaries: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    state: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    observables: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    dxdt: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    parameters: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    drivers: Optional[int] = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))

    @classmethod
    def from_system_and_output_fns(cls, system: "GenericODE",
                                   output_fns: "OutputFunctions",
                                   ) -> "LoopBufferSizes":
        summary_sizes = SummariesBufferSizes.from_output_fns(output_fns)
        system_sizes = system.sizes
        obj = cls(summary_sizes.state, summary_sizes.observables,
                  system_sizes.states, system_sizes.observables,
                  system_sizes.states, system_sizes.parameters,
                  system_sizes.drivers, )
        return obj

    @classmethod
    def from_solver(cls,
                    solver_instance: "BatchSolverKernel") -> "LoopBufferSizes":
        """
        Create a LoopBufferSizes instance from a BatchSolverKernel object.
        """
        system_sizes = solver_instance.system_sizes
        summary_sizes = solver_instance.summaries_buffer_sizes
        return cls(summary_sizes.state, summary_sizes.observables,
                   system_sizes.states, system_sizes.observables,
                   system_sizes.states, system_sizes.parameters,
                   system_sizes.drivers, )


@attrs.define
class OutputArrayHeights(ArraySizingClass):
    state: int = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    observables: int = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    state_summaries: int = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    observable_summaries: int = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))
    per_variable: int = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int))

    @classmethod
    def from_output_fns(cls,
                        output_fns: "OutputFunctions") -> "OutputArrayHeights":
        state = output_fns.n_saved_states + 1 * output_fns.save_time
        observables = output_fns.n_saved_observables
        state_summaries = output_fns.state_summaries_output_height
        observable_summaries = output_fns.observable_summaries_output_height
        per_variable = output_fns.summaries_output_height_per_var
        obj = cls(state, observables, state_summaries, observable_summaries,
                  per_variable, )
        return obj


@attrs.define
class SingleRunOutputSizes(ArraySizingClass):
    """ Returns 2d single-slice output array sizes for a single integration
    run."""
    state: Tuple[int, int] = attrs.field(
        default=(1, 1),
        validator=attrs.validators.instance_of(Tuple))
    observables: Tuple[int, int] = attrs.field(
        default=(1, 1),
        validator=attrs.validators.instance_of(Tuple))
    state_summaries: Tuple[int, int] = attrs.field(
        default=(1, 1),
        validator=attrs.validators.instance_of(Tuple))
    observable_summaries: Tuple[int, int] = attrs.field(
        default=(1, 1),
        validator=attrs.validators.instance_of(Tuple))
    stride_order: Tuple[str, ...] = attrs.field(
        default=("time", "variable"),
        validator=attrs.validators.deep_iterable(
            attrs.validators.in_(["time", "variable"])))

    @classmethod
    def from_solver(cls,
                    solver_instance: "BatchSolverKernel") -> \
            "SingleRunOutputSizes":
        """
        Create a SingleRunOutputSizes instance from a BatchSolverKernel object.
        """
        heights = solver_instance.output_array_heights
        output_samples = solver_instance.output_length
        summarise_samples = solver_instance.summaries_length

        state = (output_samples, heights.state)
        observables = (output_samples, heights.observables)
        state_summaries = (summarise_samples, heights.state_summaries)
        observable_summaries = (summarise_samples,
                                heights.observable_summaries)
        obj = cls(state, observables, state_summaries, observable_summaries, )

        return obj

    @classmethod
    def from_output_fns_and_run_settings(cls, output_fns, run_settings):
        """Only used for testing, otherwise the higher-level from_solver
        method is used"""
        heights = OutputArrayHeights.from_output_fns(output_fns)
        output_samples = int(
                ceil(run_settings.duration / run_settings.dt_save))
        summarise_samples = int(
                ceil(run_settings.duration / run_settings.dt_summarise))

        state = (output_samples, heights.state)
        observables = (output_samples, heights.observables)
        state_summaries = (summarise_samples, heights.state_summaries)
        observable_summaries = (summarise_samples,
                                heights.observable_summaries)
        obj = cls(state, observables, state_summaries, observable_summaries, )

        return obj


@attrs.define
class BatchInputSizes(ArraySizingClass):
    """ Returns 3d output array sizes for a batch of integration runs,
    given a singleintegrator sizes object and
    num_runs"""
    initial_values: Tuple[int, int] = attrs.field(
        default=(1, 1),
        validator=attrs.validators.instance_of(Tuple))
    parameters: Tuple[int, int] = attrs.field(
        default=(1, 1),
        validator=attrs.validators.instance_of(Tuple))
    forcing_vectors: Tuple[int, Optional[int]] = attrs.field(
        default=(1, None),
        validator=attrs.validators.instance_of(Tuple))

    stride_order: Tuple[str, ...] = attrs.field(
        default=("run", "variable"),
        validator=attrs.validators.deep_iterable(
            attrs.validators.in_(["run", "variable"])))

    @classmethod
    def from_solver(
            cls,
            solver_instance: "BatchSolverKernel") -> "BatchInputSizes":
        """
        Create a BatchInputSizes instance from a BatchSolverKernel object.
        """
        loopBufferSizes = LoopBufferSizes.from_solver(solver_instance)
        num_runs = solver_instance.num_runs
        initial_values = (num_runs, loopBufferSizes.state)
        parameters = (num_runs, loopBufferSizes.parameters)
        forcing_vectors = (loopBufferSizes.drivers, None)

        obj = cls(initial_values, parameters, forcing_vectors)
        return obj


@attrs.define
class BatchOutputSizes(ArraySizingClass):
    """ Returns 3d output array sizes for a batch of integration runs,
    given a singleintegrator sizes object and
    num_runs"""
    state: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1),
        validator=attrs.validators.instance_of(Tuple))
    observables: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1),
        validator=attrs.validators.instance_of(Tuple))
    state_summaries: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1),
        validator=attrs.validators.instance_of(Tuple))
    observable_summaries: Tuple[int, int, int] = attrs.field(
        default=(1, 1, 1),
        validator=attrs.validators.instance_of(Tuple))
    stride_order: Tuple[str, ...] = attrs.field(
        default=("time", "run", "variable"),
        validator=attrs.validators.deep_iterable(
            attrs.validators.in_(["time", "run", "variable"])))

    @classmethod
    def from_solver(cls,
                    solver_instance: "BatchSolverKernel") -> \
            "BatchOutputSizes":
        """
        Create a BatchOutputSizes instance from a SingleIntegratorRun object.
        """
        single_run_sizes = SingleRunOutputSizes.from_solver(solver_instance)
        num_runs = solver_instance.num_runs
        state = (single_run_sizes.state[0], num_runs,
                 single_run_sizes.state[1])
        observables = (single_run_sizes.observables[0], num_runs,
                       single_run_sizes.observables[1])
        state_summaries = (single_run_sizes.state_summaries[0], num_runs,
                           single_run_sizes.state_summaries[1])
        observable_summaries = (single_run_sizes.observable_summaries[0],
                                num_runs,
                                single_run_sizes.observable_summaries[1])
        obj = cls(state, observables, state_summaries, observable_summaries, )
        return obj
