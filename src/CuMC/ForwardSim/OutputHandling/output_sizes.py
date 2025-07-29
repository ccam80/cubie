import attrs
from typing import Optional, Tuple, Union


@attrs.define
class ArraySizingClass:
    """Base class for all array sizing classes. Provides a nonzero method which returns a copy of the object where
    all sizes have a minimum of one element, useful for allocating memory.."""

    @property
    def nonzero(self) -> "ArraySizingClass":
        new_obj = attrs.evolve(self)
        for field in attrs.fields(self.__class__):
            value = getattr(new_obj, field.name)
            if isinstance(value, (int, tuple)):
                setattr(new_obj, field.name, self._ensure_nonzero_size(value))
        return new_obj

    def _ensure_nonzero_size(self, value: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
        """Helper function to replace zeros with ones"""
        if isinstance(value, int):
            return max(1, value)
        elif isinstance(value, tuple):
            if any(v == 0 for v in value):
                return tuple(1 for v in value)
            else:
                return value
        else:
            return value


@attrs.define
class SummariesBufferSizes(ArraySizingClass):
    """Given heights of buffers, return them directly under state and observable aliases. Most useful when called
    with an adapter factory - for example, give it an output_functions object, and it returns sizes without awkward 
    property names from a more cluttered namespace"""
    state: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observables: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    per_variable: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))

    @classmethod
    def from_output_fns(cls, output_fns: "OutputFunctions") -> "SummariesBufferSizes":  # noqa: F821
        return cls(output_fns.state_summaries_buffer_height,
                   output_fns.observable_summaries_buffer_height,
                   output_fns.summaries_buffer_height_per_var,
                   )


@attrs.define
class LoopBufferSizes(ArraySizingClass):
    """Dataclass which presents the sizes of all buffers used in the inner loop of an integrator - system-size based
    buffers like state, dxdt and summary buffers derived from output functions information."""
    state_summaries: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observable_summaries: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    state: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observables: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    dxdt: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    parameters: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    drivers: Optional[int] = attrs.field(default=1, validator=attrs.validators.instance_of(int))

    @classmethod
    def from_system_and_output_fns(cls, system: "GenericODE", output_fns: "OutputFunctions") -> "LoopBufferSizes":  # noqa: F821
        summary_sizes = SummariesBufferSizes.from_output_fns(output_fns)
        system_sizes = system.sizes
        obj = cls(summary_sizes.state,
                  summary_sizes.observables,
                  system_sizes.states,
                  system_sizes.observables,
                  system_sizes.states,
                  system_sizes.parameters,
                  system_sizes.drivers,
                  )
        return obj

    @classmethod
    def from_solver(cls, solver_instance: "BatchSolverKernel") -> "LoopBufferSizes":
        """
        Create a LoopBufferSizes instance from a BatchSolverKernel object.
        """
        system = solver_instance.system
        output_fns = solver_instance.single_integrator._output_functions
        return cls.from_system_and_output_fns(system, output_fns)

@attrs.define
class OutputArrayHeights(ArraySizingClass):
    state: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observables: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    state_summaries: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    observable_summaries: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))
    per_variable: int = attrs.field(default=1, validator=attrs.validators.instance_of(int))

    @classmethod
    def from_output_fns(cls, output_fns: "OutputFunctions") -> "OutputArrayHeights":  # noqa: F821
        state = output_fns.n_saved_states + 1 * output_fns.save_time
        observables = output_fns.n_saved_observables
        state_summaries = output_fns.state_summaries_output_height
        observable_summaries = output_fns.observable_summaries_output_height
        per_variable = output_fns.summaries_output_height_per_var
        obj = cls(state,
                  observables,
                  state_summaries,
                  observable_summaries,
                  per_variable,
                  )
        return obj


@attrs.define
class SingleRunOutputSizes(ArraySizingClass):
    """ Returns 2d single-slice output array sizes for a single integration run."""
    state: Tuple[int, int] = attrs.field(default=(1, 1), validator=attrs.validators.instance_of(Tuple))
    observables: Tuple[int, int] = attrs.field(default=(1, 1), validator=attrs.validators.instance_of(Tuple))
    state_summaries: Tuple[int, int] = attrs.field(default=(1, 1), validator=attrs.validators.instance_of(Tuple))
    observable_summaries: Tuple[int, int] = attrs.field(default=(1, 1), validator=attrs.validators.instance_of(Tuple))

    @classmethod
    def from_solver(cls, solver_instance: "BatchSolverKernel") -> "SingleRunOutputSizes":  # noqa: F821
        """
        Create a SingleRunOutputSizes instance from a BatchSolverKernel object.
        """
        heights = solver_instance.output_heights
        output_samples = solver_instance.output_length
        summarise_samples = solver_instance.summaries_length

        state = (output_samples, heights.state)
        observables = (output_samples, heights.observables)
        state_summaries = (summarise_samples, heights.state_summaries)
        observable_summaries = (summarise_samples, heights.observable_summaries)
        obj = cls(state,
                  observables,
                  state_summaries,
                  observable_summaries,
                  )

        return obj

    @classmethod
    def from_output_fns_and_run_settings(cls, output_fns, run_settings):
        """Only used for testing, otherwise the higher-level from_solver method is used"""
        heights = OutputArrayHeights.from_output_fns(output_fns)
        output_samples = int(run_settings.duration // run_settings.dt_save)
        summarise_samples = int(run_settings.duration // run_settings.dt_summarise)

        state = (output_samples, heights.state)
        observables = (output_samples, heights.observables)
        state_summaries = (summarise_samples, heights.state_summaries)
        observable_summaries = (summarise_samples, heights.observable_summaries)
        obj = cls(state,
                  observables,
                  state_summaries,
                  observable_summaries,
                  )

        return obj

@attrs.define
class BatchOutputSizes(ArraySizingClass):
    """ Returns 3d output array sizes for a batch of integration runs, given a singleintegrator sizes object and
    num_runs"""
    state: Tuple[int, int, int] = attrs.field(default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple))
    observables: Tuple[int, int, int] = attrs.field(default=(1, 1, 1), validator=attrs.validators.instance_of(Tuple))
    state_summaries: Tuple[int, int, int] = attrs.field(default=(1, 1, 1),
                                                        validator=attrs.validators.instance_of(Tuple),
                                                        )
    observable_summaries: Tuple[int, int, int] = attrs.field(default=(1, 1, 1),
                                                             validator=attrs.validators.instance_of(Tuple),
                                                             )

    @classmethod
    def from_solver(cls, solver_instance: "BatchSolverKernel") -> "BatchOutputSizes":  # noqa: F821
        """
        Create a BatchOutputSizes instance from a SingleIntegratorRun object.
        """
        single_run_sizes = SingleRunOutputSizes.from_solver(solver_instance)
        num_runs = solver_instance.num_runs
        state = (single_run_sizes.state[0], num_runs, single_run_sizes.state[1])
        observables = (single_run_sizes.observables[0], num_runs, single_run_sizes.observables[1])
        state_summaries = (single_run_sizes.state_summaries[0], num_runs, single_run_sizes.state_summaries[1])
        observable_summaries = (single_run_sizes.observable_summaries[0],
                                num_runs,
                                single_run_sizes.observable_summaries[1]
                                )
        obj = cls(state,
                  observables,
                  state_summaries,
                  observable_summaries,
                  )
        return obj


