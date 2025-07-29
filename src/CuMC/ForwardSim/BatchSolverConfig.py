import attrs
from numpy import float32
from typing import Optional
from numpy.typing import ArrayLike

@attrs.define
class BatchSolverConfig:
    """Configuration for the solver kernel."""
    _precision: type = attrs.field(default=float32, validator=attrs.validators.instance_of(type))
    algorithm: str = 'euler'
    duration: float = 1.0
    warmup: float = 0.0
    parameters: Optional[ArrayLike] = None
    initial_values: Optional[ArrayLike] = None
    forcing_vectors: Optional[ArrayLike] = None
    runs_per_block: int = attrs.field(default=32, validator=attrs.validators.instance_of(int))
    stream: int = attrs.field(default=0, validator=attrs.validators.optional(attrs.validators.instance_of(
            int)))
    # Do we need to get our dirty fingers in here or can we pass them down and dig them out of the singlerun?
    # dt_min: float = 0.01
    # dt_max: float = 0.1
    # dt_save: float = 0.1
    # dt_summarise: float = 1.0
    # atol: float = 1e-6
    # rtol: float = 1e-6
    # saved_states: Optional[ArrayLike] = None
    # saved_observables: Optional[ArrayLike] = None
    # summarised_states: Optional[ArrayLike] = None
    # summarised_observables: Optional[ArrayLike] = None
    # output_types: list[str] = None
    profileCUDA: bool = False


