import attrs
from numpy import float32


@attrs.define
class BatchSolverConfig:
    """Configuration for the solver kernel."""
    precision: type = attrs.field(default=float32,
                                  validator=attrs.validators.instance_of(type))
    algorithm: str = 'euler'
    duration: float = 1.0
    warmup: float = 0.0
    stream: int = attrs.field(default=0, validator=attrs.validators.optional(
            attrs.validators.instance_of(int, ), ), )
    profileCUDA: bool = False
