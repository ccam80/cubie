from abc import abstractmethod

import attrs
import numpy as np

from cubie.outputhandling import LoopBufferSizes



@attrs.define
class BaseStepConfig:
    """Configuration settings for a single integration step.

    Explicit algorithms do not access the full range of fields.
    """
    precision = attrs.field(default=np.float32)
    buffer_sizes: LoopBufferSizes = attrs.field(
        factory=LoopBufferSizes,
        validator=attrs.validators.instance_of(LoopBufferSizes)
    )


    @abstractmethod
    @property
    def is_implicit(self):
        raise NotImplementedError("is_implicit not implemented")


    @property
    def n(self) -> int:
        """Number of stages."""
        return self.buffer_sizes.state

    threads_per_step: int = attrs.field(default=1)
