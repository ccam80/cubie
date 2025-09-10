from abc import abstractmethod
from typing import Callable

from numba import cuda
from cubie.CUDAFactory import CUDAFactory

class BaseStepController(CUDAFactory):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build(self)-> Callable:
        """Build the step control function.

        Device function signature should be:
        TODO"""
        raise NotImplementedError
