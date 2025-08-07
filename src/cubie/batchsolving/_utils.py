from os import environ
from typing import Union, Tuple


def ensure_nonzero_size(value: Union[int, Tuple[int, ...]]) -> Union[
    int, Tuple[int, ...]]:
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


if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    def is_cuda_array(value):
        return hasattr(value, 'shape')
else:
    from numba.cuda import is_cuda_array


def cuda_array_validator(instance, attribute, value, dimensions=None):
    if dimensions is None:
        return is_cuda_array(value)
    else:
        return is_cuda_array(value) and len(value.shape) == dimensions


def optional_cuda_array_validator(instance, attribute, value, dimensions=None):
    if value is None:
        return True
    return cuda_array_validator(instance, attribute, value, dimensions)


def optional_cuda_array_validator_3d(instance, attribute, value):
    return optional_cuda_array_validator(instance, attribute, value,
                                         dimensions=3)


def optional_cuda_array_validator_2d(instance, attribute, value):
    return optional_cuda_array_validator(instance, attribute, value,
                                         dimensions=2)


def cuda_array_validator_3d(instance, attribute, value):
    return cuda_array_validator(instance, attribute, value, dimensions=3)


def cuda_array_validator_2d(instance, attribute, value):
    return cuda_array_validator(instance, attribute, value, dimensions=2)
