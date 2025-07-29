from _warnings import warn

import attrs
import attrs.validators as val
from numba import from_dtype
from numpy import float32, array_equal, zeros, ndarray
from numba.cuda import device_array_like, to_device
from os import environ
if environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
    from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray as DeviceNDArray
else:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
from CuMC.ForwardSim._utils import optional_cuda_array_validator, optional_cuda_array_validator_3d
from numpy.typing import NDArray
from typing import Optional


from CuMC.ForwardSim.OutputHandling.output_sizes import LoopBufferSizes




@attrs.define
class InputArrays:
    """ Manages batch integration input arrays between the host and device. This class is initialised with a
    LoopBufferSizes instance (which is drawn from a solver instance using the from_solver factory method),
    which sets the allowable array heights from the ODE system's data. Once initialised, the object can be called
    with arguments (initial_values, parameters, forcing_vectors). Each call to the the classes __call__ method will:
    - Check if the input array has changed in shape or content since the last update
    - Allocate a new device array if the shape has changed
    - Overwrite the old device array if the shape has not changed but the content has
    - Do nothing an input array has not changed.
    """
    _sizes: Optional[LoopBufferSizes] = attrs.field(
            default=None,
            validator=val.optional(val.instance_of(LoopBufferSizes)),
            )
    _precision: type = attrs.field(default=float32, validator=val.instance_of(type))
    _initial_values: Optional[NDArray] = attrs.field(default=None, validator=val.optional(val.instance_of(ndarray)))
    _parameters: Optional[NDArray] = attrs.field(default=None, validator=val.optional(val.instance_of(ndarray)))
    _forcing_vectors: Optional[NDArray] = attrs.field(default=None, validator=val.optional(val.instance_of(ndarray)))
    _default_stream: Optional[int] = attrs.field(default=0, validator=val.instance_of(int), init=False)
    _device_inits: Optional[DeviceNDArray] = attrs.field(
            default=None,
            validator=val.optional(optional_cuda_array_validator_3d),
            init=False
            )
    _device_parameters: Optional[DeviceNDArray] = attrs.field(
            default=None,
            validator=val.optional(optional_cuda_array_validator_3d),
            init=False
            )
    _device_forcing: Optional[DeviceNDArray] = attrs.field(
            default=None,
            validator=val.optional(optional_cuda_array_validator),
            init=False
            )

    _needs_reallocation: list[str] = attrs.field(factory=list, init=False)
    _needs_overwrite: list[str] = attrs.field(factory=list, init=False)

    def __attrs_post_init__(self):
        if self._initial_values is None:
            self._initial_values = zeros((1,1,1), dtype=self._precision)
        if self._parameters is None:
            self._parameters = zeros((1,1,1), dtype=self._precision)
        if self._forcing_vectors is None:
            self._forcing_vectors = zeros((1,1,1), dtype=self._precision)

        self._needs_reallocation = ["initial_values", "parameters", "forcing_vectors"]

    def __call__(self, initial_values: NDArray, parameters: NDArray, forcing_vectors: NDArray, stream=None):
        """
        Set the initial values, parameters, and forcing vectors. This is useful for reusing already allocated arrays.
        """
        if self._check_dims_vs_system(initial_values, parameters, forcing_vectors):
            self._initial_values = self._update_host_array(initial_values, self._initial_values, "initial_values")
            self._parameters = self._update_host_array(parameters, self._parameters, "parameters")
            self._forcing_vectors = self._update_host_array(forcing_vectors, self._forcing_vectors,  "forcing_vectors")
            self.to_device(stream=stream)
        else:
            warn("Provided initial values/parameters/driver arrays do not match the sizes according to the ODE "
                 "system, ignoring update")

    @property
    def initial_values(self):
        return self._initial_values

    @property
    def parameters(self):
        return self._parameters

    @property
    def forcing_vectors(self):
        return self._forcing_vectors

    @property
    def device_initial_values(self):
        return self._device_inits

    @property
    def device_parameters(self):
        return self._device_parameters

    @property
    def device_forcing_vectors(self):
        return self._device_forcing

    def _arrays_equal(self, arr1, arr2):
        """Check if two arrays are equal in shape and content."""
        if arr1 is None or arr2 is None:
            return arr1 is arr2
        return array_equal(arr1, arr2)

    def _check_dims_vs_system(self, initial_values, parameters, forcing_vectors):
        """
        Check heights of the arrays match the expected sizes. Return True if sizes match. If no sizes are provided,
        return True.
        """
        if self._sizes is None:
            return True

        match = True
        if initial_values.shape[2] != self._sizes.state:
            match = False
        if parameters.shape[2] != self._sizes.parameters:
            match = False
        if forcing_vectors.shape[2] != self._sizes.drivers:
            match = False
        return match

    def to_device(self, stream=None):
        """Allocates or writes data to the device only if the input arrays have been updated, otherwise does nothing."""
        if stream is None:
            stream = self._default_stream

        # Handle reallocations first
        for array_label in self._needs_reallocation:
            if array_label == "initial_values":
                if self._device_inits is not None:
                    del self._device_inits
                self._device_inits = to_device(self._initial_values, stream=stream)

            elif array_label == "parameters":
                if self._device_parameters is not None:
                    del self._device_parameters
                self._device_parameters = to_device(self._parameters, stream=stream)

            elif array_label == "forcing_vectors":
                if self._device_forcing is not None:
                    del self._device_forcing
                self._device_forcing = to_device(self.forcing_vectors, stream=stream)

        # Handle overwrites
        for array_label in self._needs_overwrite:
            if array_label == "initial_values":
                to_device(self._initial_values, stream=stream, to=self._device_inits)
            elif array_label == "parameters":
                to_device(self._parameters, stream=stream, to=self._device_parameters)
            elif array_label == "forcing_vectors":
                to_device(self._forcing_vectors, stream=stream, to=self._device_forcing)

        self._needs_reallocation = []
        self._needs_overwrite = []

    def _update_host_array(self, new_array: NDArray, current_array: NDArray, label: str):
        """Check for equality and shape equality, append to reallocation or overwrite lists accordingly.
        Returns the new array if changed, otherwise returns current_array unchanged."""
        if not self._arrays_equal(new_array, current_array):
            if current_array.shape != new_array.shape:
                self._needs_reallocation.append(label)
            else:
                self._needs_overwrite.append(label)
            return new_array
        return current_array

    @property
    def num_runs(self):
        """ Number of runs in the batch, as determined by the inner dimension of the initial values and parameters
        arrays. This feels like a bit of unnecessary coupling, but as it's determined by the inputs, it may end up
        being the best place for it."""
        init_runs = self._initial_values.shape[0] if self._initial_values is not None else 0
        param_runs = self._parameters.shape[0] if self._parameters is not None else 0
        return init_runs * param_runs

    @classmethod
    def from_solver(cls, solver_instance: "BatchSolverKernel") -> "InputArrays":  # noqa: F821
        """
        Create an empty instance from a solver instance, importing the heights of the parameters, initial values,
        and driver arrays from the ODE system for checking inputs against. Does not allocate host or device arrays.
        """
        sizes = LoopBufferSizes.from_solver(solver_instance)
        return cls(sizes=sizes, precision=solver_instance.precision)
