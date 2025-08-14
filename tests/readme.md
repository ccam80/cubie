# Testing idioms and strategies for cubie

## Strategy
While learning Pytest, I have cycled through a number of strategies. The lowest-level components in this project are
compiled CUDA device functions, which are heavily problem-specific. This means that in order to test a single case,
We must pass a vast array of parameters, and we cannot easily test individual components in isoolation without completely
configuring the various components. 

Function-scope fixtures have emerged as a useful way to set up the environment for each test. Each object under test
(or involved in a test) is instantiated with some default set of parameters, usually at a function scope. The default
parameters are overridden through an "override" fixture, which can be edited through pytest.mark.parametrize.
In this way, we can indirectly parametrize tests through the override fixtures, and reuuse the parametrized settings
in other fixtures. The override, settings, and objects are kept separate to the objects to overcome a long-forgotten
error with direct/indirect parametrization. It's readable enough, so we're keeping it consistent. For a trivial example:

Here's an example:

```python
import pytest
import numpy as np
from cubie.memory.mem_manager import ArrayRequest, MemoryManager
from numba import cuda


class DummyClass:
    def __init__(self, proportion=None, stream_group=None, native_stride_order=None):
        self.proportion = proportion
        self.stream_group = stream_group
        self.native_stride_order = native_stride_order


@pytest.fixture(scope="function")
def array_request_override(request):
    return request.param if hasattr(request, 'param') else {}


@pytest.fixture(scope="function")
def array_request_settings(array_request_override):
    """Fixture to provide settings for ArrayRequest."""
    defaults = {'shape'       : (1, 1, 1), 'dtype': np.float32, 'memory': 'device',
                '_stride_order': ("time", "run", "variable")}
    if array_request_override:
        for key, value in array_request_override.items():
            if key in defaults:
                defaults[key] = value
    return defaults


@pytest.fixture(scope="function")
def array_request(array_request_settings):
    return ArrayRequest(**array_request_settings)


@pytest.fixture(scope="function")
def expected_single_array(array_request_settings):
    arr_request = array_request_settings
    if arr_request['memory'] == 'device':
        arr = cuda.device_array(array_request_settings['shape'], dtype=array_request_settings['dtype'])
    elif arr_request['memory'] == 'pinned':
        arr = cuda.pinned_array(array_request_settings['shape'], dtype=array_request_settings['dtype'])
    elif arr_request['memory'] == 'mapped':
        arr = cuda.mapped_array(array_request_settings['shape'], dtype=array_request_settings['dtype'])
    elif arr_request['memory'] == 'managed':
        raise NotImplementedError("Managed memory not implemented")
    else:
        raise ValueError(f"Invalid memory type: {arr_request['memory']}")
    return arr


@pytest.mark.parametrize("array_request_override", [{'shape': (20000,), 'dtype': np.float64}], indirect=True)
def test_array_request_instantiation(array_request):
    assert array_request.shape == (20000,)
    assert array_request.dtype == np.float64
    assert array_request.memory == 'device'
    assert array_request._stride_order == ("time", "run", "variable")


@pytest.mark.parametrize("array_request_override",
                         [{'shape': (20000,), 'dtype': np.float64}, {'memory': 'pinned'}, {'memory': 'mapped'}],
                         indirect=True)
def test_array_response(array_request, array_request_settings, expected_single_array):
    mgr = MemoryManager()
    instance = DummyClass()
    mgr.register(instance)
    resp = mgr.allocate_all(instance, {'test': array_request})
    arr = resp['test']

    # Can't directly check for equality as they'll be at different addresses
    assert arr.shape == expected_single_array.shape
    assert type(arr) == type(expected_single_array)
    assert arr.nbytes == expected_single_array.nbytes
    assert arr.strides == expected_single_array.strides
    assert arr.dtype == expected_single_array.dtype

```
Most tests are structured like this, so that if we need to test a specific function, we can only change the relevant
parameters.

As many steps in the chain of classes and subclasses and settings dicts etc are made available as fixtures as possible,
to make it easier to test that parameter changes flow throught the program.

Cuda device function have an associated test_kernel that runs it with given inputs and makes the outputs available for
assertions.

## Test structure
The test directory is structured to match the cubie package structure. A test for a given module should be in the same relative
directory as the module itself. To make it easier to create tests for new sublclasses (i.e. different integration algorithms,
different ODE systems), the common tests for a given class type are in a \*Tester class, which is ignored by pytest.
When creating a test set for a new subclass, you subclass the relevant \*tester class, and override the methods that are
specifically marked "OVERRIDE THIS METHOD". The rest of the test machinery should proceed as expected once you've overridden
these few methods to be specific to your subclass.

Classes are also sometimes used to make identifying tests easier, in cases where multiple classes are tested in a single module.

Each folder in tests has an \_\_init\_\_.py file, which allows us to import the tests as a package. You can import
things using a standard import statement, e.g.:
```python
from tests.integrators.algorithms import LoopAlgorithmTester
```
