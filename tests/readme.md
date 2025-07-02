# Testing idioms and strategies for CuMC

## Strategy
While learning Pytest, I have cycled through a number of strategies. The lowest-level components in this project are
compiled CUDA device functions, which are heavily problem-specific. This means that in order to test a single case,
We must pass a vast array of parameters, and we cannot easily test individual components in isoolation without completely
configuring the various components. 

Function-scope fixtures have emerged as a useful way to set up the environment for each test. Each set of parameters, 
grouped into functional units such as "loop_compile_settings" or "loop_run_settings" for example, has one fixture or 
function that sets up a default set of parameters. A separate fixture accepts a "request" object, and updates the
default parameter set with the request.params. A third fixture provides the updated parameters to the test, without
being explicitly parameterized. In this way, we can choose whether or not to parameterize each component in the test.

Here's an example:
```python
def loop_compile_settings(**kwargs):
    """The standard set of kwargs, some of which aren't used by certain algorithms (like dtmax for a fixed step)."""
    loop_compile_settings_dict = {'dt_min': 0.001,
                                  'dt_max': 0.01,
                                  'dt_save': 0.01,
                                  'dt_summarise': 0.1,
                                  'atol': 1e-6,
                                  'rtol': 1e-3,
                                  'saved_states': [0, 1],  # Default to first state
                                  'saved_observables': [],  # Default to no observables
                                  'output_functions': ["state"],
                                  'n_peaks': 0}
    loop_compile_settings_dict.update(kwargs)
    return loop_compile_settings_dict


@pytest.fixture(scope='function')
def compile_settings_overrides(request):
    return request.param

@pytest.fixture(scope='function')
def compile_settings(request, compile_settings_overrides):
    """
    Create a dictionary of compile settings for the loop function.

    Usage example:
    @pytest.mark.parametrize("compile_settings_overrides", [{'dt_min': 0.001, 'dt_max': 0.01}], indirect=True)
    def test_compile_settings(compile_settings_overrides):
        ...
    """
    return loop_compile_settings(**compile_settings_overrides)

```
loop_compile_settings returns an updated default dict of parameters, the updates for which can be passed to it as an 
argument (unlike a fixture). compile_settings_overrides is a fixture that accepts a request object, which is the one
which we parametrize in the test. compile_settings returns the output of the loop_compile_settings function. If we don't
parametrize the test, it just returns the default dict, and the test doesn't see anything different about the compile_settings
fixture. If we do parametrize the test, it returns the updated dict, which is then passed to the test function.
A test using this might look like:
```python
@pytest.mark.parametrize("compile_settings_overrides", [{'dt_min': 0.009, 'dt_max': 0.09}], indirect=True)
def test_compile_settings(compile_settings):
    assert compile_settings['dt_min'] == 0.001
    assert compile_settings['dt_max'] == 0.01
```

the test itself takes the unparameterized compile_settings fixture, and we _indirectly_ pass the parameters to the intermediate
update fixture.

Most tests are structured like this, so that if we need to test a specific function, we can only change the relevant
parameters.

The whole chain of class -> instance -> built instance -> built function is made available as fixtures, so that you can
pick and choose which parts to test - for example, you might want to test that a modification to a lower-level component
is successfully passed through to the component under test on instantiation, without worrying about the compile and run
process.

Each device function has an associated test_kernel that runs it with given inputs and makes the outputs available for
assertions.

## Test structure
The test directory is structured to match the CuMC package structure. A test for a given module should be in the same relative
directory as the module itself. To make it easier to create tests for new sublclasses (i.e. different integration algorithms,
different ODE systems), the common tests for a given class type are in a \*Tester class, which is ignored by pytest.
When creating a test set for a new subclass, you subclass the relevant \*tester class, and override the methods that are
specifically marked "OVERRIDE THIS METHOD". The rest of the test machinery should proceed as expected once you've overridden
these few methods to be specific to your subclass.

Each folder in tests has an \_\_init\_\_.py file, which allows us to import the tests as a package. You can import
things using a standard import statement, e.g.:
```python
from tests.ForwardSim.integrators.algorithms.LoopAlgorithmTester import LoopAlgorithmTester
```

I installed the project in editable mode, so that the tests can import the package as if it were installed. You may/may not
need to do this, I am unsure about the intricacies of how pytest works with imports..