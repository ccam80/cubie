from CuMC.ForwardSim.SolverKernel import SolverKernel
import pytest
import numpy as np
from tests._utils import random_array

@pytest.fixture(scope="function")
def kernelclass(system, loop_compile_settings, precision):
    """Fixture to create a SolverKernel instance with a ThreeChamberModel."""
    kernel = SolverKernel(system=system,
                          algorithm='euler',
                          dt_min=loop_compile_settings['dt_min'],
                          dt_max=loop_compile_settings['dt_max'],
                          dt_save=loop_compile_settings['dt_save'],
                          dt_summarise=loop_compile_settings['dt_summarise'],
                          atol=loop_compile_settings['atol'],
                          rtol=loop_compile_settings['rtol'],
                          saved_states=loop_compile_settings['saved_states'],
                          saved_observables=loop_compile_settings['saved_observables'],
                          output_types=loop_compile_settings['output_functions'],
                          profileCUDA=False,
                          )
    kernel.build()
    return kernel


@pytest.fixture(scope="function")
def run_settings_override(request):
    """Override default run settings for kernel testing."""
    # Default run settings
    default_run_settings = {
        'numruns':        None,
        'duration':       None,
        'warmup':         None,
        'randscale':      None,
        'runs_per_block': None
        }

    # Override with any provided values
    if hasattr(request, 'param'):
        return default_run_settings.update(request.param)
    else:
        return default_run_settings


@pytest.fixture(scope="function")
def run_settings(system, loop_compile_settings, run_settings_override, precision):
    """Fixture to provide run settings for the kernel."""
    numruns = 32 if run_settings_override['numruns'] is None else run_settings_override['numruns']
    duration = 1.0 if run_settings_override['duration'] is None else run_settings_override['duration']
    warmup = 0.0 if run_settings_override['warmup'] is None else run_settings_override['warmup']
    randscale = 1.0 if run_settings_override['randscale'] is None else run_settings_override['randscale']
    runs_per_block = 32 if run_settings_override['runs_per_block'] is None else run_settings_override['runs_per_block']

    output_samples = int(np.ceil(duration / loop_compile_settings['dt_save']))
    sizes = system.sizes()
    params = random_array(precision, (int(numruns/4), sizes['n_parameters']), randscale)
    inits = random_array(precision, (4, sizes['n_states']), randscale)
    forcing_vectors = random_array(precision, (output_samples, sizes['n_drivers']), randscale)

    run_settings = {'duration':        duration,
                    'warmup':          warmup,
                    'numruns':         numruns,
                    'params':          params,
                    'inits':           inits,
                    'forcing_vectors': forcing_vectors,
                    'runs_per_block':  runs_per_block,
                    'randscale':       randscale,
                    }

    return run_settings

@pytest.mark.parametrize("system_override", ["ThreeChamber"], indirect=True)
def test_kernel_cooks(kernelclass, run_settings, precision):
    """
    Test the SolverKernel with a ThreeChamberModel and default run settings.

    This test checks if the kernel can be built and run without errors.
    """

    # Run the kernel with the provided run settings
    results = kernelclass.run(
            run_settings['duration'],
            run_settings['numruns'],
            run_settings['params'],
            run_settings['inits'],
            run_settings['forcing_vectors'],
            runs_per_block=run_settings['runs_per_block'],
            warmup=run_settings['warmup'], )
    # Check if results are returned correctly
    #TODO: Time to figure out allocation of arrays using the inputs list
    assert results is not None, "Kernel run should return results"
