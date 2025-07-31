from CuMC.ForwardSim.BatchSolverKernel import BatchSolverKernel
import pytest

# Incorporate batch configurator
@pytest.fixture(scope="function")
def SolverKernel(system, loop_compile_settings, precision):
    """Fixture to create a SolverKernel instance with a ThreeChamberModel."""
    kernel = BatchSolverKernel(system=system,
                               algorithm='euler',
                               dt_min=loop_compile_settings['dt_min'],
                               dt_max=loop_compile_settings['dt_max'],
                               dt_save=loop_compile_settings['dt_save'],
                               dt_summarise=loop_compile_settings['dt_summarise'],
                               atol=loop_compile_settings['atol'],
                               rtol=loop_compile_settings['rtol'],
                               saved_states=loop_compile_settings['saved_states'],
                               saved_observables=loop_compile_settings['saved_observables'],
                               summarised_states=loop_compile_settings['summarised_states'],
                               summarised_observables=loop_compile_settings['summarised_observables'],
                               output_types=loop_compile_settings['output_functions'],
                               precision=precision,
                               profileCUDA=False,
                               )
    return kernel

def test_kernel_builds(SolverKernel):
    """Test that the SolverKernel builds without errors."""
    kernelfunc = SolverKernel.kernel

# def test_run(SolverKernel):
#     """Test that the SolverKernel can run with the provided inputs and settings."""
#     inputs = inputs_dict(SolverKernel.system, SolverKernel.system.precision)
#     outputs = SolverKernel.run(inputs)
#
#     # Check that outputs are as expected
#     assert outputs is not None, "Outputs should not be None"
#     assert isinstance(outputs, dict), "Outputs should be a dictionary"
#     assert 'states' in outputs, "Outputs should contain 'states'"
#     assert 'observables' in outputs, "Outputs should contain 'observables'"


# Test that edits made to the kernel class make it down to the furthest reaches of the children - check sizes,
# output functions, loop step config, etc.

#Run one each euler and generic algorithms to check that we get the same results as loop tests.