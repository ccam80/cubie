from CuMC.ForwardSim.BatchSolverKernel import BatchSolverKernel
from CuMC.ForwardSim.OutputHandling.output_sizes import BatchOutputSizes
import pytest
import numpy as np


def test_kernel_builds(solverkernel):
    """Test that the solver builds without errors."""
    kernelfunc = solverkernel.kernel

# def test_run(solver):
#     """Test that the solver can run with the provided inputs and settings."""
#     inputs = inputs_dict(solver.system, solver.system.precision)
#     outputs = solver.run(inputs)
#
#     # Check that outputs are as expected
#     assert outputs is not None, "Outputs should not be None"
#     assert isinstance(outputs, dict), "Outputs should be a dictionary"
#     assert 'states' in outputs, "Outputs should contain 'states'"
#     assert 'observables' in outputs, "Outputs should contain 'observables'"

def test_algorithm_change(solverkernel):
    solverkernel.update({'algorithm': 'generic'})
    assert solverkernel.single_integrator._integrator_instance.shared_memory_required == 0

def test_all_lower_plumbing(system, solverkernel):
    """Big plumbing integration check - check that config classes match exactly between an updated solver and one
    instantiated with the update settings."""
    new_settings = {
        'duration': 1.0,
        'dt_min': 0.0001,
        'dt_max': 0.01,
        'dt_save': 0.01,
        'dt_summarise': 0.1,
        'atol': 1e-2,
        'rtol': 1e-1,
        'saved_state_indices': [0,1,2],
        'saved_observable_indices': [0,1,2],
        'summarised_state_indices': [0,],
        'summarised_observable_indices': [0,],
        'output_types': ["state", "observables", "mean", "max", "rms", "peaks[3]"],
        'precision': np.float64,
    }
    solverkernel.update(new_settings)
    freshsolver = BatchSolverKernel(system,
                                          algorithm='euler',
                                          **new_settings)

    assert freshsolver.compile_settings == solverkernel.compile_settings, "BatchSolverConfig mismatch"
    assert freshsolver.single_integrator.config == solverkernel.single_integrator.config, "IntegratorRunSettings mismatch"
    assert freshsolver.single_integrator._output_functions.compile_settings == \
           solverkernel.single_integrator._output_functions.compile_settings, "OutputFunctions mismatch"
    assert freshsolver.single_integrator._system.compile_settings == \
           solverkernel.single_integrator._system.compile_settings, "SystemCompileSettings mismatch"
    assert BatchOutputSizes.from_solver(freshsolver) == BatchOutputSizes.from_solver(solverkernel), \
        "BatchOutputSizes mismatch"

def test_bogus_update_fails(solverkernel):
    solverkernel.update(dt_min=0.0001)
    with pytest.raises(KeyError):
        solverkernel.update(obviously_bogus_key="this should not work")
