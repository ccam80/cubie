from CuMC.ForwardSim.SolverKernel import SolverKernel
import pytest

@pytest.fixture(scope="function")
def kernel(threecm_model, loop_compile_settings, precision):
    """Fixture to create a SolverKernel instance with a ThreeChamberModel."""
    kernel = SolverKernel(system=threecm_model, precision=precision)
    kernel.build()
    return kernel
