import pytest
from CuMC.SystemModels.Systems.decays import Decays

#Bring in test kernels for systems, integrator loops, integrators. Create these as fixtures, and also expected results generators.
@pytest.fixture(scope="session")
def SystemModelRunner():
    """Kernel to run a system model with a known input and expected output"""
