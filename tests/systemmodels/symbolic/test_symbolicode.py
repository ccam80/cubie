import pytest

from cubie.systemmodels.symbolic.symbolicODE import SymbolicODE

@pytest.fixture(scope="module")
def symbolic_ode():
    return SymbolicODE()