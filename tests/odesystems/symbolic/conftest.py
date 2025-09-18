import pytest
import sympy as sp

from cubie.odesystems.symbolic.indexedbasemaps import (
    IndexedBaseMap,
    IndexedBases,
)
from cubie.odesystems.symbolic.symbolicODE import SymbolicODE


@pytest.fixture(scope="function")
def simple_system_defaults():
    states = {"one": 0.9, "foo": 0.5}
    parameters = {"zebra": 0.2, "fox": 0.4}
    constants = {"apple": 0.43, "linen": 0.32}
    drivers = ["driver1"]
    observables = ["safari", "zoo"]

    # Out of order, aux-refs-aux, dx-refs-dx
    dxdt_str = """safari = linen * fox + apple * zebra + zoo
    zoo = linen ** 2 * fox + apple * one**2 - zebra * foo + driver1
    uninited = zoo**2 + safari**2
    done = uninited*2 + dfoo
    dfoo = zoo + safari"""
    dxdt_list = [
        "safari = linen * fox + apple * zebra + zoo",
        "zoo = linen ** 2 * fox + apple * one**2 - zebra * foo + driver1",
        "uninited = zoo**2 + safari**2",
        "done = uninited*2 + dfoo",
        "dfoo = zoo + safari",
    ]

    return (
        states,
        parameters,
        constants,
        drivers,
        observables,
        dxdt_str,
        dxdt_list,
    )


@pytest.fixture(scope="function")
def simple_symbols_dict(simple_system_defaults):
    (
        states,
        parameters,
        constants,
        drivers,
        observables,
        dxdt_str,
        dxdt_list,
    ) = simple_system_defaults
    ib = IndexedBases.from_user_inputs(
        states, parameters, constants, observables, drivers
    )
    symbols = ib.all_symbols
    return symbols


@pytest.fixture
def simple_symbols():
    """Basic SymPy symbols for testing."""
    x, y, z = sp.symbols("x y z", real=True)
    a, b, c = sp.symbols("a b c", real=True)
    return {"states": [x, y], "params": [a, b], "constants": [c], "aux": [z]}


@pytest.fixture
def simple_equations():
    """Simple symbolic equations for testing."""
    x, y = sp.symbols("x y", real=True)
    a, b = sp.symbols("a b", real=True)

    # dx/dt = -a*x + b*y
    # dy/dt = a*x - b*y
    equations = [
        (sp.Symbol("dx", real=True), -a * x + b * y),
        (sp.Symbol("dy", real=True), a * x - b * y),
    ]
    return equations


@pytest.fixture
def complex_equations():
    """More complex equations with auxiliary variables."""
    x, y, z = sp.symbols("x y z", real=True)
    a, b, c = sp.symbols("a b c", real=True)

    # Auxiliary variable
    aux = a * x + b * y

    equations = [
        (sp.Symbol("aux", real=True), aux),
        (sp.Symbol("dx", real=True), -aux + c),
        (sp.Symbol("dy", real=True), aux - c * y),
        (sp.Symbol("dz", real=True), x + y - z),
    ]
    return equations


@pytest.fixture(scope="function")
def observables_kernel_system(precision):
    """Return a ``SymbolicODE`` used for observables parity tests."""

    dxdt_lines = [
        "obs_rate = alpha * x + c0",
        "obs_total = obs_rate + beta * y + drive",
        "dx = obs_total - y + alpha * drive",
        "dy = obs_rate * x + c0",
    ]

    system = SymbolicODE.create(
        dxdt=dxdt_lines,
        states={"x": precision(0.0), "y": precision(0.0)},
        parameters={"alpha": precision(0.0), "beta": precision(0.0)},
        constants={"c0": precision(1.1)},
        drivers={"drive": precision(0.0)},
        observables=["obs_rate", "obs_total"],
        precision=precision,
        strict=True,
        name="observables_kernel_system",
    )

    return system


@pytest.fixture
def indexed_base_map():
    """Sample IndexedBaseMap for testing."""
    symbols = ["x", "y", "z"]
    defaults = [1.0, 2.0, 3.0]
    return IndexedBaseMap("test_base", symbols, defaults)


@pytest.fixture
def indexed_bases():
    """Sample IndexedBases for testing."""
    states = ["x", "y"]
    parameters = ["a", "b"]
    constants = ["c"]
    observables = ["obs1"]
    drivers = ["driver1"]

    return IndexedBases.from_user_inputs(
        states=states,
        parameters=parameters,
        constants=constants,
        observables=observables,
        drivers=drivers,
    )


@pytest.fixture
def sample_hash():
    """Sample hash string for testing."""
    return "# hash: test_hash_123456"


@pytest.fixture
def linear_system_equations():
    """Linear system equations for Jacobian testing."""
    x, y = sp.symbols("x y", real=True)
    a, b, c, d = sp.symbols("a b c d", real=True)

    # Linear system: dx/dt = ax + by, dy/dt = cx + dy
    equations = [
        (sp.Symbol("dx", real=True), a * x + b * y),
        (sp.Symbol("dy", real=True), c * x + d * y),
    ]
    return equations


@pytest.fixture
def nonlinear_equations():
    """Nonlinear equations for comprehensive testing."""
    x, y = sp.symbols("x y", real=True)
    a, b = sp.symbols("a b", real=True)

    equations = [
        (sp.Symbol('dx', real=True), a*x - b*x*y),
        (sp.Symbol('dy', real=True), b*x*y - a*y)
    ]
    return equations
