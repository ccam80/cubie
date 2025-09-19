import inspect
import numpy as np
import pytest
from pathlib import Path
from pytest import MonkeyPatch

from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.solver import Solver
from cubie.memory import default_memmgr
from cubie.outputhandling.output_functions import OutputFunctions
from tests.system_fixtures import (
    build_large_nonlinear_system,
    build_three_chamber_system,
    build_three_state_linear_system,
    build_three_state_nonlinear_system,
    build_three_state_very_stiff_system,
)


@pytest.fixture(scope="function", autouse=True)
def codegen_dir():
    """Redirect code generation to a temporary directory for the whole session.

    Use tempfile.mkdtemp instead of pytest's tmp path so the directory isn't
    removed automatically between parameterized test cases. Remove the
    directory at session teardown.
    """
    import tempfile
    import shutil
    from cubie.odesystems.symbolic import odefile

    gen_dir = Path(tempfile.mkdtemp(prefix="cubie_generated_"))
    mp = MonkeyPatch()
    mp.setattr(odefile, "GENERATED_DIR", gen_dir, raising=True)
    try:
        yield gen_dir
    finally:
        # restore original attribute and remove temporary dir
        mp.undo()
        shutil.rmtree(gen_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def precision_override(request):
    return request.param if hasattr(request, "param") else None


@pytest.fixture(scope="function")
def precision(precision_override, system_override):
    """
    Run tests with float32 by default, upgrade to float64 for stiff problems.

    Usage:
    @pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
    def test_something(precision):
        # precision will be np.float64 here
    """
    if precision_override is not None:
        return precision_override
    if system_override == "stiff":
        return np.float64
    return np.float32


@pytest.fixture(scope="function")
def controller_tolerances(precision: np.dtype):
    """Return default tolerances and gain limits for controller tests."""

    atol = np.asarray([precision(1e-6)], dtype=precision)
    rtol = np.asarray([precision(1e-6)], dtype=precision)
    return {
        "atol": atol,
        "rtol": rtol,
        "order": 2,
        "safety": float(precision(0.9)),
        "min_gain": float(precision(0.2)),
        "max_gain": float(precision(5.0)),
    }


@pytest.fixture(scope="function")
def linear_symbolic_system(precision: np.dtype):
    """Three-state linear symbolic system for tests."""

    return build_three_state_linear_system(precision)


@pytest.fixture(scope="function")
def nonlinear_symbolic_system(precision: np.dtype):
    """Three-state nonlinear symbolic system for tests."""

    return build_three_state_nonlinear_system(precision)


@pytest.fixture(scope="function")
def three_chamber_symbolic_system(precision: np.dtype):
    """Symbolic three chamber cardiovascular system."""

    return build_three_chamber_system(precision)


@pytest.fixture(scope="function")
def stiff_symbolic_system(precision: np.dtype):
    """Very stiff nonlinear symbolic system for implicit solver stress tests."""
    return build_three_state_very_stiff_system(precision)


@pytest.fixture(scope="function")
def large_symbolic_system(precision: np.dtype):
    """Large nonlinear symbolic system with 100 states."""

    return build_large_nonlinear_system(precision)



@pytest.fixture(scope="function")
def system_override(request):
    """Override for system model type, if provided."""
    if hasattr(request, 'param'):
        if request.param == {}:
            request.param = 'linear'
    return request.param


@pytest.fixture(scope="function")
def system(request, system_override, precision):
    """
    Return the appropriate symbolic system, defaulting to ``linear``.

    Usage:
    @pytest.mark.parametrize("system_override", ["three_chamber"], indirect=True)
    def test_something(system):
        # system will be the cardiovascular symbolic model here
    """
    model_type = system_override

    if model_type == "linear":
        return request.getfixturevalue("linear_symbolic_system")
    if model_type == "nonlinear":
        return request.getfixturevalue("nonlinear_symbolic_system")
    if model_type in ["three_chamber", "threecm"]:
        return request.getfixturevalue("three_chamber_symbolic_system")
    if model_type == "stiff":
        return request.getfixturevalue("stiff_symbolic_system")
    if model_type == "large":
        return request.getfixturevalue("large_symbolic_system")

    raise ValueError(f"Unknown model type: {model_type}")


@pytest.fixture(scope="function")
def output_functions(loop_compile_settings, system):
    # Merge the default config with any overrides

    outputfunctions = OutputFunctions(
        system.sizes.states,
        system.sizes.parameters,
        loop_compile_settings["output_functions"],
        loop_compile_settings["saved_state_indices"],
        loop_compile_settings["saved_observable_indices"],
    )
    return outputfunctions


def update_loop_compile_settings(system, **kwargs):
    """The standard set of compile arguments, some of which aren't used by certain algorithms (like dtmax for a fixed step)."""
    loop_compile_settings_dict = {
        "dt_min": 0.001,
        "dt_max": 0.01,
        "dt_save": 0.01,
        "dt_summarise": 0.1,
        "atol": 1e-6,
        "rtol": 1e-3,
        "saved_state_indices": [0, 1],
        "saved_observable_indices": [0, 1],
        "summarised_state_indices": [0, 1],
        "summarised_observable_indices": [0, 1],
        "output_functions": ["state"],
    }
    loop_compile_settings_dict.update(kwargs)
    for key in ("dt_min", "dt_max", "dt_save", "dt_summarise", "atol", "rtol"):
        loop_compile_settings_dict[key] = float(loop_compile_settings_dict[key])
    return loop_compile_settings_dict


@pytest.fixture(scope="function")
def loop_compile_settings_overrides(request):
    """Parametrize this fixture indirectly to change compile settings, no need to request this fixture directly
    unless you're testing that it worked."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def solver_settings_override(request):
    """Override for solver settings, if provided."""
    return request.param if hasattr(request, "param") else {}


_SOLVER_SIGNATURE = inspect.signature(Solver.__init__)


def _solver_default(name: str):
    """Return the default value for ``name`` from :class:`Solver`."""

    return _SOLVER_SIGNATURE.parameters[name].default


@pytest.fixture(scope="function")
def solver_settings(
    loop_compile_settings, solver_settings_override, precision
):
    """Create LoopStepConfig from loop_compile_settings."""
    defaults = {
        "algorithm": "euler",
        "duration": 1.0,
        "warmup": 0.0,
        "dt_min": float(loop_compile_settings["dt_min"]),
        "dt_max": float(loop_compile_settings["dt_max"]),
        "dt_save": float(loop_compile_settings["dt_save"]),
        "dt_summarise": float(loop_compile_settings["dt_summarise"]),
        "atol": float(loop_compile_settings["atol"]),
        "rtol": float(loop_compile_settings["rtol"]),
        "saved_state_indices": loop_compile_settings["saved_state_indices"],
        "saved_observable_indices": loop_compile_settings[
            "saved_observable_indices"
        ],
        "output_types": loop_compile_settings["output_functions"],
        "precision": precision,
        "blocksize": 32,
        "stream": 0,
        "profileCUDA": False,
        "memory_manager": default_memmgr,
        "stream_group": "test_group",
        "mem_proportion": None,
    }

    if solver_settings_override:
        # Update defaults with any overrides provided
        for key, value in solver_settings_override.items():
            if key in defaults:
                if key in {
                    "dt_min",
                    "dt_max",
                    "dt_save",
                    "dt_summarise",
                    "atol",
                    "rtol",
                    "duration",
                    "warmup",
                }:
                    defaults[key] = float(value)
                else:
                    defaults[key] = value
    for key in (
        "dt_min",
        "dt_max",
        "dt_save",
        "dt_summarise",
        "atol",
        "rtol",
        "duration",
        "warmup",
    ):
        if defaults[key] is not None:
            defaults[key] = float(defaults[key])
    return defaults


@pytest.fixture(scope="function")
def implicit_step_settings_override(request):
    """Override values for implicit solver helper settings."""

    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def implicit_step_settings(implicit_step_settings_override):
    """Default tolerances and iteration limits for implicit solves."""

    defaults = {
        "atol": _solver_default("atol"),
        "rtol": _solver_default("rtol"),
        "linear_tolerance": 1e-3,
        "nonlinear_tolerance": 1e-3,
        "max_linear_iters": 100,
        "max_newton_iters": 100,
        "newton_damping": 0.5,
        "newton_max_backtracks": 10,
        "reuse_factorisation": True,
        "reuse_jacobian": True,
    }
    defaults.update(implicit_step_settings_override)
    return defaults


@pytest.fixture(scope="function")
def solverkernel(solver_settings, system):
    return BatchSolverKernel(
        system,
        algorithm=solver_settings["algorithm"],
        duration=solver_settings["duration"],
        warmup=solver_settings["warmup"],
        dt_min=solver_settings["dt_min"],
        dt_max=solver_settings["dt_max"],
        dt_save=solver_settings["dt_save"],
        dt_summarise=solver_settings["dt_summarise"],
        atol=solver_settings["atol"],
        rtol=solver_settings["rtol"],
        saved_state_indices=solver_settings["saved_state_indices"],
        saved_observable_indices=solver_settings["saved_observable_indices"],
        output_types=solver_settings["output_types"],
        profileCUDA=solver_settings.get("profileCUDA", False),
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        mem_proportion=solver_settings["mem_proportion"],
    )


@pytest.fixture(scope="function")
def solver(system, solver_settings):
    return Solver(
        system,
        algorithm=solver_settings["algorithm"],
        duration=solver_settings["duration"],
        warmup=solver_settings["warmup"],
        dt_min=solver_settings["dt_min"],
        dt_max=solver_settings["dt_max"],
        dt_save=solver_settings["dt_save"],
        dt_summarise=solver_settings["dt_summarise"],
        atol=solver_settings["atol"],
        rtol=solver_settings["rtol"],
        saved_states=solver_settings["saved_state_indices"],
        saved_observables=solver_settings["saved_observable_indices"],
        output_types=solver_settings["output_types"],
        precision=solver_settings["precision"],
        profileCUDA=solver_settings.get("profileCUDA", False),
        memory_manager=solver_settings["memory_manager"],
        stream_group=solver_settings["stream_group"],
        mem_proportion=solver_settings["mem_proportion"],
    )


@pytest.fixture(scope="function")
def loop_compile_settings(request, system, loop_compile_settings_overrides):
    """
    Create a dictionary of compile settings for the loop function.
    This is the fixture your test should use - if you want to change the compile settings, indirectly parametrize the
    compile_settings_overrides fixture.
    """
    return update_loop_compile_settings(
        system, **loop_compile_settings_overrides
    )
