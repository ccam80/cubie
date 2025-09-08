import numpy as np
import pytest
from pathlib import Path
from pytest import MonkeyPatch

from cubie.batchsolving.BatchSolverKernel import BatchSolverKernel
from cubie.batchsolving.solver import Solver
from cubie.memory import default_memmgr
from cubie.outputhandling.output_functions import OutputFunctions


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
def precision(precision_override):
    """
    Run tests with float32 by default, or override with float64.

    Usage:
    @pytest.mark.parametrize("precision_override", [np.float64], indirect=True)
    def test_something(precision):
        # precision will be np.float64 here
    """
    return (
        precision_override if precision_override == np.float64 else np.float32
    )


@pytest.fixture(scope="function")
def threecm_model(precision):
    from cubie.odesystems.systems.threeCM import ThreeChamberModel

    threeCM = ThreeChamberModel(precision=precision)
    threeCM.build()
    return threeCM


@pytest.fixture(scope="function")
def decays_123_model(precision):
    from cubie.odesystems.systems.decays import Decays

    decays3 = Decays(
        coefficients=[precision(1.0), precision(2.0), precision(3.0)],
        precision=precision,
    )
    decays3.build()
    return decays3


@pytest.fixture(scope="function")
def decays_1_100_model(precision):
    from cubie.odesystems.systems.decays import Decays

    decays100 = Decays(
        coefficients=np.arange(1, 101, dtype=precision), precision=precision
    )
    decays100.build()
    return decays100


@pytest.fixture(scope="function")
def system_override(request):
    """Override for system model type, if provided."""
    return request.param if hasattr(request, "param") else {}


@pytest.fixture(scope="function")
def system(request, system_override, precision):
    """
    Return the appropriate system model, defaulting to Decays123.

    Usage:
    @pytest.mark.parametrize("system_override", ["ThreeChamber"], indirect=True)
    def test_something(system):
        # system will be the ThreeChamber model here
    """
    # Use the override if provided, otherwise default to Decays123
    if system_override == {} or system_override is None:
        model_type = "Decays123"
    else:
        model_type = system_override

    # Initialize the appropriate model fixture based on the parameter
    if model_type == "ThreeChamber":
        model = request.getfixturevalue("threecm_model")
    elif model_type == "Decays123":
        model = request.getfixturevalue("decays_123_model")
    elif model_type == "Decays1_100":
        model = request.getfixturevalue("decays_1_100_model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.build()
    return model


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


@pytest.fixture(scope="function")
def solver_settings(
    loop_compile_settings, solver_settings_override, precision
):
    """Create LoopStepConfig from loop_compile_settings."""
    defaults = {
        "algorithm": "euler",
        "duration": 1.0,
        "warmup": 0.0,
        "dt_min": loop_compile_settings["dt_min"],
        "dt_max": loop_compile_settings["dt_max"],
        "dt_save": loop_compile_settings["dt_save"],
        "dt_summarise": loop_compile_settings["dt_summarise"],
        "atol": loop_compile_settings["atol"],
        "rtol": loop_compile_settings["rtol"],
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
                defaults[key] = value
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
        precision=solver_settings["precision"],
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
