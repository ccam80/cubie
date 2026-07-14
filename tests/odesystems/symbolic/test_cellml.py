import os
import pytest
import numpy as np

from cubie import solve_ivp, SolveResult
from cubie.odesystems.symbolic.parsing.cellml import (
    load_cellml_model,
    _sanitize_symbol_name,
)
from cubie._utils import is_devfunc


def test_load_simple_cellml_model(basic_model):
    """Load a simple CellML model successfully."""
    assert basic_model.num_states == 1
    assert is_devfunc(basic_model.evaluate_f)


def test_load_complex_cellml_model(beeler_reuter_model):
    """Load Beeler-Reuter cardiac model successfully."""
    assert beeler_reuter_model.num_states == 8
    assert is_devfunc(beeler_reuter_model.evaluate_f)


def test_algebraic_equations_as_observables(cellml_fixtures_dir):
    """Verify algebraic equations can be assigned as observables."""
    observable_names = [
        "sodium_current_i_Na",
        "sodium_current_m_gate_alpha_m",
    ]
    br_path = cellml_fixtures_dir / "beeler_reuter_model_1977.cellml"
    model = load_cellml_model(
        str(br_path),
        observables=observable_names,
        fix_singularities=False,
    )

    # Keys are symbols, so we compare names
    obs_map = model.indices.observables.index_map
    assert len(obs_map) == 2
    obs_symbol_names = [str(k) for k in obs_map.keys()]
    for obs_name in observable_names:
        assert obs_name in obs_symbol_names


def test_invalid_path_type():
    """Verify TypeError raised for non-string path."""
    with pytest.raises(TypeError, match="path must be a string"):
        load_cellml_model(123)


def test_nonexistent_file():
    """Verify FileNotFoundError raised for missing file."""
    with pytest.raises(FileNotFoundError, match="CellML file not found"):
        load_cellml_model("/nonexistent/path/model.cellml")


def test_invalid_extension():
    """Verify ValueError raised for non-.cellml extension."""
    import tempfile

    # Create a temporary file with wrong extension
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="must have .cellml extension"):
            load_cellml_model(temp_path)
    finally:
        os.unlink(temp_path)


def test_custom_precision(cellml_fixtures_dir):
    """Verify custom precision can be specified."""
    model = load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        precision=np.float64,
        fix_singularities=False,
    )
    assert model.precision == np.float64


def test_custom_name(cellml_fixtures_dir):
    """Verify custom name can be specified."""
    model = load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        name="custom_model",
        fix_singularities=False,
    )
    assert model.name == "custom_model"


def test_integration_with_solve_ivp(basic_model):
    """Test that loaded model builds and is ready for solve_ivp."""

    # Verify the model has the necessary components
    assert is_devfunc(basic_model.evaluate_f)
    assert basic_model.num_states == 1
    # Verify initial values are accessible
    assert basic_model.indices.states.defaults is not None
    results = solve_ivp(basic_model, [1.0])
    assert isinstance(results, SolveResult)


def test_initial_values_from_cellml(beeler_reuter_model):
    """Verify initial values from CellML model are preserved."""
    # Check that initial values were set using defaults dict
    assert beeler_reuter_model.indices.states.defaults is not None
    assert len(beeler_reuter_model.indices.states.defaults) == 8

    # Initial values should be non-zero (from the model)
    assert any(
        v != 0 for v in beeler_reuter_model.indices.states.defaults.values()
    )


def test_units_extracted_from_cellml(basic_model):
    """Verify units are extracted from CellML model."""
    # Check that units are available
    assert hasattr(basic_model, "state_units")
    assert hasattr(basic_model, "parameter_units")
    assert hasattr(basic_model, "observable_units")

    # Basic model should have dimensionless units
    assert "main_x" in basic_model.state_units
    assert basic_model.state_units["main_x"] == "dimensionless"


def test_default_units_for_symbolic_ode():
    """Verify SymbolicODE defaults to dimensionless units."""
    from cubie import SymbolicODE
    import numpy as np

    ode = SymbolicODE.create(
        dxdt="dx = -a * x",
        states={"x": 1.0},
        parameters={"a": 0.5},
        precision=np.float32,
    )

    assert ode.state_units == {"x": "dimensionless"}
    assert ode.parameter_units == {"a": "dimensionless"}
    assert ode.observable_units == {}


def test_cellml_uses_sympy_pathway(basic_model):
    """Verify CellML adapter uses SymPy pathway internally."""
    assert basic_model.num_states == 1
    assert is_devfunc(basic_model.evaluate_f)

    initial_vals = basic_model.indices.states.default_values
    assert len(initial_vals) > 0


def test_cellml_timing_events_updated():
    """Verify timing events use new SymPy preparation name."""
    from cubie.time_logger import default_timelogger

    registered_events = default_timelogger._event_registry
    assert "codegen_cellml_sympy_preparation" in registered_events

    assert "codegen_cellml_string_formatting" not in registered_events


def test_custom_units_for_symbolic_ode():
    """Verify custom units can be specified for SymbolicODE."""
    from cubie import SymbolicODE
    import numpy as np

    ode = SymbolicODE.create(
        dxdt=["dx = -a * x", "y = 2 * x"],
        states={"x": 1.0},
        parameters={"a": 0.5},
        observables=["y"],
        state_units={"x": "meters"},
        parameter_units={"a": "per_second"},
        observable_units={"y": "meters"},
        precision=np.float32,
    )

    assert ode.state_units == {"x": "meters"}
    assert ode.parameter_units == {"a": "per_second"}
    assert ode.observable_units == {"y": "meters"}


def test_numeric_assignments_become_constants(basic_model):
    """Verify variables with numeric assignments become constants by default."""
    # Variable 'a' has numeric value 0.5 in the CellML model
    # It should become a constant
    constants_map = basic_model.indices.constants.index_map
    assert len(constants_map) > 0

    # Check that 'main_a' is in constants (name is sanitized)
    constant_names = [str(k) for k in constants_map.keys()]
    assert "main_a" in constant_names

    # Check that the default value is correct
    constants_defaults = basic_model.indices.constants.defaults
    assert constants_defaults is not None
    assert "main_a" in constants_defaults
    assert constants_defaults["main_a"] == 0.5


def test_numeric_assignments_as_parameters(cellml_fixtures_dir):
    """Verify variables with numeric assignments become parameters if specified."""
    model = load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        name="basic_ode",
        parameters=["main_a"],
        fix_singularities=False,
    )

    # 'main_a' should now be a parameter instead of a constant
    parameters_map = model.indices.parameters.index_map
    parameter_names = [str(k) for k in parameters_map.keys()]
    assert "main_a" in parameter_names

    # Check that the default value is correct
    parameters_defaults = model.indices.parameters.defaults
    assert parameters_defaults is not None
    assert "main_a" in parameters_defaults
    assert parameters_defaults["main_a"] == 0.5

    # Should not be in constants
    constants_map = model.indices.constants.index_map
    constant_names = [str(k) for k in constants_map.keys()]
    assert "main_a" not in constant_names


def test_parameters_dict_preserves_numeric_values(cellml_fixtures_dir):
    """Verify numeric values are preserved when parameters is a dict."""
    # User can provide parameters as dict with custom default values
    # But if the CellML has a numeric value, it should be preserved
    model = load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        name="basic_ode",
        parameters={"main_a": 1.0},
        fix_singularities=False,
    )

    # The user-provided value doesnt take precedence - users can override
    # these per run.
    parameters_defaults = model.indices.parameters.defaults
    assert parameters_defaults is not None
    assert "main_a" in parameters_defaults
    assert parameters_defaults["main_a"] == 0.5


def test_non_numeric_algebraic_equations_remain(beeler_reuter_model):
    # The Beeler-Reuter model has complex algebraic equations
    # These should remain as equations, not become constants
    # We can check by ensuring there are equations beyond just the differential ones

    # Model has 8 state variables, so 8 differential equations
    # Check that we have state derivatives
    state_derivatives = beeler_reuter_model.equations.state_derivatives
    assert len(state_derivatives) == 8

    # Check that we have some observables or auxiliaries
    # (algebraic equations that aren't simple numeric assignments)
    observables = beeler_reuter_model.equations.observables
    auxiliaries = beeler_reuter_model.equations.auxiliaries

    # Total algebraic equations should be > 0
    algebraic_eq_count = len(observables) + len(auxiliaries)
    assert algebraic_eq_count > 0


def test_cellml_time_logging_events_registered():
    """Verify time logging events are registered for cellml import."""
    from cubie.time_logger import default_timelogger

    # Check that all cellml events are registered
    expected_events = [
        "codegen_cellml_load_model",
        "codegen_cellml_symbol_conversion",
        "codegen_cellml_equation_processing",
        "codegen_cellml_sympy_preparation",
    ]

    for event_name in expected_events:
        assert event_name in default_timelogger._event_registry
        assert (
            default_timelogger._event_registry[event_name]["category"]
            == "codegen"
        )


def test_cache_used_on_reload(cellml_fixtures_dir, tmp_path):
    """Verify CellML cache is used on second load of same model."""
    import shutil

    # Copy fixture to tmp directory so we can control generated/ location
    tmp_cellml = tmp_path / "basic_ode.cellml"
    shutil.copy(cellml_fixtures_dir / "basic_ode.cellml", tmp_cellml)

    # Change working directory to tmp_path for generated/ directory
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # First load - creates cache
        ode1 = load_cellml_model(
            str(tmp_cellml), name="basic_ode", fix_singularities=False
        )

        # Verify cache manifest created (LRU cache uses manifest file)
        manifest_file = (
            tmp_path / "generated" / "basic_ode" / "cellml_cache_manifest.json"
        )
        assert manifest_file.exists(), (
            "Cache manifest should exist after first load"
        )

        # Second load - should use cache
        ode2 = load_cellml_model(
            str(tmp_cellml), name="basic_ode", fix_singularities=False
        )

        # Verify both ODEs are equivalent
        assert ode1.num_states == ode2.num_states
        assert ode1.fn_hash == ode2.fn_hash
        assert len(ode1.indices.states.index_map) == len(
            ode2.indices.states.index_map
        )

    finally:
        os.chdir(original_cwd)


def test_cache_invalidated_on_file_change(cellml_fixtures_dir, tmp_path):
    """Verify cache invalidates when CellML file content changes."""
    import shutil

    # Copy fixture to tmp directory
    tmp_cellml = tmp_path / "basic_ode.cellml"
    shutil.copy(cellml_fixtures_dir / "basic_ode.cellml", tmp_cellml)

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # First load - creates cache
        load_cellml_model(
            str(tmp_cellml), name="basic_ode", fix_singularities=False
        )
        manifest_file = (
            tmp_path / "generated" / "basic_ode" / "cellml_cache_manifest.json"
        )
        assert manifest_file.exists()

        # Modify CellML file (add comment)
        with open(tmp_cellml, "a") as f:
            f.write("\n<!-- Modified for test -->\n")

        # Verify cache becomes invalid (file hash changed)
        from cubie.odesystems.symbolic.parsing.cellml_cache import CellMLCache
        import numpy as np

        cache = CellMLCache("basic_ode", str(tmp_cellml))
        # Compute args_hash for default arguments (precision=np.float32)
        args_hash = cache.compute_cache_key(
            None, None, np.float32, "basic_ode", fix_singularities=False
        )
        assert not cache.cache_valid(args_hash), (
            "Cache should be invalid after file change"
        )

        # Load again - should re-parse and update cache
        load_cellml_model(
            str(tmp_cellml), name="basic_ode", fix_singularities=False
        )

        # Verify new cache is valid (need fresh CellMLCache for updated file hash)
        cache2 = CellMLCache("basic_ode", str(tmp_cellml))
        args_hash2 = cache2.compute_cache_key(
            None, None, np.float32, "basic_ode", fix_singularities=False
        )
        assert cache2.cache_valid(args_hash2), (
            "Cache should be valid after re-parse"
        )

    finally:
        os.chdir(original_cwd)


def test_cache_isolated_per_model(cellml_fixtures_dir, tmp_path):
    """Verify each model has separate cache file."""
    import shutil

    # Copy both fixtures to tmp directory
    tmp_basic = tmp_path / "basic_ode.cellml"
    tmp_br = tmp_path / "beeler_reuter_model_1977.cellml"
    shutil.copy(cellml_fixtures_dir / "basic_ode.cellml", tmp_basic)
    shutil.copy(
        cellml_fixtures_dir / "beeler_reuter_model_1977.cellml", tmp_br
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Load both models
        ode_basic = load_cellml_model(
            str(tmp_basic), name="basic_ode", fix_singularities=False
        )
        ode_br = load_cellml_model(
            str(tmp_br), name="beeler_reuter_model_1977"
        )

        # Verify separate cache manifests exist (LRU cache uses manifest files)
        manifest_basic = (
            tmp_path / "generated" / "basic_ode" / "cellml_cache_manifest.json"
        )
        manifest_br = (
            tmp_path
            / "generated"
            / "beeler_reuter_model_1977"
            / "cellml_cache_manifest.json"
        )

        assert manifest_basic.exists(), "basic_ode cache manifest should exist"
        assert manifest_br.exists(), (
            "beeler_reuter cache manifest should exist"
        )

        # Verify different models have different hashes
        assert ode_basic.fn_hash != ode_br.fn_hash
        assert ode_basic.num_states != ode_br.num_states

    finally:
        os.chdir(original_cwd)


def test_sanitize_symbol_name_leading_digit():
    """A name starting with a digit is prefixed to stay a valid identifier."""
    assert _sanitize_symbol_name("3rate") == "var_3rate"


def test_sanitize_symbol_name_leading_underscore_digit():
    """A leading underscore followed by a digit is prefixed with 'var'."""
    assert _sanitize_symbol_name("_2x") == "var_2x"


def test_load_with_parameters_dict(
    cellml_fixtures_dir, tmp_path, monkeypatch
):
    """A parameters dict is accepted and merged with CellML values."""
    monkeypatch.chdir(tmp_path)
    path = str(cellml_fixtures_dir / "basic_ode.cellml")
    model = load_cellml_model(
        path,
        parameters={"user_param": 1.5},
        fix_singularities=False,
    )
    assert "user_param" in model.parameters.values_dict
    assert model.parameters.values_dict["user_param"] == 1.5


def test_underscore_component_names_load(
    cellml_fixtures_dir, tmp_path, monkeypatch
):
    """Variables qualified by a leading-underscore component load."""
    monkeypatch.chdir(tmp_path)
    model = load_cellml_model(
        str(cellml_fixtures_dir / "underscore_names.cellml"),
        fix_singularities=False,
    )
    assert model.num_states == 1
    state_names = [str(s) for s in model.indices.states.index_map]
    assert state_names == ["_main_x"]


def test_multiple_time_variables_raise(cellml_fixtures_dir):
    """Derivatives against two time variables raise a clear error."""
    with pytest.raises(ValueError, match="single shared time"):
        load_cellml_model(
            str(cellml_fixtures_dir / "two_time_variables.cellml"),
            fix_singularities=False,
        )


def test_constant_as_observable_raises(cellml_fixtures_dir):
    """Requesting a numeric-valued variable as an observable raises."""
    with pytest.raises(ValueError, match="no defining equation"):
        load_cellml_model(
            str(cellml_fixtures_dir / "basic_ode.cellml"),
            observables=["main_a"],
            fix_singularities=False,
        )


def test_repeat_load_hits_persistent_cache(
    cellml_fixtures_dir, tmp_path, monkeypatch
):
    """A second identical load returns the cached parsed model."""
    monkeypatch.chdir(tmp_path)
    path = str(cellml_fixtures_dir / "basic_ode.cellml")
    first = load_cellml_model(
        path, precision=np.float64, fix_singularities=False
    )
    second = load_cellml_model(
        path, precision=np.float64, fix_singularities=False
    )
    assert second.fn_hash == first.fn_hash
    assert second.num_states == first.num_states


def test_unknown_parameter_name_reuses_effective_cache(
    cellml_fixtures_dir, tmp_path, monkeypatch
):
    """Parameter names absent from the model resolve to the plain config.

    The pre-parse cache key is built from the requested parameter
    names, while the post-parse key uses the parameters the model
    actually yielded. A name that matches nothing therefore misses the
    early check but lands on the cached plain-configuration entry
    after parsing.
    """
    monkeypatch.chdir(tmp_path)
    path = str(cellml_fixtures_dir / "basic_ode.cellml")
    baseline = load_cellml_model(path, fix_singularities=False)
    aliased = load_cellml_model(
        path,
        parameters=["not_in_model"],
        fix_singularities=False,
    )
    assert aliased.fn_hash == baseline.fn_hash
    assert "not_in_model" not in aliased.parameters.values_dict


def test_parameters_as_list(cellml_fixtures_dir, tmp_path, monkeypatch):
    """A parameters list promotes named constants to parameters."""
    monkeypatch.chdir(tmp_path)
    model = load_cellml_model(
        str(cellml_fixtures_dir / "basic_ode.cellml"),
        parameters=["main_a"],
        fix_singularities=False,
    )
    assert "main_a" in model.parameters.values_dict
    assert model.parameters.values_dict["main_a"] == 0.5
