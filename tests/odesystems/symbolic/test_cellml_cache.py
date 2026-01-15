"""Unit tests for CellML caching functionality."""


import pytest
from pathlib import Path
import pickle
from numpy import float64

from cubie.odesystems.symbolic.parsing.cellml_cache import CellMLCache




@pytest.fixture
def cellml_fixtures_dir():
    """Return path to cellml test fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "cellml"


@pytest.fixture
def basic_cellml_path(cellml_fixtures_dir):
    """Return path to basic_ode.cellml fixture."""
    return str(cellml_fixtures_dir / "basic_ode.cellml")


@pytest.fixture
def tmp_cellml_file(tmp_path):
    """Create temporary CellML file for testing."""
    cellml_content = """<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/1.0#">
    <component name="test">
        <variable name="x" initial_value="1.0"/>
    </component>
</model>
"""
    tmp_file = tmp_path / "test.cellml"
    tmp_file.write_text(cellml_content)
    return str(tmp_file)


def test_cache_initialization_valid_inputs(basic_cellml_path):
    """Verify CellMLCache initializes with valid model_name and cellml_path.
    
    Tests that the cache object is created successfully with proper
    attributes when given valid string inputs for both parameters.
    """
    cache = CellMLCache(model_name="basic_ode", cellml_path=basic_cellml_path)
    
    # Verify attributes are set correctly
    assert cache.model_name == "basic_ode"
    assert cache.cellml_path == basic_cellml_path
    assert cache.cache_file.name == "cellml_cache.pkl"
    assert "basic_ode" in str(cache.cache_dir)


def test_cache_initialization_invalid_inputs(basic_cellml_path):
    """Verify TypeError raised for non-string inputs, ValueError for empty
    model_name.
    
    Tests input validation for the __init__ method, ensuring proper
    exceptions are raised for invalid input types and values.
    """
    # Test non-string model_name
    with pytest.raises(TypeError, match="model_name must be str"):
        CellMLCache(model_name=123, cellml_path=basic_cellml_path)
    
    # Test non-string cellml_path
    with pytest.raises(TypeError, match="cellml_path must be str"):
        CellMLCache(model_name="test", cellml_path=Path(basic_cellml_path))
    
    # Test empty model_name
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        CellMLCache(model_name="", cellml_path=basic_cellml_path)
    
    # Test non-existent cellml_path
    with pytest.raises(FileNotFoundError, match="CellML file not found"):
        CellMLCache(model_name="test", cellml_path="/nonexistent/file.cellml")


def test_get_cellml_hash_consistent(tmp_cellml_file):
    """Verify hash is consistent for same file, changes when file changes.
    
    Tests that the SHA256 hash computation is deterministic for unchanged
    files and properly detects file modifications.
    """
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    
    # Get hash twice - should be identical
    hash1 = cache.get_cellml_hash()
    hash2 = cache.get_cellml_hash()
    assert hash1 == hash2
    
    # Verify hash is 64 characters (SHA256 in hex)
    assert len(hash1) == 64
    
    # Modify file and verify hash changes
    with open(tmp_cellml_file, 'a') as f:
        f.write("\n<!-- Modified -->\n")
    
    hash3 = cache.get_cellml_hash()
    assert hash3 != hash1


def test_cache_valid_missing_file(basic_cellml_path, tmp_path, monkeypatch):
    """Verify cache_valid() returns False when cache file doesn't exist.
    
    Tests that cache validation correctly identifies when no cache file
    is present in the expected location.
    """
    # Change to tmp directory so cache files don't interfere with real ones
    monkeypatch.chdir(tmp_path)
    
    cache = CellMLCache(model_name="basic_ode", cellml_path=basic_cellml_path)
    
    # Cache file should not exist yet
    assert not cache.cache_file.exists()
    assert cache.cache_valid() is False


def test_cache_valid_hash_mismatch(tmp_cellml_file, tmp_path, monkeypatch):
    """Verify cache_valid() returns False when hash doesn't match.
    
    Tests cache invalidation when the source CellML file has been
    modified after cache creation.
    """
    # Change to tmp directory
    monkeypatch.chdir(tmp_path)
    
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    
    # Create cache directory and file with wrong hash
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache.cache_file, 'w') as f:
        f.write("#wronghash123\n")
        f.write("fake cached data\n")
    
    # Cache should be invalid due to hash mismatch
    assert cache.cache_valid() is False


def test_load_from_cache_returns_none_invalid(
    tmp_cellml_file, tmp_path, monkeypatch
):
    """Verify load_from_cache() returns None when cache invalid.
    
    Tests that attempting to load from an invalid or non-existent cache
    returns None rather than raising an exception.
    """
    # Change to tmp directory
    monkeypatch.chdir(tmp_path)
    
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    
    # No cache file exists
    assert cache.load_from_cache() is None
    
    # Create cache file with wrong hash
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache.cache_file, 'w') as f:
        f.write("#wronghash\n")
    
    # Should still return None
    assert cache.load_from_cache() is None


def test_save_and_load_roundtrip(tmp_cellml_file, tmp_path, monkeypatch):
    """Verify data saved to cache can be loaded back correctly.
    
    Tests the complete save/load cycle, ensuring that all data saved to
    the cache can be successfully retrieved with the same values.
    """
    # Import required types for creating mock data
    from attrs import define, field
    from sympy import symbols
    
    # Change to tmp directory
    monkeypatch.chdir(tmp_path)
    
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    
    # Create minimal mock objects for testing
    # (In real use, these come from parse_input)
    x = symbols('x')
    
    # Create mock ParsedEquations with minimal required attributes
    # Using a simple dict to represent it for testing
    mock_equations = {"states": [x], "derivatives": [x]}
    
    # Create mock IndexedBases (simple dict representation)
    mock_indices = {"states": {"x": 0}}
    
    # Create test data
    all_symbols = {"x": x}
    user_functions = None
    fn_hash = "test_hash_12345"
    precision = float64
    name = "test"
    
    # Save to cache
    cache.save_to_cache(
        parsed_equations=mock_equations,
        indexed_bases=mock_indices,
        all_symbols=all_symbols,
        user_functions=user_functions,
        fn_hash=fn_hash,
        precision=precision,
        name=name,
    )
    
    # Verify cache file was created
    assert cache.cache_file.exists()
    
    # Verify cache is valid
    assert cache.cache_valid() is True
    
    # Load from cache
    loaded_data = cache.load_from_cache()
    
    # Verify data was loaded
    assert loaded_data is not None
    
    # Verify all expected keys are present
    assert 'cellml_hash' in loaded_data
    assert 'parsed_equations' in loaded_data
    assert 'indexed_bases' in loaded_data
    assert 'all_symbols' in loaded_data
    assert 'user_functions' in loaded_data
    assert 'fn_hash' in loaded_data
    assert 'precision' in loaded_data
    assert 'name' in loaded_data
    
    # Verify data matches what was saved
    assert loaded_data['fn_hash'] == fn_hash
    assert loaded_data['name'] == name
    assert loaded_data['precision'] == precision
    assert loaded_data['user_functions'] is None


def test_corrupted_cache_returns_none(tmp_cellml_file, tmp_path, monkeypatch):
    """Verify load_from_cache() handles corrupted pickle gracefully.
    
    Tests that corrupted cache files are handled gracefully by returning
    None rather than raising exceptions that would crash the parsing.
    """
    # Change to tmp directory
    monkeypatch.chdir(tmp_path)
    
    cache = CellMLCache(model_name="test", cellml_path=tmp_cellml_file)
    
    # Create cache directory
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get actual hash
    actual_hash = cache.get_cellml_hash()
    
    # Write corrupted cache file (correct hash but invalid pickle)
    with open(cache.cache_file, 'wb') as f:
        f.write(f"#{actual_hash}\n".encode('utf-8'))
        f.write(b"corrupted pickle data that cannot be unpickled")
    
    # Cache should be valid (hash matches) but load should return None
    assert cache.cache_valid() is True
    assert cache.load_from_cache() is None
    
    # Test cache file missing required keys
    with open(cache.cache_file, 'wb') as f:
        f.write(f"#{actual_hash}\n".encode('utf-8'))
        # Pickle a dict with missing keys
        incomplete_data = {'some_key': 'some_value'}
        pickle.dump(incomplete_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Should return None due to missing required keys
    assert cache.load_from_cache() is None
