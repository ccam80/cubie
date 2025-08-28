from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp
from numba import cuda, from_dtype
from numpy.testing import assert_allclose

from cubie.systemmodels.symbolic.odefile import (
    DXDT_MATCHLINE,
    HEADER,
    JVP_MATCHLINE,
    VJP_MATCHLINE,
    ODEFile,
)
from cubie.systemmodels.symbolic.parser import IndexedBases
from cubie.systemmodels.symbolic.sym_utils import hash_system_definition


class TestODEFileConstants:
    """Test the constants defined in the odefile module."""

    def test_header_content(self):
        """Test the HEADER constant."""
        assert isinstance(HEADER, str)
        assert "# This file was generated automatically by Cubie" in HEADER
        assert "from numba import cuda" in HEADER
        assert "Don't make changes in here" in HEADER

    def test_matchlines(self):
        """Test the matchline constants."""
        assert isinstance(DXDT_MATCHLINE, str)
        assert isinstance(JVP_MATCHLINE, str)
        assert isinstance(VJP_MATCHLINE, str)
        assert "AUTO-GENERATED" in DXDT_MATCHLINE
        assert "AUTO-GENERATED" in JVP_MATCHLINE
        assert "AUTO-GENERATED" in VJP_MATCHLINE


class TestODEFileInit:
    """Test ODEFile initialization."""

    def test_init_creates_file(self, temp_dir):
        """Test that ODEFile.__init__ creates a file."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            hash_value = "test_hash_123"
            ode_file = ODEFile("test_system", hash_value)

            assert ode_file.file_path.exists()
            assert ode_file.file_path.name == "test_system.py"

            # Check file content
            content = ode_file.file_path.read_text()
            assert hash_value in content
            assert HEADER in content

    def test_init_existing_valid_file(self, temp_dir):
        """Test initialization with existing valid cached file."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            hash_value = "test_hash_456"

            # Create ODEFile first time
            ode_file1 = ODEFile("test_system", hash_value)
            original_mtime = ode_file1.file_path.stat().st_mtime

            # Create ODEFile second time with same hash
            ode_file2 = ODEFile("test_system", hash_value)
            new_mtime = ode_file2.file_path.stat().st_mtime

            # File should not have been recreated
            assert original_mtime == new_mtime

    def test_init_existing_invalid_file(self, temp_dir):
        """Test initialization with existing file with different hash."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            hash1 = "test_hash_old"
            hash2 = "test_hash_new"

            # Create ODEFile with first hash
            ode_file1 = ODEFile("test_system", hash1)

            # Create ODEFile with different hash
            ode_file2 = ODEFile("test_system", hash2)

            # File should contain new hash
            content = ode_file2.file_path.read_text()
            assert hash2 in content
            assert hash1 not in content


class TestODEFileProperties:
    """Test ODEFile property methods."""

    def test_dxdt_generated_false(self, temp_dir):
        """Test _dxdt_generated when no dxdt function exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")
            assert not ode_file._dxdt_generated

    def test_dxdt_generated_true(self, temp_dir):
        """Test _dxdt_generated when dxdt function exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            # Add dxdt function to file
            with open(ode_file.file_path, "a") as f:
                f.write(f"\n{DXDT_MATCHLINE}\n")

            assert ode_file._dxdt_generated

    def test_jvp_generated_false(self, temp_dir):
        """Test _jvp_generated when no JVP function exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")
            assert not ode_file._jvp_generated

    def test_jvp_generated_true(self, temp_dir):
        """Test _jvp_generated when JVP function exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            # Add JVP function to file
            with open(ode_file.file_path, "a") as f:
                f.write(f"\n{JVP_MATCHLINE}\n")

            assert ode_file._jvp_generated

    def test_vjp_generated_false(self, temp_dir):
        """Test _vjp_generated when no VJP function exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")
            assert not ode_file._vjp_generated

    def test_vjp_generated_true(self, temp_dir):
        """Test _vjp_generated when VJP function exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            # Add VJP function to file
            with open(ode_file.file_path, "a") as f:
                f.write(f"\n{VJP_MATCHLINE}\n")

            assert ode_file._vjp_generated


class TestODEFileCacheValidation:
    """Test cache validation methods."""

    def test_cached_file_valid_nonexistent(self, temp_dir):
        """Test cache validation when file doesn't exist."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")
            # Delete the file that was created during init
            ode_file.file_path.unlink()

            assert not ode_file.cached_file_valid("any_hash")

    def test_cached_file_valid_matching_hash(self, temp_dir):
        """Test cache validation with matching hash."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            hash_value = "test_hash_match"
            ode_file = ODEFile("test_system", hash_value)

            assert ode_file.cached_file_valid(hash_value)

    def test_cached_file_valid_different_hash(self, temp_dir):
        """Test cache validation with different hash."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            original_hash = "original_hash"
            different_hash = "different_hash"

            ode_file = ODEFile("test_system", original_hash)

            assert not ode_file.cached_file_valid(different_hash)


class TestODEFileDxdtMethods:
    """Test DXDT-related methods."""

    def test_generate_dxdt_fac_new(
        self, temp_dir, simple_equations, indexed_bases
    ):
        """Test generating dxdt factory when none exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            ode_file.generate_dxdt_fac(simple_equations, indexed_bases)

            # Check that function was added
            content = ode_file.file_path.read_text()
            assert "def dxdt_factory" in content
            assert DXDT_MATCHLINE in content

    def test_generate_dxdt_fac_existing(
        self, temp_dir, simple_equations, indexed_bases
    ):
        """Test generating dxdt factory when one already exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            # Add dxdt function first
            ode_file.generate_dxdt_fac(simple_equations, indexed_bases)
            original_content = ode_file.file_path.read_text()

            # Try to generate again
            ode_file.generate_dxdt_fac(simple_equations, indexed_bases)
            new_content = ode_file.file_path.read_text()

            # Content should be the same (no duplicate generation)
            assert original_content == new_content

    def test_get_dxdt_fac(self, temp_dir, simple_equations, indexed_bases):
        """Test getting dxdt factory function."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            factory_func = ode_file.get_dxdt_fac(
                simple_equations, indexed_bases
            )

            # Should return a callable
            assert callable(factory_func)

            # File should contain the function
            content = ode_file.file_path.read_text()
            assert "def dxdt_factory" in content


class TestODEFileJvpMethods:
    """Test JVP-related methods."""

    def test_generate_jvp_fac_new(
        self, temp_dir, simple_equations, indexed_bases
    ):
        """Test generating JVP factory when none exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            ode_file.generate_jvp_fac(simple_equations, indexed_bases)

            # Check that function was added
            content = ode_file.file_path.read_text()
            assert "def jvp_factory" in content
            assert JVP_MATCHLINE in content

    def test_generate_jvp_fac_existing(
        self, temp_dir, simple_equations, indexed_bases
    ):
        """Test generating JVP factory when one already exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            # Add JVP function first
            ode_file.generate_jvp_fac(simple_equations, indexed_bases)
            original_content = ode_file.file_path.read_text()

            # Try to generate again
            ode_file.generate_jvp_fac(simple_equations, indexed_bases)
            new_content = ode_file.file_path.read_text()

            # Content should be the same
            assert original_content == new_content

    def test_get_jvp_fac(self, temp_dir, simple_equations, indexed_bases):
        """Test getting JVP factory function."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            factory_func = ode_file.get_jvp_fac(
                simple_equations, indexed_bases
            )

            # Should return a callable
            assert callable(factory_func)

            # File should contain the function
            content = ode_file.file_path.read_text()
            assert "def jvp_factory" in content


class TestODEFileVjpMethods:
    """Test VJP-related methods."""

    def test_generate_vjp_fac_new(
        self, temp_dir, simple_equations, indexed_bases
    ):
        """Test generating VJP factory when none exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            ode_file.generate_vjp_fac(simple_equations, indexed_bases)

            # Check that function was added
            content = ode_file.file_path.read_text()
            assert "def vjp_factory" in content
            assert VJP_MATCHLINE in content

    def test_generate_vjp_fac_existing(
        self, temp_dir, simple_equations, indexed_bases
    ):
        """Test generating VJP factory when one already exists."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            # Add VJP function first
            ode_file.generate_vjp_fac(simple_equations, indexed_bases)
            original_content = ode_file.file_path.read_text()

            # Try to generate again
            ode_file.generate_vjp_fac(simple_equations, indexed_bases)
            new_content = ode_file.file_path.read_text()

            # Content should be the same
            assert original_content == new_content

    def test_get_vjp_fac(self, temp_dir, simple_equations, indexed_bases):
        """Test getting VJP factory function."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            factory_func = ode_file.get_vjp_fac(
                simple_equations, indexed_bases
            )

            # Should return a callable
            assert callable(factory_func)

            # File should contain the function
            content = ode_file.file_path.read_text()
            assert "def vjp_factory" in content


class TestODEFileFactoryMethod:
    """Test the generic get_factory method."""

    def test_get_factory_dxdt(self, temp_dir, simple_equations, indexed_bases):
        """Test get_factory for dxdt type."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            factory_func = ode_file.get_factory(
                "dxdt", simple_equations, indexed_bases
            )

            assert callable(factory_func)
            content = ode_file.file_path.read_text()
            assert "def dxdt_factory" in content

    def test_get_factory_jvp(self, temp_dir, simple_equations, indexed_bases):
        """Test get_factory for JVP type."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            factory_func = ode_file.get_factory(
                "jvp", simple_equations, indexed_bases
            )

            assert callable(factory_func)
            content = ode_file.file_path.read_text()
            assert "def jvp_factory" in content

    def test_get_factory_vjp(self, temp_dir, simple_equations, indexed_bases):
        """Test get_factory for VJP type."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            factory_func = ode_file.get_factory(
                "vjp", simple_equations, indexed_bases
            )

            assert callable(factory_func)
            content = ode_file.file_path.read_text()
            assert "def vjp_factory" in content

    def test_get_factory_invalid_type(
        self, temp_dir, simple_equations, indexed_bases
    ):
        """Test get_factory with invalid function type."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            with pytest.raises(ValueError, match="Invalid function type"):
                ode_file.get_factory(
                    "invalid_type", simple_equations, indexed_bases
                )


class TestODEFileUtilityMethods:
    """Test utility methods."""

    def test_add_function(self, temp_dir):
        """Test adding a function to the file."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            test_code = "\ndef test_function():\n    return 42\n"
            ode_file.add_function(test_code, "test_function")

            content = ode_file.file_path.read_text()
            assert "def test_function():" in content
            assert "return 42" in content

    def test_import_function(self, temp_dir):
        """Test importing a function from the generated file."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            # Add a simple test function
            test_code = "\ndef test_function():\n    return 'hello world'\n"
            ode_file.add_function(test_code, "test_function")

            # Import and test the function
            imported_func = ode_file._import_function("test_function")
            assert callable(imported_func)
            assert imported_func() == "hello world"

    def test_import_nonexistent_function(self, temp_dir):
        """Test importing a non-existent function."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            with pytest.raises(AttributeError):
                ode_file._import_function("nonexistent_function")


class TestODEFileEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_equations(self, temp_dir, indexed_bases):
        """Test with empty equations list."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            empty_equations = []
            factory_func = ode_file.get_dxdt_fac(
                empty_equations, indexed_bases
            )

            assert callable(factory_func)

    def test_complex_system_name(self, temp_dir):
        """Test with complex system name."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            complex_name = "system_with_underscores_123"
            ode_file = ODEFile(complex_name, "hash")

            assert ode_file.file_path.name == f"{complex_name}.py"
            assert ode_file.file_path.exists()

    def test_long_hash(self, temp_dir):
        """Test with very long hash string."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            long_hash = "a" * 1000  # Very long hash
            ode_file = ODEFile("test_system", long_hash)

            content = ode_file.file_path.read_text()
            assert long_hash in content

    def test_special_characters_in_equations(self, temp_dir, indexed_bases):
        """Test with equations containing special mathematical functions."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            x = sp.symbols("x")
            special_equations = [
                (
                    sp.Symbol("dx"),
                    sp.sin(x) + sp.cos(x) + sp.exp(x) + sp.log(x + 1),
                )
            ]

            factory_func = ode_file.get_dxdt_fac(
                special_equations, indexed_bases
            )
            assert callable(factory_func)


class TestODEFileIntegration:
    """Integration tests for ODEFile functionality."""

    def test_full_workflow_multiple_functions(
        self, temp_dir, simple_equations, indexed_bases
    ):
        """Test complete workflow with all function types."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("test_system", "hash")

            # Generate all three types of functions
            dxdt_func = ode_file.get_dxdt_fac(simple_equations, indexed_bases)
            jvp_func = ode_file.get_jvp_fac(simple_equations, indexed_bases)
            vjp_func = ode_file.get_vjp_fac(simple_equations, indexed_bases)

            # All should be callable
            assert callable(dxdt_func)
            assert callable(jvp_func)
            assert callable(vjp_func)

            # File should contain all functions
            content = ode_file.file_path.read_text()
            assert "def dxdt_factory" in content
            assert "def jvp_factory" in content
            assert "def vjp_factory" in content

    def test_realistic_biochemical_system(self, temp_dir):
        """Test with a realistic biochemical reaction system."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            ode_file = ODEFile("biochemical_system", "hash")

            # Define a simple enzyme kinetics system
            states = ["Su", "En", "ES", "Pr"]  # Substrate, Enzyme, Complex,
            # Product
            parameters = ["k1", "k2", "k3"]  # Rate constants
            constants = ["Et"]  # Total enzyme
            observables = []
            drivers = []

            indexed_bases = IndexedBases.from_user_inputs(
                states=states,
                parameters=parameters,
                constants=constants,
                observables=observables,
                drivers=drivers,
            )

            Su, En, ES, P, k1, k2, k3, Et = sp.symbols(
                "Su En ES Pr k1 k2 k3 Et"
            )

            # Michaelis-Menten kinetics: S + E <-> ES -> P + E
            equations = [
                (sp.Symbol("dS"), -k1 * Su * En + k2 * ES),
                (sp.Symbol("dE"), -k1 * Su * En + k2 * ES + k3 * ES),
                (sp.Symbol("dES"), k1 * Su * En - k2 * ES - k3 * ES),
                (sp.Symbol("dP"), k3 * ES),
            ]

            # Test all factory types
            dxdt_func = ode_file.get_dxdt_fac(equations, indexed_bases)
            jvp_func = ode_file.get_jvp_fac(equations, indexed_bases)
            vjp_func = ode_file.get_vjp_fac(equations, indexed_bases)

            assert callable(dxdt_func)
            assert callable(jvp_func)
            assert callable(vjp_func)

            # File should be substantial
            content = ode_file.file_path.read_text()
            assert len(content) > 1000  # Should be a substantial file

    def test_caching_behavior(self, temp_dir, simple_equations, indexed_bases):
        """Test that caching works correctly across multiple ODEFile instances."""
        with patch(
            "cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir
        ):
            hash_value = "consistent_hash"

            # Create first instance and generate function
            ode_file1 = ODEFile("cached_system", hash_value)
            ode_file1.get_dxdt_fac(simple_equations, indexed_bases)
            original_mtime = ode_file1.file_path.stat().st_mtime

            # Create second instance with same hash
            ode_file2 = ODEFile("cached_system", hash_value)

            # we should be able to get the function without the file changing
            factory_func = ode_file2.get_dxdt_fac(
                simple_equations, indexed_bases
            )
            new_mtime = ode_file2.file_path.stat().st_mtime

            assert callable(factory_func)
            assert original_mtime == new_mtime

            # And a different hash value should modify the file
            ode_file3 = ODEFile("cached_system", hash_value[0:4])
            new_mtime = ode_file3.file_path.stat().st_mtime
            # File should not have been modified
            assert original_mtime != new_mtime

    def test_constant_changes_cache(self):
        # Test that changing constants causes a rebuild

        # Create a simple system with constants for testing
        states = {"x": 1.0, "y": 2.0}
        parameters = {"a": 0.1, "b": 0.2}
        constants_original = {"c": 0.5, "d": 1.0}
        observables = []
        drivers = []

        # Create indexed bases with original constants
        indexed_bases = IndexedBases.from_user_inputs(
            states, parameters, constants_original, observables, drivers
        )

        dx, dy, c, x, d, y = sp.symbols("dx dy c x d y", real=True)
        # Simple dxdt equations that use constants
        dxdt_str = ["dx = -c*x+d*y", "dy = c*x-d*y"]
        dxdt_equations = [(dx, -c * x + d * y), (dy, c * x - d * y)]

        # Generate hash with original constants
        hash_orig = hash_system_definition(
            dxdt_str, indexed_bases.constants.default_values
        )


        # Create ODEFile with original constants and generate function
        ode_file_orig = ODEFile("constants_test", hash_orig)

        ode_file_orig.get_dxdt_fac(dxdt_equations, indexed_bases)
        orig_constants_mtime = ode_file_orig.file_path.stat().st_mtime

        constants_modified = {c: 0.7, d: 1.0}  # Changed value of 'c'
        # Generate hash with modified constants
        hash_mod = hash_system_definition(dxdt_str, constants_modified)

        # Hashes should be different
        assert hash_orig != hash_mod

        indexed_bases.update_constants(constants_modified)
        ode_file_orig.get_dxdt_fac(dxdt_equations, indexed_bases)

        ode_file_new = ODEFile("constants_test", hash_mod)
        ode_file_new.get_dxdt_fac(dxdt_equations, indexed_bases)
        mod_constants_mtime = ode_file_new.file_path.stat().st_mtime

        # File should have been rebuilt due to changed constants
        assert orig_constants_mtime != mod_constants_mtime


def kernel_test_fac(
    equations, idx_bases, odefile, constants, precision, cse=True
):
    dxdt_fac = odefile.get_dxdt_fac(equations, idx_bases, cse=cse)
    jvp_fac = odefile.get_jvp_fac(equations, idx_bases, cse=cse)
    vjp_fac = odefile.get_vjp_fac(equations, idx_bases, cse=cse)

    nb_precision = from_dtype(precision)
    dxdt_fn = dxdt_fac(constants, nb_precision)
    jvp_fn = jvp_fac(constants, nb_precision)
    vjp_fn = vjp_fac(constants, nb_precision)

    @cuda.jit()
    def kernel(state, parameters, drivers, observables, dxdt, v, jvp, vjp):
        dxdt_fn(state, parameters, drivers, observables, dxdt)
        jvp_fn(state, parameters, drivers, v, jvp)
        vjp_fn(state, parameters, drivers, v, vjp)

    return kernel


def numerical_tester(
    kernel, state_vals, param_vals, driver_vals, v, num_observables, precision
):
    d_dxdt = cuda.device_array(len(state_vals), dtype=precision)
    d_jvp = cuda.device_array(len(state_vals), dtype=precision)
    d_vjp = cuda.device_array(len(state_vals), dtype=precision)
    d_observables = cuda.device_array(num_observables, dtype=precision)

    d_state_vals = cuda.to_device(state_vals)
    d_param_vals = cuda.to_device(param_vals)
    d_driver_vals = cuda.to_device(driver_vals)
    d_v = cuda.to_device(v)

    kernel[1, 1](
        d_state_vals,
        d_param_vals,
        d_driver_vals,
        d_observables,
        d_dxdt,
        d_v,
        d_jvp,
        d_vjp,
    )

    dxdt = d_dxdt.copy_to_host()
    jvp = d_jvp.copy_to_host()
    vjp = d_vjp.copy_to_host()

    return dxdt, jvp, vjp


def test_biochemical_numerical(temp_dir, precision):
    with patch("cubie.systemmodels.symbolic.odefile.GENERATED_DIR", temp_dir):
        ode_file = ODEFile("biochemical_system", "hash")

        # Define a simple enzyme kinetics system
        states = ["Su", "En", "ES", "Pr"]  # Substrate, Enzyme, Complex,
        # Product
        parameters = ["k1", "k2", "k3"]  # Rate constants
        constants = ["Et"]  # Total enzyme
        observables = []
        drivers = []

        indexed_bases = IndexedBases.from_user_inputs(
            states=states,
            parameters=parameters,
            constants=constants,
            observables=observables,
            drivers=drivers,
        )

        Su, En, ES, P, k1, k2, k3, Et = sp.symbols(
            "Su En ES Pr k1 k2 k3 Et", real=True
        )

        # Michaelis-Menten kinetics: S + E <-> ES -> P + E
        equations = [
            (sp.Symbol("dSu", real=True), -k1 * Su * En + k2 * ES),
            (sp.Symbol("dEn", real=True), -k1 * Su * En + k2 * ES + k3 * ES),
            (sp.Symbol("dES", real=True), k1 * Su * En - k2 * ES - k3 * ES),
            (sp.Symbol("dPr", real=True), k3 * ES),
        ]
        initial_values = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=precision)
        parameters = np.asarray([0.5, 0.5, 0.5], dtype=precision)
        drivers = np.zeros(1, dtype=precision)
        v = np.asarray([0.2, 0.1, 0.1, 0.2], dtype=precision)
        num_observables = 1
        kernel = kernel_test_fac(
            equations,
            indexed_bases,
            ode_file,
            np.asarray(1.0, dtype=precision),
            precision,
            cse=False,
        )
        dxdt, jvp, vjp = numerical_tester(
            kernel,
            initial_values,
            parameters,
            drivers,
            v,
            num_observables,
            precision,
        )

        expected_dxdt = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=precision)
        expected_jvp = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=precision)
        expected_vjp = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=precision)
        p = parameters
        x = initial_values

        expected_dxdt[0] = -p[0] * x[0] * x[1] + p[1] * x[2]
        expected_dxdt[1] = -p[0] * x[0] * x[1] + (p[1] + p[2]) * x[2]
        expected_dxdt[2] = p[0] * x[0] * x[1] - (p[1] + p[2]) * x[2]
        expected_dxdt[3] = p[2] * x[2]

        expected_jvp[0] = (
            -p[0] * x[1] * v[0] - p[0] * x[0] * v[1] + p[1] * v[2]
        )
        expected_jvp[1] = (
            -p[0] * x[1] * v[0] - p[0] * x[0] * v[1] + (p[1] + p[2]) * v[2]
        )
        expected_jvp[2] = (
            p[0] * x[1] * v[0] + p[0] * x[0] * v[1] - (p[1] + p[2]) * v[2]
        )
        expected_jvp[3] = (parameters[2] * v[2])

        expected_vjp[0] = (p[0]*x[1]*(v[2] - v[1] -v[0]))
        expected_vjp[1] = (p[0]*x[0] * (v[2] - v[1] -v[0]))
        expected_vjp[2] = (p[1]*v[0] + (p[1] + p[2]) * (v[1] - v[2]) +
                           p[2]*v[3])

        expected_vjp[3] = 0

        assert_allclose(dxdt, expected_dxdt, atol=1e-6, rtol=1e-6,
                        err_msg="dxdt")
        assert_allclose(jvp, expected_jvp, atol=1e-6, rtol=1e-6, err_msg="jvp")
        assert_allclose(vjp, expected_vjp, atol=1e-6, rtol=1e-6, err_msg="vjp")

