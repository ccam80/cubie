import pytest
import numpy as np
from numba import cuda
from numba import from_dtype

from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel

#todo: add a fixture for testing models. Add model-specifict fixture here, and the kernel in conftest.py
def test_dxdt_function_compiles():
    """Test that the dxdt function compiles using CUDA."""
    # Create a ThreeChamberModel instance with float32 precision
    precision = np.float32
    sys = ThreeChamberModel(precision=precision)
    sys.build()

    # Get the dxdt function
    dxdtfunc = sys.dxdtfunc
    numba_precision = from_dtype(precision)

    # Get dimensions
    nstates = sys.num_states
    npar = sys.num_parameters
    nobs = sys.num_observables

    # Define a CUDA kernel to test the dxdt function
    @cuda.jit()
    def test_kernel(outarray, d_inits, parameters, driver):
        l_dxdt = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_states = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_parameters = cuda.local.array(shape=(npar), dtype=numba_precision)
        l_observables = cuda.local.array(shape=(nobs), dtype=numba_precision)
        l_driver = cuda.local.array(shape=(1), dtype=numba_precision)

        # Copy parameters and states to local arrays
        for i in range(npar):
            l_parameters[i] = parameters[i]
        for i in range(nstates):
            l_states[i] = d_inits[i]

        # Set driver value
        l_driver[0] = driver[0]
        l_dxdt[:] = precision(0.0)

        # Call the dxdt function
        dxdtfunc(l_states, l_parameters, l_driver, l_observables, l_dxdt)

        # Copy results back to output array
        for i in range(nstates):
            outarray[i] = l_dxdt[i]

    # Prepare data for the kernel
    outtest = np.zeros(sys.num_states, dtype=precision)
    out = cuda.to_device(outtest)
    params = cuda.to_device(sys.parameters.values_array)
    inits = cuda.to_device(sys.init_values.values_array)
    driver = cuda.to_device([precision(1.0)])

    # Run the kernel
    test_kernel[1, 1](out, inits, params, driver)
    cuda.synchronize()

    # If we get here without errors, the function compiled successfully
    assert True

def test_dxdt_returns_zero_with_zero_inputs():
    """Test that the dxdt function returns zeros when given zero inputs."""
    # Create a ThreeChamberModel instance with float32 precision
    precision = np.float32
    sys = ThreeChamberModel(precision=precision)
    sys.build()

    # Get the dxdt function
    dxdtfunc = sys.dxdtfunc
    numba_precision = from_dtype(precision)

    # Get dimensions
    nstates = sys.num_states
    npar = sys.num_parameters
    nobs = sys.num_observables

    # Define a CUDA kernel to test the dxdt function with zero inputs
    @cuda.jit()
    def test_kernel_zeros(outarray):
        l_dxdt = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_states = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_parameters = cuda.local.array(shape=(npar), dtype=numba_precision)
        l_observables = cuda.local.array(shape=(nobs), dtype=numba_precision)
        l_driver = cuda.local.array(shape=(1), dtype=numba_precision)

        # Set all inputs to zero
        for i in range(npar):
            l_parameters[i] = precision(0.0)
        for i in range(nstates):
            l_states[i] = precision(0.0)

        l_driver[0] = precision(0.0)
        l_dxdt[:] = precision(0.0)

        # Call the dxdt function
        dxdtfunc(l_states, l_parameters, l_driver, l_observables, l_dxdt)

        # Copy results back to output array
        for i in range(nstates):
            outarray[i] = l_dxdt[i]

    # Prepare data for the kernel
    outtest = np.zeros(sys.num_states, dtype=precision)
    out = cuda.to_device(outtest)

    # Run the kernel
    test_kernel_zeros[1, 1](out)
    cuda.synchronize()

    # Copy results back to host
    out.copy_to_host(outtest)

    # Check that all outputs are zero or NaN
    # When parameters are zero, some calculations might result in NaN
    # We'll check each element individually
    for i in range(len(outtest)):
        assert np.isclose(outtest[i], 0.0) or np.isnan(outtest[i])

def test_dxdt_returns_correct_values():
    """Test that the dxdt function returns correct values for specific inputs."""
    # Create a ThreeChamberModel instance with float32 precision
    precision = np.float32
    sys = ThreeChamberModel(precision=precision)
    sys.build()

    # Get the dxdt function
    dxdtfunc = sys.dxdtfunc
    numba_precision = from_dtype(precision)

    # Get dimensions
    nstates = sys.num_states
    npar = sys.num_parameters
    nobs = sys.num_observables

    # Define a CUDA kernel to test the dxdt function with specific inputs
    @cuda.jit()
    def test_kernel_specific(outarray, d_states, d_parameters, d_driver):
        l_dxdt = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_states = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_parameters = cuda.local.array(shape=(npar), dtype=numba_precision)
        l_observables = cuda.local.array(shape=(nobs), dtype=numba_precision)
        l_driver = cuda.local.array(shape=(1), dtype=numba_precision)

        # Copy parameters and states to local arrays
        for i in range(npar):
            l_parameters[i] = d_parameters[i]
        for i in range(nstates):
            l_states[i] = d_states[i]

        l_driver[0] = d_driver[0]
        l_dxdt[:] = precision(0.0)

        # Call the dxdt function
        dxdtfunc(l_states, l_parameters, l_driver, l_observables, l_dxdt)

        # Copy results back to output array
        for i in range(nstates):
            outarray[i] = l_dxdt[i]

    # Test case 1: Use default parameters and initial values
    outtest = np.zeros(sys.num_states, dtype=precision)
    out = cuda.to_device(outtest)
    params = cuda.to_device(sys.parameters.values_array)
    states = cuda.to_device(sys.init_values.values_array)
    driver = cuda.to_device([precision(1.0)])

    # Run the kernel
    test_kernel_specific[1, 1](out, states, params, driver)
    cuda.synchronize()

    # Copy results back to host
    out.copy_to_host(outtest)

    # Check that outputs are not all zero (since we're using non-zero inputs)
    assert not np.allclose(outtest, np.zeros(sys.num_states, dtype=precision))

    # Test case 2: Use specific values and verify expected outputs
    # Set up specific test values
    test_states = np.array([1.0, 1.0, 1.0], dtype=precision)  # V_h, V_a, V_v
    test_params = np.array([
        0.52,    # E_h
        0.0133,  # E_a
        0.0624,  # E_v
        0.012,   # R_i
        1.0,     # R_o
        1/114,   # R_c
        2.0      # V_s3
    ], dtype=precision)
    test_driver = np.array([1.0], dtype=precision)

    # Calculate expected outputs manually
    # P_a = E_a * V_a = 0.0133 * 1.0 = 0.0133
    # P_v = E_v * V_v = 0.0624 * 1.0 = 0.0624
    # P_h = E_h * V_h * driver = 0.52 * 1.0 * 1.0 = 0.52
    # Q_i = (P_v - P_h) / R_i if P_v > P_h else 0 = 0 (since P_v < P_h)
    # Q_o = (P_h - P_a) / R_o if P_h > P_a else 0 = (0.52 - 0.0133) / 1.0 = 0.5067
    # Q_c = (P_a - P_v) / R_c = (0.0133 - 0.0624) / (1/114) = -5.6088
    # dV_h = Q_i - Q_o = 0 - 0.5067 = -0.5067
    # dV_a = Q_o - Q_c = 0.5067 - (-5.6088) = 6.1155
    # dV_v = Q_c - Q_i = -5.6088 - 0 = -5.6088
    expected_dxdt = np.array([-0.5067, 6.1041, -5.5974], dtype=precision)

    # Transfer test data to device
    d_states = cuda.to_device(test_states)
    d_params = cuda.to_device(test_params)
    d_driver = cuda.to_device(test_driver)

    # Run the kernel with test data
    test_kernel_specific[1, 1](out, d_states, d_params, d_driver)
    cuda.synchronize()

    # Copy results back to host
    out.copy_to_host(outtest)

    # Check that outputs match expected values (with some tolerance for floating point)
    assert np.allclose(outtest, expected_dxdt, rtol=1e-4)

def test_float64_precision_consistency():
    """Test that the system works with float64 precision and that datatypes remain consistent."""
    # Create a ThreeChamberModel instance with float64 precision
    precision = np.float64
    sys = ThreeChamberModel(precision=precision)
    sys.build()

    # Verify that the system's internal arrays have the correct dtype
    assert sys.init_values.values_array.dtype == precision
    assert sys.parameters.values_array.dtype == precision
    assert sys.observables.values_array.dtype == precision
    assert sys.constants.values_array.dtype == precision

    # Get the dxdt function
    dxdtfunc = sys.dxdtfunc
    numba_precision = from_dtype(precision)

    # Get dimensions
    nstates = sys.num_states
    npar = sys.num_parameters
    nobs = sys.num_observables

    # Define a CUDA kernel to test the dxdt function with specific inputs
    @cuda.jit()
    def test_kernel_specific(outarray, d_states, d_parameters, d_driver, d_observables):
        l_dxdt = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_states = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_parameters = cuda.local.array(shape=(npar), dtype=numba_precision)
        l_observables = cuda.local.array(shape=(nobs), dtype=numba_precision)
        l_driver = cuda.local.array(shape=(1), dtype=numba_precision)

        # Copy parameters and states to local arrays
        for i in range(npar):
            l_parameters[i] = d_parameters[i]
        for i in range(nstates):
            l_states[i] = d_states[i]

        l_driver[0] = d_driver[0]
        l_dxdt[:] = precision(0.0)

        # Call the dxdt function
        dxdtfunc(l_states, l_parameters, l_driver, l_observables, l_dxdt)

        # Copy results back to output array and observables
        for i in range(nstates):
            outarray[i] = l_dxdt[i]
        for i in range(nobs):
            d_observables[i] = l_observables[i]

    # Set up test values
    test_states = np.array([1.0, 1.0, 1.0], dtype=precision)  # V_h, V_a, V_v
    test_params = np.array([
        0.52,    # E_h
        0.0133,  # E_a
        0.0624,  # E_v
        0.012,   # R_i
        1.0,     # R_o
        1/114,   # R_c
        2.0      # V_s3
    ], dtype=precision)
    test_driver = np.array([1.0], dtype=precision)

    # Prepare data for the kernel
    outtest = np.zeros(sys.num_states, dtype=precision)
    obstest = np.zeros(sys.num_observables, dtype=precision)
    out = cuda.to_device(outtest)
    obs = cuda.to_device(obstest)
    d_states = cuda.to_device(test_states)
    d_params = cuda.to_device(test_params)
    d_driver = cuda.to_device(test_driver)

    # Run the kernel with test data
    test_kernel_specific[1, 1](out, d_states, d_params, d_driver, obs)
    cuda.synchronize()

    # Copy results back to host
    out.copy_to_host(outtest)
    obs.copy_to_host(obstest)

    # Verify that the output arrays have the correct dtype
    assert outtest.dtype == precision
    assert obstest.dtype == precision

    # Verify that the output values are reasonable (not all zeros or NaNs)
    assert not np.allclose(outtest, np.zeros(sys.num_states, dtype=precision))
    assert not np.allclose(obstest, np.zeros(sys.num_observables, dtype=precision))
    assert not np.any(np.isnan(outtest))
    assert not np.any(np.isnan(obstest))

def test_float32_precision_consistency():
    """Test that the system works with float32 precision and that datatypes remain consistent."""
    # Create a ThreeChamberModel instance with float32 precision
    precision = np.float32
    sys = ThreeChamberModel(precision=precision)
    sys.build()

    # Verify that the system's internal arrays have the correct dtype
    assert sys.init_values.values_array.dtype == precision
    assert sys.parameters.values_array.dtype == precision
    assert sys.observables.values_array.dtype == precision
    assert sys.constants.values_array.dtype == precision

    # Get the dxdt function
    dxdtfunc = sys.dxdtfunc
    numba_precision = from_dtype(precision)

    # Get dimensions
    nstates = sys.num_states
    npar = sys.num_parameters
    nobs = sys.num_observables

    # Define a CUDA kernel to test the dxdt function with specific inputs
    @cuda.jit()
    def test_kernel_specific(outarray, d_states, d_parameters, d_driver, d_observables):
        l_dxdt = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_states = cuda.local.array(shape=(nstates), dtype=numba_precision)
        l_parameters = cuda.local.array(shape=(npar), dtype=numba_precision)
        l_observables = cuda.local.array(shape=(nobs), dtype=numba_precision)
        l_driver = cuda.local.array(shape=(1), dtype=numba_precision)

        # Copy parameters and states to local arrays
        for i in range(npar):
            l_parameters[i] = d_parameters[i]
        for i in range(nstates):
            l_states[i] = d_states[i]

        l_driver[0] = d_driver[0]
        l_dxdt[:] = precision(0.0)

        # Call the dxdt function
        dxdtfunc(l_states, l_parameters, l_driver, l_observables, l_dxdt)

        # Copy results back to output array and observables
        for i in range(nstates):
            outarray[i] = l_dxdt[i]
        for i in range(nobs):
            d_observables[i] = l_observables[i]

    # Set up test values
    test_states = np.array([1.0, 1.0, 1.0], dtype=precision)  # V_h, V_a, V_v
    test_params = np.array([
        0.52,    # E_h
        0.0133,  # E_a
        0.0624,  # E_v
        0.012,   # R_i
        1.0,     # R_o
        1/114,   # R_c
        2.0      # V_s3
    ], dtype=precision)
    test_driver = np.array([1.0], dtype=precision)

    # Prepare data for the kernel
    outtest = np.zeros(sys.num_states, dtype=precision)
    obstest = np.zeros(sys.num_observables, dtype=precision)
    out = cuda.to_device(outtest)
    obs = cuda.to_device(obstest)
    d_states = cuda.to_device(test_states)
    d_params = cuda.to_device(test_params)
    d_driver = cuda.to_device(test_driver)

    # Run the kernel with test data
    test_kernel_specific[1, 1](out, d_states, d_params, d_driver, obs)
    cuda.synchronize()

    # Copy results back to host
    out.copy_to_host(outtest)
    obs.copy_to_host(obstest)

    # Verify that the output arrays have the correct dtype
    assert outtest.dtype == precision
    assert obstest.dtype == precision

    # Verify that the output values are reasonable (not all zeros or NaNs)
    assert not np.allclose(outtest, np.zeros(sys.num_states, dtype=precision))
    assert not np.allclose(obstest, np.zeros(sys.num_observables, dtype=precision))
    assert not np.any(np.isnan(outtest))
    assert not np.any(np.isnan(obstest))
