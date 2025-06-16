import pytest
import numpy as np
from numba import cuda, from_dtype
from CuMC.SystemModels.Systems.threeCM import ThreeChamberModel
from CuMC.ForwardSim.integrators.algorithms.euler import genericODEIntegrator

def test_euler_integration():
    """
    Test the Euler integrator by comparing it with a non-GPU Euler loop.

    The test verifies:
    1. Result is close to a CPU implementation of the same algorithm
    2. All numbers are of the right type
    3. The GPU integrator loop succesfully compiles
    """

    precision = np.float32
    numba_precision = from_dtype(precision)

    # Create and build the system
    sys = ThreeChamberModel(precision=precision)
    sys.build()

    # Set up integration parameters
    internal_step = 0.001
    save_step = 0.001
    duration = 0.1
    warmup = 0.0

    output_samples = int(duration / save_step)
    warmup_samples = int(warmup / save_step)
    save_every = int(save_step / internal_step)

    # Create and build the integrator under test
    integrator = genericODEIntegrator(precision=precision)
    integrator.build_loop(sys, internal_step, save_step)
    intfunc = integrator.integratorLoop

    # Define the test kernel
    @cuda.jit()
    def loop_test_kernel(inits, params, forcing_vector, output, observables):
        c_forcing_vector = cuda.const.array_like(forcing_vector)
        shared_memory = cuda.shared.array(0, dtype=numba_precision)

        intfunc(
            inits,
            params,
            c_forcing_vector,
            shared_memory,
            output,
            observables,
            output_samples,
            warmup_samples
        )

    # Allocate memory for inputs and outputs
    output = cuda.pinned_array((output_samples, sys.num_states), dtype=precision)
    observables = cuda.pinned_array((output_samples, sys.num_observables), dtype=precision)
    forcing_vector = cuda.pinned_array(((output_samples + warmup_samples)*save_every, sys.num_drivers), dtype=precision)

    # Initialize arrays
    output[:, :] = precision(0.0)
    observables[:, :] = precision(0.0)
    forcing_vector[:, :] = precision(0.0)
    forcing_vector[0, :] = precision(1.0)

    # Copy to device
    d_forcing = cuda.to_device(forcing_vector)
    d_inits = cuda.to_device(sys.init_values.values_array)
    d_params = cuda.to_device(sys.parameters.values_array)
    d_output = cuda.to_device(output)
    d_observables = cuda.to_device(observables)

    # Get shared memory requirements
    sharedmem = integrator.get_shared_memory_requirements(sys)

    # Run the kernel
    loop_test_kernel[1, 1, 0, sharedmem](
        d_inits,
        d_params,
        d_forcing,
        d_output,
        d_observables
    )

    # Synchronize and copy results back to host
    cuda.synchronize()
    cuda_output = d_output.copy_to_host()
    cuda_obs = d_observables.copy_to_host()

    # Verify all numbers are of the right type
    assert cuda_output.dtype == precision, f"Output array has wrong dtype: {cuda_output.dtype} instead of {precision}"
    assert cuda_obs.dtype == precision, f"Observables array has wrong dtype: {cuda_obs.dtype} instead of {precision}"

    # Implement a scipy version for comparison
    # Define the ODE function for scipy
    def three_chamber_ode(state, parameters, driver_value):
        """NON-CUDA version of the three chamber model"""
        # Extract parameters
        E_h = parameters[0]
        E_a = parameters[1]
        E_v = parameters[2]
        R_i = parameters[3]
        R_o = parameters[4]
        R_c = parameters[5]

        V_h = state[0]
        V_a = state[1]
        V_v = state[2]

        # Calculate auxiliary values
        P_a = E_a * V_a
        P_v = E_v * V_v
        P_h = E_h * V_h * driver_value[0]
        Q_i = ((P_v - P_h) / R_i) if (P_v > P_h) else 0
        Q_o = ((P_h - P_a) / R_o) if (P_h > P_a) else 0
        Q_c = (P_a - P_v) / R_c

        # Calculate gradient
        dV_h = Q_i - Q_o
        dV_a = Q_o - Q_c
        dV_v = Q_c - Q_i

        return np.asarray([dV_h, dV_a, dV_v])


    # Get initial values and parameters
    init_values = sys.init_values.values_array.copy()
    params = sys.parameters.values_array.copy()

    y = init_values.copy()
    CPU_result = np.zeros((len(init_values), output_samples))

    # Euler integration with fixed step size
    dt = internal_step


    for i in range(output_samples + warmup_samples):
        for j in range(save_every):
            # Calculate derivatives
            driver_value = forcing_vector[i*save_every + j]
            dydt = three_chamber_ode(y, params, driver_value)

            # Euler step
            y = y + dydt * dt


        # Store the state at this output time
        if i > (warmup_samples - 1):
            # Store the state in the CPU result array:
            CPU_result[:, i-warmup_samples] = y


    # Compare results
    # Note: We expect some differences due to floating point math
    print("CUDA output (first few values):")
    print(cuda_output[:5, :])
    print("\nCPU output (first few values):")
    print(CPU_result[:, :5].T)

    for i in range(sys.num_states):
        assert np.allclose(cuda_output[i,:], CPU_result[:,i], rtol=1e-3, atol=1e-6), \
            f"State {i} results don't match between CUDA and scipy"

    # The test passes if we get here, indicating successful compilation and execution
    assert True, "Test completed successfully"
