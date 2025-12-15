# """Test symbolic implementation of ThreeCM model against the original."""
#
# import numpy as np
# import pytest
# from numpy.testing import assert_allclose
# from numba import cuda
#
# from cubie.odesystems.systems.threeCM import ThreeChamberModel, default_parameters, default_initial_values, default_observable_names
# from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
# from tests.odesystems._utils import generate_system_tests
#
#
# # Define the symbolic equations for the three chamber model
# threeCM_equations = [
#     # Observables/auxiliary variables
#     "P_a = E_a * V_a",
#     "P_v = E_v * V_v",
#     "P_h = E_h * V_h * d1",
#     "Q_i = (P_v - P_h) / R_i if P_v > P_h else 0",
#     "Q_o = (P_h - P_a) / R_o if P_h > P_a else 0",
#     "Q_c = (P_a - P_v) / R_c",
#     # State derivatives
#     "dV_h = Q_i - Q_o",
#     "dV_a = Q_o - Q_c",
#     "dV_v = Q_c - Q_i"
# ]
#
# # Create fixtures for symbolic models
# @pytest.fixture(scope="session")
# def symbolic_threeCM_strict():
#     """Symbolic ThreeCM with all parameters, states, and observables specified."""
#     return create_ODE_system(
#         dxdt=threeCM_equations,
#         states=default_initial_values,
#         parameters=default_parameters,
#         observables=default_observable_names,
#         drivers=["d1"],
#         name="symbolic_threeCM_strict",
#         strict=True
#     )
#
# @pytest.fixture(scope="session")
# def symbolic_threeCM_nonstrict():
#     """Symbolic ThreeCM with only equations specified."""
#     return create_ODE_system(
#         dxdt=threeCM_equations,
#         name="symbolic_threeCM_nonstrict",
#         strict=False
#     )
#
# @pytest.fixture(scope="session")
# def original_threeCM():
#     """Original ThreeCM model for comparison."""
#     return ThreeChamberModel()
#
# # Generate test data using the same utilities as the original tests
# testsets = [generate_system_tests(ThreeChamberModel, (-6, 6))[0]]
#
# @pytest.mark.parametrize(
#     "instantiate_settings, input_data, test_name",
#     testsets,
#     ids=[testset[2] for testset in testsets],
# )
# class TestSymbolicThreeCM:
#     """Test class comparing symbolic and original ThreeCM implementations."""
#
#     def build_comparison_kernel(self, system1, system2, system3):
#         """Build a CUDA kernel to test all systems with the same inputs."""
#         precision = system1.precision
#
#         # Get system sizes
#         n_states = system1.sizes.states
#         n_params = system1.sizes.parameters
#         n_obs1 = system1.sizes.observables
#         n_obs2 = system2.sizes.observables
#         n_drivers = system1.sizes.drivers
#         n_obs3 = system3.sizes.observables
#
#         state_order = system1.states.names
#         state_order_1 = system1.states.get_indices(state_order)
#         state_order_2 = system2.states.get_indices(state_order)
#         state_order_3 = system3.states.get_indices(state_order)
#
#         # Get device functions
#         dxdt1 = system1.dxdt_function
#         dxdt2 = system2.dxdt_function
#         dxdt3 = system3.dxdt_function
#
#
#         @cuda.jit()
#         def comparison_kernel(
#             dx1_out, obs1_out, dx2_out, obs2_out, dx3_out, obs3_out,
#             state1, params1, drivers1, state2, params2, drivers2,
#                 state3, params3, drivers3
#         ):
#             # Local arrays for system 1
#             l_obs1 = cuda.local.array(shape=n_obs1, dtype=precision)
#             l_dx1 = cuda.local.array(shape=n_states, dtype=precision)
#
#             # Local arrays for system 2
#             l_obs2 = cuda.local.array(shape=n_obs2, dtype=precision)
#             l_dx2 = cuda.local.array(shape=n_states, dtype=precision)
#
#             # Local arrays for system 3
#             l_obs3 = cuda.local.array(shape=n_obs3, dtype=precision)
#             l_dx3 = cuda.local.array(shape=n_states, dtype=precision)
#
#
#             # Initialize outputs
#             for i in range(n_states):
#                 l_dx1[i] = precision(0.0)
#                 l_dx2[i] = precision(0.0)
#                 l_dx3[i] = precision(0.0)
#
#             # Call device functions
#             dxdt1(state1, params1, drivers1, l_obs1, l_dx1)
#             dxdt2(state2, params2, drivers2, l_obs2, l_dx2)
#             dxdt3(state3, params3, drivers3, l_obs3, l_dx3)
#
#             # Copy outputs
#             for i in range(n_states):
#                 dx1_out[state_order_1[i]] = l_dx1[i]
#                 dx2_out[state_order_2[i]] = l_dx2[i]
#                 dx3_out[state_order_3[i]] = l_dx3[i]
#             for i in range(n_obs1):
#                 obs1_out[i] = l_obs1[i]
#             for i in range(n_obs2):
#                 obs2_out[i] = l_obs2[i]
#             for i in range(n_obs3):
#                 obs3_out[i] = l_obs3[i]
#
#         return comparison_kernel
#
#     def test_all_three_systems(
#         self, symbolic_threeCM_strict, symbolic_threeCM_nonstrict, original_threeCM,
#         instantiate_settings, input_data, test_name
#     ):
#         """Test all three systems together for consistency."""
#         precision, _, _, _, _, _ = instantiate_settings
#
#         # Build all systems
#         original_threeCM.build()
#         symbolic_threeCM_strict.build()
#         symbolic_threeCM_nonstrict.build()
#
#         parameters_order = original_threeCM.parameters.names[:-1]
#         parameter_order_1 = original_threeCM.parameters.get_indices(parameters_order)
#         parameter_order_2 = symbolic_threeCM_strict.parameters.get_indices(parameters_order)
#         parameter_order_3 = symbolic_threeCM_nonstrict.parameters.get_indices(parameters_order)
#
#         driver_index = symbolic_threeCM_nonstrict.parameters.get_indices(["d1"])[0]
#
#         state_order = original_threeCM.states.names
#         state_order_1 = original_threeCM.states.get_indices(state_order)
#         state_order_2 = symbolic_threeCM_strict.states.get_indices(state_order)
#         state_order_3 = symbolic_threeCM_nonstrict.states.get_indices(state_order)
#
#         # Create three-way comparison kernel
#         kernel = self.build_comparison_kernel(
#             original_threeCM, symbolic_threeCM_strict, symbolic_threeCM_nonstrict
#         )
#
#         # Prepare output arrays
#         dx_orig = np.zeros(original_threeCM.sizes.states, dtype=precision)
#         obs_orig = np.zeros(original_threeCM.sizes.observables, dtype=precision)
#         dx_strict = np.zeros(symbolic_threeCM_strict.sizes.states, dtype=precision)
#         obs_strict = np.zeros(symbolic_threeCM_strict.sizes.observables, dtype=precision)
#         dx_nonstrict = np.zeros(symbolic_threeCM_nonstrict.sizes.states, dtype=precision)
#         obs_nonstrict = np.zeros(symbolic_threeCM_nonstrict.sizes.observables, dtype=precision)
#
#         params_orig = np.zeros_like(input_data[1])[:-1]
#         params_strict = np.zeros_like(input_data[1])
#         params_nonstrict = np.zeros_like(input_data[1])
#
#         drivers_orig = np.zeros_like(input_data[2])
#         drivers_strict = np.zeros_like(input_data[2])
#         drivers_nonstrict = np.zeros_like(input_data[2])
#
#         state_orig = np.zeros_like(input_data[0])
#         state_strict = np.zeros_like(input_data[0])
#         state_nonstrict = np.zeros_like(input_data[0])
#         # Copy inputs to local arrays
#         n_states = len(state_orig)
#         n_params = len(params_orig)
#         n_drivers = len(drivers_orig)
#
#         states_in = input_data[0]
#         params_in = input_data[1]
#         drivers_in = input_data[2]
#
#         for i in range(n_states):
#             state_orig[state_order_1[i]] = states_in[i]
#             state_strict[state_order_2[i]] = states_in[i]
#             state_nonstrict[state_order_3[i]] = states_in[i]
#         for i in range(n_params):
#             params_orig[parameter_order_1[i]] = params_in[i]
#             params_strict[parameter_order_2[i]] = params_in[i]
#             params_nonstrict[parameter_order_3[i]] = params_in[i]
#         for i in range(n_drivers):
#             drivers_orig[i] = drivers_in[i]
#             drivers_strict[i] = drivers_in[i]
#             params_nonstrict[driver_index] = drivers_in[i]
#
#
#         # Run comparison
#         kernel[1, 1](
#             dx_orig, obs_orig, dx_strict, obs_strict, dx_nonstrict, obs_nonstrict,
#             state_orig, params_orig, drivers_orig, state_strict,
#                 params_strict, drivers_strict, state_nonstrict,
#                 params_nonstrict, drivers_nonstrict)
#
#         # Set tolerance based on precision
#         rtol = 1e-6 if precision == np.float32 else 1e-12
#         # All three should match on state derivatives
#         assert_allclose(dx_strict, dx_orig, rtol=rtol,
#                        err_msg=f"Strict vs original state derivatives mismatch in {test_name}")
#         assert_allclose(dx_nonstrict, dx_orig, rtol=rtol,
#                        err_msg=f"Nonstrict vs original state derivatives mismatch in {test_name}")
#         assert_allclose(dx_nonstrict, dx_strict, rtol=rtol,
#                        err_msg=f"Nonstrict vs strict state derivatives mismatch in {test_name}")
#
#         # Strict should match original on observables
#         assert_allclose(obs_strict, obs_orig, rtol=rtol,
#                        err_msg=f"Strict vs original observables mismatch in {test_name}")
