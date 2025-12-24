"""Test CUDA compilation timing and caching behavior."""

import time
import pytest
from cubie.odesystems.symbolic.symbolicODE import create_ODE_system
from cubie.batchsolving.solver import solve_ivp
from cubie.time_logger import _default_timelogger


@pytest.mark.nocudasim
def test_compilation_caching():
    """Test whether CUDA compilations are cached after specialise_and_compile.
    
    This test creates two SymbolicODE systems with the same equations but
    in different orders (to avoid codegen cache) and compares execution times.
    The second run should be faster if compiled kernels are cached.
    """
    # Define a reasonably complex ODE system
    # Using a simple 3-state system with nonlinear terms
    equations = [
        'dx = -k1*x + k2*y*z',
        'dy = k1*x - k2*y*z - k3*y',
        'dz = k3*y - k4*z'
    ]
    
    initial_values = {
        'x': [1.0],
        'y': [0.0],
        'z': [0.0]
    }
    
    parameters = {
        'k1': [0.1],
        'k2': [0.2],
        'k3': [0.3],
        'k4': [0.4]
    }
    
    # First run: with verbose logging
    print("\n=== First run (verbose logging) ===")
    _default_timelogger.set_verbosity('verbose')
    _default_timelogger.events = []  # Clear previous events
    
    system1 = create_ODE_system(
        dxdt=equations,
        parameters=list(parameters.keys()),
        name="TestSystem1"
    )
    
    start_time1 = time.perf_counter()
    result1 = solve_ivp(
        system=system1,
        y0=initial_values,
        parameters=parameters,
        duration=0.01,
        dt=1e-4,
        method='euler',
        settling_time=0.0,
    )
    end_time1 = time.perf_counter()
    
    time1 = end_time1 - start_time1
    print(f"First run total time: {time1:.4f}s")
    
    # Get compilation events from first run
    compile_events = [e for e in _default_timelogger.events 
                     if 'compile' in e.name.lower()]
    print(f"Number of compile events in first run: {len(compile_events)}")
    
    # Second run: same equations but swapped order (avoid codegen cache)
    # with no logging
    print("\n=== Second run (no logging, swapped equation order) ===")
    _default_timelogger.set_verbosity(None)
    _default_timelogger.events = []  # Clear events
    
    # Swap equation order to force new codegen
    equations_swapped = [
        'dz = k3*y - k4*z',
        'dx = -k1*x + k2*y*z',
        'dy = k1*x - k2*y*z - k3*y',
    ]
    
    system2 = create_ODE_system(
        dxdt=equations_swapped,
        parameters=list(parameters.keys()),
        name="TestSystem2"
    )
    
    # Reorder initial values to match
    initial_values_swapped = {
        'z': [0.0],
        'x': [1.0],
        'y': [0.0],
    }
    
    start_time2 = time.perf_counter()
    result2 = solve_ivp(
        system=system2,
        y0=initial_values_swapped,
        parameters=parameters,
        duration=0.01,
        dt=1e-4,
        method='euler',
        settling_time=0.0,
    )
    end_time2 = time.perf_counter()
    
    time2 = end_time2 - start_time2
    print(f"Second run total time: {time2:.4f}s")
    
    # Compare times
    print("\n=== Comparison ===")
    print(f"First run time:  {time1:.4f}s")
    print(f"Second run time: {time2:.4f}s")
    print(f"Speedup: {time1/time2:.2f}x")
    
    if time2 < time1:
        print("✓ Second run was faster - compilations may be cached")
    else:
        print("✗ Second run was not faster - compilations may not be cached")
    
    # Reset logger
    _default_timelogger.set_verbosity(None)
    _default_timelogger.events = []
    
    # Assert results are valid
    assert result1 is not None
    assert result2 is not None


@pytest.mark.nocudasim
def test_timelogger_default_mode_printing():
    """Test TimeLogger printing behavior in default mode.
    
    In default mode, the user wants:
    - Codegen summary to print after parsing is complete
    - Compilation time to print after top-level compile is completed
    - Runtime to print after kernels return
    """
    # This test documents the current behavior
    # Future implementation should modify print_summary or add new methods
    # to print category-specific summaries at appropriate times
    
    equations = ['dx = -k*x',
        'dy = k*x'
    ]

    initial_values = {
        'x': [1.0],
        'y': [0.0]
    }
    
    parameters = {'k': [0.5]}
    
    _default_timelogger.set_verbosity('default')
    _default_timelogger.events = []
    
    system = create_ODE_system(
        dxdt=equations,
        parameters=list(parameters.keys()),
        name="TestPrintingSystem"
    )
    
    result = solve_ivp(
        system=system,
        y0=initial_values,
        parameters=parameters,
        duration=0.01,

        method='radau',
        settling_time=0.0,
    )
    
    # Print summary at the end
    _default_timelogger.print_summary()
    
    # Get events by category
    codegen_events = [e for e in _default_timelogger.events 
                     if _default_timelogger._event_registry.get(e.name, {}).get('category') == 'codegen']
    compile_events = [e for e in _default_timelogger.events 
                     if _default_timelogger._event_registry.get(e.name, {}).get('category') == 'compile']
    runtime_events = [e for e in _default_timelogger.events 
                     if _default_timelogger._event_registry.get(e.name, {}).get('category') == 'runtime']
    
    print(f"\nCodegen events: {len(codegen_events)}")
    print(f"Compile events: {len(compile_events)}")
    print(f"Runtime events: {len(runtime_events)}")
    
    # Reset logger
    _default_timelogger.set_verbosity(None)
    _default_timelogger.events = []
    
    assert result is not None


if __name__ == "__main__":
    # Allow running this test file directly
    print("Running compilation timing tests...")
    test_compilation_caching()
    test_timelogger_default_mode_printing()
