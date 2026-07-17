import numpy as np
import pytest

from cubie.array_interpolator import ArrayInterpolator
from tests.integrators.cpu_reference.cpu_utils import DriverEvaluator, newton_solve


@pytest.mark.parametrize(
    "mode, initial, atol, rtol, max_iters, expected_converged, expected_iters",
    [
        ("zero", 3.0, 1e-6, 0.0, 4, True, 0),
        ("stalled", 1e8, 1.0, 1.0, 2, False, 2),
        ("small", 1e8, 1e8, 0.0, 2, False, 2),
        ("residual", 0.0, 1e-6, 0.0, 4, True, 1),
    ],
    ids=(
        "zero-residual",
        "noncontracting",
        "small-update",
        "accepted-residual",
    ),
)
def test_cpu_newton_convergence_edges(
    mode,
    initial,
    atol,
    rtol,
    max_iters,
    expected_converged,
    expected_iters,
):
    """Newton reports exact convergence iterations."""
    precision = np.float32

    def residual(state):
        if mode == "zero":
            return np.zeros_like(state)
        if mode == "residual":
            return precision(4.0) - state
        return precision(2e9) - state

    def jacobian(state):
        step = precision(1.0)
        if mode == "stalled":
            step = precision(10.0)
        elif mode == "small":
            step = precision(10000.0) - precision(1e-5) * (
                state[0] - precision(1e8)
            )
        elif mode == "residual":
            step = precision(4.0)
        return np.array([[step]], dtype=precision)

    def linear_solver(operator, rhs, **kwargs):
        return np.array([operator[0, 0]], dtype=precision), True, 1

    initial_state = np.array([initial], dtype=precision)
    result, converged, iterations = newton_solve(
        initial_state,
        precision=precision,
        residual_fn=residual,
        jacobian_fn=jacobian,
        linear_solver=linear_solver,
        newton_tol=precision(atol),
        newton_rtol=precision(rtol),
        newton_max_iters=max_iters,
        newton_damping=precision(0.5),
        newton_max_backtracks=2,
    )

    assert bool(converged) is expected_converged
    assert iterations == expected_iters
    if mode == "zero":
        np.testing.assert_array_equal(result, initial_state)
    elif mode == "residual":
        np.testing.assert_array_equal(result, np.array([4.0], dtype=precision))
    else:
        scale = precision(atol) + precision(rtol) * abs(result[0])
        assert abs(residual(result)[0]) / scale > precision(1.0)


@pytest.mark.parametrize(
    "precision, atol, rtol",
    [
        (np.float32, 5e-6, 5e-6),
        (np.float64, 1e-12, 1e-12),
    ],
)
def test_cpu_driver_evaluator_matches_gpu_time_alignment(
    precision, atol, rtol
):
    dt = precision(0.125)
    warmup = precision(0.35)
    sample_count = 12
    sample_times = warmup + dt * np.arange(sample_count, dtype=precision)
    driver_values = np.cos(sample_times * precision(1.3))
    driver_values += np.sin(sample_times * precision(0.7))

    driver_dict = {
        "t0": warmup,
        "dt": dt,
        "wrap": False,
        "order": 3,
        "driver": driver_values.astype(precision),
    }

    interpolator = ArrayInterpolator(precision=precision, input_dict=driver_dict)
    evaluator = DriverEvaluator(
        coefficients=interpolator.coefficients,
        dt=precision(interpolator.dt),
        t0=precision(interpolator.t0),
        wrap=interpolator.wrap,
        precision=precision,
        boundary_condition=interpolator.boundary_condition,
    )

    query_times = np.linspace(
        warmup - dt,
        warmup + dt * precision(sample_count - 1) + dt,
        num=40,
        dtype=precision,
    )
    gpu_values = interpolator.get_interpolated(query_times)
    cpu_values = np.stack(
        [evaluator.evaluate(float(time)) for time in query_times]
    )

    np.testing.assert_allclose(cpu_values, gpu_values, rtol=rtol, atol=atol)
