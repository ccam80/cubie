import numpy as np
import pytest

from cubie.array_interpolator import ArrayInterpolator
from tests.integrators.cpu_reference.cpu_utils import (
    DriverEvaluator,
    krylov_solve,
    newton_solve,
)


# Each case runs two sequential solves sharing one contraction
# history store, mirroring the device edge cases.
@pytest.mark.parametrize(
    "mode, initials, atol, max_iters, expected_converged, "
    "expected_iters, expected_finals",
    [
        ("zero", (3.0, 3.0), 1e-6, 4, (True, True), (1, 1), (3.0, 3.0)),
        (
            "linear",
            (3.0, 3.9999),
            1e-2,
            8,
            (True, True),
            (2, 1),
            (4.0, 4.0),
        ),
        (
            "constant",
            (0.0, 0.0),
            1e-2,
            4,
            (False, False),
            (2, 2),
            (-1.0, -1.0),
        ),
        (
            "root",
            (1.0, 1.0),
            1e-2,
            4,
            (False, False),
            (2, 2),
            (-3.0, -3.0),
        ),
        (
            "linear-failure",
            (3.0, 3.0),
            1e-2,
            3,
            (False, False),
            (3, 3),
            (3.0, 3.0),
        ),
    ],
    ids=(
        "small-first-step",
        "warm-start",
        "stagnation-divergence",
        "theta-growth-divergence",
        "linear-failure-no-commit",
    ),
)
def test_cpu_newton_convergence_edges(
    mode,
    initials,
    atol,
    max_iters,
    expected_converged,
    expected_iters,
    expected_finals,
):
    """The CPU Newton mirror matches the device convergence rules."""
    precision = np.float32
    target = precision(4.0)

    def residual(state):
        if mode == "zero":
            return np.zeros_like(state)
        if mode == "constant":
            return np.ones_like(state)
        if mode == "root":
            magnitude = np.abs(state) ** precision(0.25)
            return np.copysign(magnitude, state).astype(precision)
        return (target - state).astype(precision)

    def jacobian(state):
        if mode == "root":
            value = precision(0.25) * np.abs(state[0]) ** precision(
                -0.75
            )
        elif mode in ("linear", "linear-failure"):
            value = precision(-1.0)
        else:
            value = precision(1.0)
        return np.array([[value]], dtype=precision)

    def linear_solver(jac, rhs, **kwargs):
        direction = np.asarray(rhs, dtype=precision) / jac[0, 0]
        lin_converged = mode != "linear-failure"
        return direction.astype(precision), lin_converged, 1

    prev_theta_store = np.zeros(1, dtype=precision)
    for solve_index in range(2):
        state = np.array([initials[solve_index]], dtype=precision)
        result, converged, iterations = newton_solve(
            state,
            precision=precision,
            residual_fn=residual,
            jacobian_fn=jacobian,
            linear_solver=linear_solver,
            newton_tol=precision(atol),
            newton_rtol=precision(0.0),
            newton_max_iters=max_iters,
            prev_theta_store=prev_theta_store,
        )

        assert bool(converged) is expected_converged[solve_index]
        assert iterations == expected_iters[solve_index]
        np.testing.assert_allclose(
            result,
            np.array(
                [expected_finals[solve_index]], dtype=precision
            ),
            atol=1e-3,
        )


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
        "driver_sample_period": dt,
        "wrap": False,
        "order": 3,
        "driver": driver_values.astype(precision),
    }

    interpolator = ArrayInterpolator(precision=precision, input_dict=driver_dict)
    evaluator = DriverEvaluator(
        coefficients=interpolator.coefficients,
        dt=precision(interpolator.driver_sample_period),
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


@pytest.mark.parametrize(
    "correction_type", ["minimal_residual", "bicgstab"]
)
def test_cpu_krylov_reduction_measures_entry_rhs(
    precision, correction_type
):
    """The CPU solve stops once the residual is reduced below the
    target fixed from the untouched right-hand side."""
    matrix = np.eye(3, dtype=precision)
    rhs = np.array([10.0, -20.0, 30.0], dtype=precision)
    guess = (precision(0.95) * rhs).astype(precision)

    solution, converged, iterations = krylov_solve(
        matrix,
        rhs,
        tolerance=precision(1.0),
        max_iterations=8,
        precision=precision,
        rtol=precision(0.0),
        neumann_order=0,
        correction_type=correction_type,
        initial_guess=guess,
        norm_reference=np.zeros(3, dtype=precision),
        residual_reduction=0.1,
        residual_floor=0.0,
    )

    assert converged
    assert iterations == 0
    assert np.array_equal(solution, guess)


def test_cpu_krylov_default_floor_derives_sqrt_eps(precision):
    """Unset stopping settings derive the sqrt(eps) absolute term.

    The derived defaults must reproduce an explicit
    ``sqrt(eps)``/``eps`` pin exactly (same iterates, same solution)
    and stop later than an explicit envelope-level floor.
    """
    matrix = np.diag(np.array([4.0, 3.0, 2.0], dtype=precision))
    rhs = np.array([1.0, 2.0, 3.0], dtype=precision)
    expected = rhs / np.diag(matrix)
    eps = float(np.finfo(precision).eps)

    common = dict(
        tolerance=precision(1e-5),
        max_iterations=64,
        precision=precision,
        neumann_order=0,
    )
    derived, converged_default, iters_default = krylov_solve(
        matrix, rhs, **common
    )
    pinned, converged_pinned, iters_pinned = krylov_solve(
        matrix,
        rhs,
        residual_reduction=eps,
        residual_floor=eps ** 0.5,
        **common,
    )
    envelope, _, iters_envelope = krylov_solve(
        matrix, rhs, residual_floor=1.0, **common
    )

    assert converged_default and converged_pinned
    assert iters_default == iters_pinned
    assert np.array_equal(derived, pinned)
    assert iters_envelope < iters_default
    assert np.allclose(derived, expected, atol=1e-4)
