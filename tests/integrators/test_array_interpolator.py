"""Tests for CUDA input-array interpolation helpers."""

import math
from typing import Tuple

import numpy as np
import pytest
from numba import cuda

from cubie.integrators.array_interpolator import ArrayInterpolator


@pytest.fixture(scope="session")
def quadratic_input(precision) -> ArrayInterpolator:
    """input array sampling ``f(t) = t^2`` on unit spacing."""

    times = np.arange(0.0, 6.0, 1.0, dtype=precision)
    values = times**2
    input_dict = {"values": values, "time": times, "order": 2, "wrap": False}
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    return input


@pytest.fixture(scope="session")
def cubic_inputs(precision) -> ArrayInterpolator:
    """Two input signals that are exactly represented by cubic polynomials."""

    times = np.linspace(0.0, 5.0, 11, dtype=precision)
    t = times
    input_dict = {
        "cubic1": t**3 - 2.0 * t,
        "cubic2": 0.5 * t**3 + 2 * t**2 + t,
        "time": times,
        "order": 3,
        "wrap": False,
    }
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    # Evaluation times avoid the final endpoint to prevent wrap behaviour.
    return input


@pytest.fixture(scope="session")
def wrapping_inputs(precision) -> Tuple[ArrayInterpolator, ArrayInterpolator]:
    """Return clamp and wrap input arrays sharing identical samples."""

    times = np.linspace(0.0, 4.0, 6, dtype=precision)
    values = np.array([0.0, 1.5, -0.75, 2.25, -3.0, 0.0], dtype=precision)
    clamp_input_dict = {"values": values, "time": times, "order": 3,
                        'wrap': False}
    wrap_input_dict = {"values": values, "time": times, "order": 3, 'wrap': True}
    clamp = ArrayInterpolator(
        precision=precision,
        input_dict=clamp_input_dict,
    )
    wrap = ArrayInterpolator(
        precision=precision,
        input_dict=wrap_input_dict,
    )
    return clamp, wrap


def _cpu_evaluate(
    time: float,
    coefficients: np.ndarray,
    start: float,
    resolution: float,
    wrap: bool,
    boundary_condition: str,
) -> np.ndarray:
    """Evaluate all input polynomials on the CPU.

    Parameters
    ----------
    time : float
        Query time to evaluate.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients.
    start : float
        Start time of the input samples.
    resolution : float
        Temporal resolution between consecutive samples.
    wrap : bool
        Whether the input wraps beyond the final segment.

    Returns
    -------
    numpy.ndarray
        Evaluated input values at ``time``.
    """

    pad_clamped = (not wrap) and (boundary_condition == "clamped")
    evaluation_start = start - (resolution if pad_clamped else 0.0)
    inv_res = 1.0 / resolution
    num_segments = coefficients.shape[0]
    scaled = (time - evaluation_start) * inv_res
    scaled_floor = math.floor(scaled)
    idx = int(scaled_floor)

    if wrap:
        segment = idx % num_segments
        if segment < 0:
            segment += num_segments
        tau = scaled - scaled_floor
        in_range = True
    else:
        in_range = 0.0 <= scaled <= float(num_segments)
        segment = idx if idx >= 0 else 0
        if segment >= num_segments:
            segment = num_segments - 1
        tau = scaled - float(segment)
    zero_value = coefficients.dtype.type(0.0)
    out = np.empty(coefficients.shape[1], dtype=coefficients.dtype)
    for input_index in range(coefficients.shape[1]):
        coeffs = coefficients[segment, input_index]
        acc = zero_value
        for k in range(coeffs.size - 1, -1, -1):
            acc = acc * tau + coeffs[k]
        if wrap or in_range:
            out[input_index] = acc
        else:
            out[input_index] = zero_value
    return out


def _run_evaluate(
    device_fn, coefficients: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """Execute the device evaluation function across supplied samples.

    Parameters
    ----------
    device_fn : callable
        Device function to execute.
    coefficients : numpy.ndarray
        Segment-major polynomial coefficients.
    query_times : numpy.ndarray
        Time samples to evaluate on the device.

    Returns
    -------
    numpy.ndarray
        Evaluated input values for each query time.
    """

    n_times = query_times.size
    n_inputs = coefficients.shape[1]
    out_host = np.empty((n_times, n_inputs), dtype=coefficients.dtype)

    @cuda.jit
    def kernel(times, coeffs, out):
        idx = cuda.grid(1)
        if idx < times.size:
            device_fn(times[idx], coeffs, out[idx])

    d_times = cuda.to_device(query_times)
    d_coeffs = cuda.to_device(coefficients)
    d_out = cuda.to_device(out_host)
    threads_per_block = 64
    blocks = (n_times + threads_per_block - 1) // threads_per_block
    kernel[blocks, threads_per_block](d_times, d_coeffs, d_out)
    d_out.copy_to_host(out_host)
    return out_host


def test_build_coefficients_matches_polynomial(quadratic_input, tolerance):
    """Segment-wise coefficients should reproduce the quadratic exactly."""

    coefficients = quadratic_input.coefficients[:, 0, :]
    interior = coefficients[1:-1]
    segment_starts = (
        np.arange(interior.shape[0], dtype=quadratic_input.precision)
        * quadratic_input.dt
    )
    expected = np.column_stack(
        (
            segment_starts ** 2,
            2.0 * segment_starts,
            np.ones_like(segment_starts),
        )
    )
    np.testing.assert_allclose(
        interior,
        expected,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "interior coefficients diverged from quadratic reference\n"
            f"device:\n{np.array2string(interior)}\n"
            f"expected:\n{np.array2string(expected)}"
        ),
    )

    def _horner(coeffs, tau):
        value = 0.0
        for coef in coeffs[::-1]:
            value = value * tau + coef
        return value

    leading = coefficients[0]
    trailing = coefficients[-1]
    np.testing.assert_allclose(
        [_horner(leading, tau) for tau in (0.0, 1.0)],
        [0.0, 0.0],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg="leading ghost segment should start and end at zero",
    )
    last_sample = quadratic_input.input_array[-1, 0]
    np.testing.assert_allclose(
        [_horner(trailing, tau) for tau in (0.0, 1.0)],
        [last_sample, 0.0],
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg="trailing ghost segment should decay to zero",
    )


def test_device_interpolation_matches_cpu(cubic_inputs, tolerance) -> None:
    """Device evaluation must agree with a CPU Horner evaluation."""

    input = cubic_inputs
    query_times = np.arange(input.num_samples + 2, dtype=np.int32)
    query_times = query_times.astype(input.precision) * input.dt

    coefficients = input.coefficients
    device_fn = input.evaluation_function

    gpu = _run_evaluate(device_fn, coefficients, query_times)

    cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                coefficients,
                input.t0,
                input.dt,
                wrap=False,
                boundary_condition=input.boundary_condition,
            )
            for t in query_times
        ]
    )

    np.testing.assert_allclose(
        gpu,
        cpu,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "device vs cpu Horner evaluation mismatch\n"
            f"gpu:\n{np.array2string(gpu)}\n"
            f"cpu:\n{np.array2string(cpu)}"
        ),
    )


def test_get_interpolated_matches_kernel_output(cubic_inputs):
    """Host helper should agree with explicit device kernel launches."""

    query_times = np.linspace(
        cubic_inputs.t0,
        cubic_inputs.t0 + cubic_inputs.dt * (cubic_inputs.num_segments - 1),
        17,
        dtype=cubic_inputs.precision,
    )

    expected = _run_evaluate(
        cubic_inputs.evaluation_function,
        cubic_inputs.coefficients,
        query_times,
    )
    observed = cubic_inputs.get_interpolated(query_times)

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg=(
            "get_interpolated diverged from device reference\n"
            f"observed:\n{np.array2string(observed)}\n"
            f"expected:\n{np.array2string(expected)}"
        ),
    )


def test_get_interpolated_requires_one_dimensional_input(cubic_inputs):
    """Supplying multidimensional evaluation grids should fail."""

    bad_times = np.zeros((2, 2), dtype=cubic_inputs.precision)
    with pytest.raises(ValueError):
        cubic_inputs.get_interpolated(bad_times)


def test_wrap_vs_clamp_evaluation(wrapping_inputs, tolerance) -> None:
    """Wrapping alters extrapolation compared to clamping semantics."""

    clamp_input, wrap_input = wrapping_inputs

    query_times = np.array(
        [-0.5, 0.0, 1.5, 4.0, 4.5, 6.2], dtype=clamp_input.precision
    )

    clamp_gpu = _run_evaluate(
        clamp_input.evaluation_function, clamp_input.coefficients, query_times
    )
    wrap_gpu = _run_evaluate(
        wrap_input.evaluation_function, wrap_input.coefficients, query_times
    )

    clamp_cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                clamp_input.coefficients,
                clamp_input.t0,
                clamp_input.dt,
                wrap=False,
                boundary_condition=clamp_input.boundary_condition,
            )
            for t in query_times
        ]
    )
    wrap_cpu = np.vstack(
        [
            _cpu_evaluate(
                t,
                wrap_input.coefficients,
                wrap_input.t0,
                wrap_input.dt,
                wrap=True,
                boundary_condition=wrap_input.boundary_condition,
            )
            for t in query_times
        ]
    )

    np.testing.assert_allclose(
        clamp_gpu,
        clamp_cpu,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "clamp device evaluation diverged from CPU reference\n"
            f"gpu:\n{np.array2string(clamp_gpu)}\n"
            f"cpu:\n{np.array2string(clamp_cpu)}"
        ),
    )
    np.testing.assert_allclose(
        wrap_gpu,
        wrap_cpu,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "wrap device evaluation diverged from CPU reference\n"
            f"gpu:\n{np.array2string(wrap_gpu)}\n"
            f"cpu:\n{np.array2string(wrap_cpu)}"
        ),
    )
    assert not np.allclose(
        clamp_gpu,
        wrap_gpu,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )


def test_non_wrap_returns_zero_outside_range(quadratic_input, tolerance) -> None:
    """Non-wrapping inputs must output zeros beyond sampled times."""

    input = quadratic_input
    coefficients = input.coefficients
    device_fn = input.evaluation_function
    dtype = input.precision
    end_time = input.t0 + (input.num_samples - 1) * input.dt
    query_times = np.array(
        [
            input.t0 - input.dt,
            input.t0 + 0.5 * input.dt,
            end_time + input.dt,
        ],
        dtype=dtype,
    )

    gpu = _run_evaluate(device_fn, coefficients, query_times)
    np.testing.assert_allclose(
        gpu[0],
        0.0,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "leading extrapolation should clamp to zero\n"
            f"gpu:\n{np.array2string(gpu[0])}"
        ),
    )
    np.testing.assert_allclose(
        gpu[2],
        0.0,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "trailing extrapolation should clamp to zero\n"
            f"gpu:\n{np.array2string(gpu[2])}"
        ),
    )
    assert not np.allclose(
        gpu[1],
        0.0,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
    )


def test_non_wrap_defaults_to_clamped_boundary(quadratic_input) -> None:
    """Non-wrapping inputs should apply clamped spline boundaries."""

    assert quadratic_input.boundary_condition == "clamped"


def test_wrap_repeats_periodically(wrapping_inputs, tolerance) -> None:
    """Wrapping inputs must repeat their samples beyond the final index."""

    _, wrap_input = wrapping_inputs
    coefficients = wrap_input.coefficients
    device_fn = wrap_input.evaluation_function
    period = wrap_input.num_segments * wrap_input.dt
    dtype = wrap_input.precision
    query_times = np.array(
        [
            wrap_input.t0 + 0.25 * period,
            wrap_input.t0 + 1.25 * period,
            wrap_input.t0 - 0.75 * period,
        ],
        dtype=dtype,
    )

    gpu = _run_evaluate(device_fn, coefficients, query_times)
    np.testing.assert_allclose(
        gpu[0],
        gpu[1],
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg=(
            "wrap evaluation should repeat forward period\n"
            f"gpu0:\n{np.array2string(gpu[0])}\n"
            f"gpu1:\n{np.array2string(gpu[1])}"
        ),
    )
    np.testing.assert_allclose(
        gpu[0],
        gpu[2],
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg=(
            "wrap evaluation should repeat backward period\n"
            f"gpu0:\n{np.array2string(gpu[0])}\n"
            f"gpu2:\n{np.array2string(gpu[2])}"
        ),
    )


def test_plot_interpolated_wraps_markers(wrapping_inputs):
    """Plot helper should repeat markers when wrapping is enabled."""

    plt = pytest.importorskip("matplotlib.pyplot")
    _, wrap_input = wrapping_inputs
    eval_times = np.linspace(
        wrap_input.t0 - wrap_input.dt,
        wrap_input.t0
        + wrap_input.dt * (wrap_input.num_segments + 1),
        64,
        dtype=wrap_input.precision,
    )
    fig, ax = wrap_input.plot_interpolated(eval_times)
    plt.close(fig)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_polynomial_samples_are_reproduced(order, precision, tolerance) -> None:
    """Spline evaluation must match supplied samples for polynomial data."""

    times = np.linspace(0.0, 6.0, 25, dtype=precision)
    coeffs = np.arange(order + 1, dtype=precision) + 1.0
    values = np.zeros_like(times)
    for power, coef in enumerate(coeffs):
        values += coef * times**power
    input_dict = {"values": values, "time": times, "order": order, "wrap": False}
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    gpu_samples = _run_evaluate(
        input.evaluation_function,
        input.coefficients,
        times,
    )
    np.testing.assert_allclose(
        gpu_samples[:, 0],
        values,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg=(
            "polynomial samples were not reproduced\n"
            f"gpu:\n{np.array2string(gpu_samples[:, 0])}\n"
            f"reference:\n{np.array2string(values)}"
        ),
    )

@pytest.mark.parametrize(
    "bc",
    ["natural", "periodic", "clamped", "not-a-knot"],
)
def test_order_three_matches_scipy_reference(precision, bc, tolerance) -> None:
    """Order-three interpolation should match SciPy's cubic spline."""

    scipy = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(1234)
    times = np.linspace(0.0, 2.0, 21, dtype=precision)
    samples = np.sin(2.3 * times) + 0.3 * np.cos(4.1 * times)
    samples += 0.05 * rng.standard_normal(times.size).astype(precision)
    wrap = bc == "periodic"
    if wrap:
        samples[-1] = samples[0]
    input_dict = {
        "drive": samples,
        "time": times,
        "order": 3,
        "wrap": wrap,
        "boundary_condition": bc,
    }
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )
    query = np.linspace(times[0], times[-1], 257, dtype=precision)
    gpu = _run_evaluate(input.evaluation_function, input.coefficients, query)
    scipy_samples = samples.copy()
    scipy_times = times.copy()
    if not wrap and bc == "clamped":
        dt = np.diff(times)[0]
        scipy_samples = np.hstack([precision(0), samples, precision(0)])
        scipy_times = np.hstack([times[0] - dt, times, times[-1] + dt])
    spline = scipy.CubicSpline(scipy_times, scipy_samples, bc_type=bc)
    scipy_values = np.asarray(spline(query), dtype=precision)

    np.testing.assert_allclose(
        gpu[:, 0],
        scipy_values,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg=(
            "cubic interpolation diverged from SciPy reference\n"
            f"gpu:\n{np.array2string(gpu[:, 0])}\n"
            f"scipy:\n{np.array2string(scipy_values)}"
        ),
    )


def _falling_factorial(power: int, derivative: int) -> int:
    """Return the falling factorial ``power! / (power - derivative)!``."""

    result = 1
    for step in range(derivative):
        result *= power - step
    return result


def _evaluate_derivative(
    coefficients: np.ndarray,
    segment: int,
    input_index: int,
    tau: float,
    derivative: int,
    dt: float,
) -> float:
    """Evaluate a spline derivative at ``tau`` in the supplied segment."""

    total = 0.0
    for power in range(derivative, coefficients.shape[-1]):
        factor = _falling_factorial(power, derivative)
        total += (
            coefficients[segment, input_index, power]
            * factor
            * (tau ** (power - derivative))
        )
    return total / (dt ** derivative)


def test_natural_boundary_supports_higher_orders(precision, tolerance) -> None:
    """Natural constraints should zero high derivatives for any order."""

    order = 4
    times = np.linspace(0.0, 3.0, 9, dtype=precision)
    samples = np.sin(times) + 0.25 * times**2
    input_dict = {"drive": samples, "time": times, "order": order, "wrap": False, "boundary_condition": "natural"}
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )

    coefficients = input.coefficients
    dt = input.dt
    expected_zero = []
    remaining = order - 1
    derivative = 2
    while remaining > 0 and derivative <= order:
        expected_zero.append((0, derivative))
        remaining -= 1
        if remaining == 0:
            break
        expected_zero.append((input.num_segments - 1, derivative))
        remaining -= 1
        derivative += 1

    for segment, derivative in expected_zero:
        tau = 0.0 if segment == 0 else 1.0
        value = _evaluate_derivative(coefficients, segment, 0, tau, derivative, dt)
        np.testing.assert_allclose(
            value,
            0.0,
            rtol=tolerance.rel_tight,
            atol=tolerance.abs_tight,
            err_msg=(
                "natural boundary derivative failed to vanish\n"
                f"segment={segment} derivative={derivative} value={value}"
            ),
        )

    gpu = _run_evaluate(input.evaluation_function, input.coefficients, times)
    reference = samples
    np.testing.assert_allclose(
        gpu[:, 0],
        reference,
        rtol=tolerance.rel_tight,
        atol=tolerance.abs_tight,
        err_msg="natural boundary spline failed to reproduce samples",
    )


def test_periodic_boundary_respects_general_order(precision, tolerance) -> None:
    """Periodic constraints should hold for arbitrary spline order."""

    order = 5
    num_samples = 12
    times = np.linspace(0.0, 2.0 * np.pi, num_samples, dtype=precision)
    values = np.column_stack(
        (
            np.sin(times),
            0.75 * np.cos(2.0 * times),
        )
    )
    values[0] = values[-1]
    input_dict = {"s": values[:, 0], "c": values[:, 1], "time": times, "order": order, "wrap": True, "boundary_condition": "periodic"}
    input = ArrayInterpolator(
        precision=precision,
        input_dict=input_dict,
    )

    coefficients = input.coefficients
    dt = input.dt
    last_segment = input.num_segments - 1
    for derivative in range(1, order):
        left = _evaluate_derivative(coefficients, 0, 0, 0.0, derivative, dt)
        right = _evaluate_derivative(
            coefficients,
            last_segment,
            0,
            1.0,
            derivative,
            dt,
        )
        np.testing.assert_allclose(
            left,
            right,
            rtol=tolerance.rel_loose,
            atol=tolerance.abs_loose,
            err_msg=(
                "periodic derivative mismatch for sine input\n"
                f"derivative={derivative} left={left} right={right}"
            ),
        )

    for derivative in range(1, order):
        left = _evaluate_derivative(coefficients, 0, 1, 0.0, derivative, dt)
        right = _evaluate_derivative(
            coefficients,
            last_segment,
            1,
            1.0,
            derivative,
            dt,
        )
        np.testing.assert_allclose(
            left,
            right,
            rtol=tolerance.rel_loose,
            atol=tolerance.abs_loose,
            err_msg=(
                "periodic derivative mismatch for cosine input\n"
                f"derivative={derivative} left={left} right={right}"
            ),
        )

    gpu = _run_evaluate(input.evaluation_function, input.coefficients, times)
    reference = values
    np.testing.assert_allclose(
        gpu,
        reference,
        rtol=tolerance.rel_loose,
        atol=tolerance.abs_loose,
        err_msg="periodic spline failed to reproduce samples",
    )
