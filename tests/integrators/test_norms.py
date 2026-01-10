import numpy as np

from numba import cuda

from cubie.integrators.norms import ScaledNorm, ScaledNormConfig


def test_scaled_norm_config_default_tolerance():
    """Verify ScaledNormConfig creates with default tolerances."""
    config = ScaledNormConfig(precision=np.float64, n=3)

    assert config.precision == np.float64
    assert config.n == 3
    # Default tolerances broadcast to (n,) shape
    assert config.atol.shape == (3,)
    assert config.rtol.shape == (3,)
    assert np.allclose(config.atol, 1e-6)
    assert np.allclose(config.rtol, 1e-6)


def test_scaled_norm_config_custom_tolerance():
    """Verify ScaledNormConfig accepts custom atol/rtol arrays."""
    atol = np.array([1e-4, 1e-5, 1e-6], dtype=np.float64)
    rtol = np.array([1e-3, 1e-4, 1e-5], dtype=np.float64)

    config = ScaledNormConfig(
        precision=np.float64, n=3, atol=atol, rtol=rtol
    )

    assert config.atol.shape == (3,)
    assert config.rtol.shape == (3,)
    np.testing.assert_array_almost_equal(config.atol, atol)
    np.testing.assert_array_almost_equal(config.rtol, rtol)


def test_scaled_norm_factory_builds_device_function():
    """Verify ScaledNorm factory builds a valid device function."""
    factory = ScaledNorm(
        precision=np.float64,
        n=3,
        atol=1e-6,
        rtol=1e-6,
    )

    # Access device_function property triggers build
    device_fn = factory.device_function
    assert device_fn is not None
    # Verify cache is now valid
    assert factory.cache_valid


def test_scaled_norm_converged_when_under_tolerance():
    """Verify norm returns <= 1.0 when errors within tolerance."""
    n = 3
    atol = 1e-3
    rtol = 1e-3

    factory = ScaledNorm(
        precision=np.float64,
        n=n,
        atol=atol,
        rtol=rtol,
    )

    scaled_norm_fn = factory.device_function

    # Create test kernel to invoke the device function
    @cuda.jit
    def test_kernel(values, reference, result):
        result[0] = scaled_norm_fn(values, reference)

    # Values well within tolerance: |error| << atol + rtol * |ref|
    values = cuda.to_device(np.array([1e-5, 1e-5, 1e-5], dtype=np.float64))
    reference = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    result = cuda.to_device(np.array([0.0], dtype=np.float64))

    test_kernel[1, 1](values, reference, result)

    result_host = result.copy_to_host()
    # tol = atol + rtol * |ref| = 1e-3 + 1e-3 * 1.0 = 2e-3
    # ratio = 1e-5 / 2e-3 = 0.005
    # nrm2 = mean((0.005)^2) = 2.5e-5
    assert result_host[0] <= 1.0


def test_scaled_norm_exceeds_when_over_tolerance():
    """Verify norm returns > 1.0 when errors exceed tolerance."""
    n = 3
    atol = 1e-6
    rtol = 1e-6

    factory = ScaledNorm(
        precision=np.float64,
        n=n,
        atol=atol,
        rtol=rtol,
    )

    scaled_norm_fn = factory.device_function

    # Create test kernel to invoke the device function
    @cuda.jit
    def test_kernel(values, reference, result):
        result[0] = scaled_norm_fn(values, reference)

    # Values much larger than tolerance
    values = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    reference = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    result = cuda.to_device(np.array([0.0], dtype=np.float64))

    test_kernel[1, 1](values, reference, result)

    result_host = result.copy_to_host()
    # tol = atol + rtol * |ref| = 1e-6 + 1e-6 * 1.0 = 2e-6
    # ratio = 1.0 / 2e-6 = 5e5
    # nrm2 = mean((5e5)^2) = 2.5e11 >> 1.0
    assert result_host[0] > 1.0


def test_scaled_norm_update_invalidates_cache():
    """Verify update() triggers rebuild with new tolerances."""
    factory = ScaledNorm(
        precision=np.float64,
        n=3,
        atol=1e-6,
        rtol=1e-6,
    )

    # Build the cache first
    _ = factory.device_function
    assert factory.cache_valid

    # Update tolerance
    new_atol = np.array([1e-3, 1e-3, 1e-3], dtype=np.float64)
    factory.update(atol=new_atol)

    # Cache should now be invalid
    assert not factory.cache_valid

    # Verify tolerance was updated
    np.testing.assert_array_almost_equal(factory.atol, new_atol)


def test_scaled_norm_instance_label_prefix_at_init():
    """Verify ScaledNorm accepts prefixed kwargs (e.g., krylov_atol) at init.

    When instance_label is provided, prefixed kwargs should be transformed
    to unprefixed keys before being passed to the config.
    """
    n = 3
    atol = np.array([1e-10, 1e-9, 1e-8], dtype=np.float64)
    rtol = np.array([1e-5, 1e-4, 1e-3], dtype=np.float64)

    factory = ScaledNorm(
        precision=np.float64,
        n=n,
        instance_label="krylov",
        krylov_atol=atol,
        krylov_rtol=rtol,
    )

    # Verify prefixed kwargs were transformed and applied
    np.testing.assert_array_almost_equal(factory.atol, atol)
    np.testing.assert_array_almost_equal(factory.rtol, rtol)


def test_scaled_norm_instance_label_empty_no_prefix():
    """Verify empty instance_label works with unprefixed kwargs.

    With empty instance_label, no prefix transformation occurs and
    unprefixed kwargs should work directly.
    """
    n = 3
    atol = np.array([1e-8, 1e-7, 1e-6], dtype=np.float64)
    rtol = np.array([1e-4, 1e-3, 1e-2], dtype=np.float64)

    factory = ScaledNorm(
        precision=np.float64,
        n=n,
        instance_label="",
        atol=atol,
        rtol=rtol,
    )

    # Verify unprefixed kwargs were applied directly
    np.testing.assert_array_almost_equal(factory.atol, atol)
    np.testing.assert_array_almost_equal(factory.rtol, rtol)
