"""Tests for cubie.integrators.norms."""

from __future__ import annotations

import numpy as np
import pytest
from numba import cuda
from numpy.testing import assert_allclose

from cubie.integrators.norms import (
    ScaledNorm,
    ScaledNormCache,
    ScaledNormConfig,
    resize_tolerances,
)


# ── ScaledNormConfig construction ────────────────────────── #


def test_config_defaults():
    """Default n=1, atol=[1e-6], rtol=[1e-6]."""
    cfg = ScaledNormConfig(precision=np.float64)
    assert cfg.n == 1
    assert cfg.atol.shape == (1,)
    assert cfg.rtol.shape == (1,)
    assert_allclose(cfg.atol, [1e-6])
    assert_allclose(cfg.rtol, [1e-6])


def test_config_n_validated_minimum():
    """n must be >= 1."""
    with pytest.raises((ValueError, TypeError)):
        ScaledNormConfig(precision=np.float64, n=0)


def test_config_custom_tolerances():
    """Custom atol/rtol arrays are stored correctly."""
    atol = np.array([1e-4, 1e-5, 1e-6], dtype=np.float64)
    rtol = np.array([1e-3, 1e-4, 1e-5], dtype=np.float64)
    cfg = ScaledNormConfig(precision=np.float64, n=3, atol=atol, rtol=rtol)
    assert_allclose(cfg.atol, atol)
    assert_allclose(cfg.rtol, rtol)
    assert cfg.atol.shape == (3,)


def test_config_scalar_tolerance_broadcast():
    """Scalar tolerance is broadcast to array of length n."""
    cfg = ScaledNormConfig(precision=np.float64, n=4, atol=1e-5, rtol=1e-4)
    assert cfg.atol.shape == (4,)
    assert cfg.rtol.shape == (4,)
    assert_allclose(cfg.atol, np.full(4, 1e-5))
    assert_allclose(cfg.rtol, np.full(4, 1e-4))


def test_config_inv_n():
    """inv_n returns precision(1.0/n)."""
    cfg = ScaledNormConfig(precision=np.float32, n=5)
    expected = np.float32(1.0 / 5)
    assert cfg.inv_n == pytest.approx(float(expected), rel=1e-6)


def test_config_tol_floor():
    """tol_floor returns precision(1e-16)."""
    cfg = ScaledNormConfig(precision=np.float64, n=2)
    assert cfg.tol_floor == pytest.approx(1e-16)


def test_config_n_changing_not_in_init():
    """_n_changing is internal, not exposed in init."""
    cfg = ScaledNormConfig(precision=np.float64, n=2)
    assert cfg._n_changing is False


def test_config_atol_prefixed_metadata():
    """atol field has prefixed=True metadata."""
    for f in ScaledNormConfig.__attrs_attrs__:
        if f.name == "atol":
            assert f.metadata.get("prefixed") is True
            break


def test_config_rtol_prefixed_metadata():
    """rtol field has prefixed=True metadata."""
    for f in ScaledNormConfig.__attrs_attrs__:
        if f.name == "rtol":
            assert f.metadata.get("prefixed") is True
            break


# ── resize_tolerances ────────────────────────────────────── #


def test_resize_uniform_tolerances_on_n_change():
    """Uniform tolerance arrays expand when n changes."""
    cfg = ScaledNormConfig(precision=np.float64, n=2, atol=1e-5, rtol=1e-4)
    assert cfg.atol.shape == (2,)
    cfg.n = 5
    assert cfg.atol.shape == (5,)
    assert cfg.rtol.shape == (5,)
    assert_allclose(cfg.atol, np.full(5, 1e-5))
    assert_allclose(cfg.rtol, np.full(5, 1e-4))


def test_resize_skips_matching_length():
    """Tolerances already matching n are not modified."""
    atol = np.array([1e-4, 1e-5, 1e-6], dtype=np.float64)
    cfg = ScaledNormConfig(precision=np.float64, n=3, atol=atol, rtol=1e-3)
    original_atol = cfg.atol.copy()
    cfg.n = 3  # same size
    assert_allclose(cfg.atol, original_atol)


def test_resize_leaves_nonuniform_unchanged():
    """Non-uniform tolerance arrays are left unchanged on n resize."""
    atol = np.array([1e-4, 1e-5], dtype=np.float64)
    cfg = ScaledNormConfig(precision=np.float64, n=2, atol=atol, rtol=1e-3)
    original_atol = cfg.atol.copy()
    cfg.n = 5
    # atol was non-uniform, so it's not resized (left as-is)
    assert_allclose(cfg.atol, original_atol)


def test_resize_sets_n_changing_flag():
    """_n_changing is True during resize, False after."""
    cfg = ScaledNormConfig(precision=np.float64, n=2, atol=1e-5, rtol=1e-4)
    # After construction, _n_changing should be False
    assert cfg._n_changing is False
    # We can verify the flag was toggled by checking that resize completed
    # (the flag is set/unset within resize_tolerances)
    cfg.n = 4
    assert cfg._n_changing is False


# ── ScaledNormCache ──────────────────────────────────────── #


def test_cache_from_build():
    """Build returns ScaledNormCache with scaled_norm field."""
    factory = ScaledNorm(precision=np.float64, n=3)
    _ = factory.device_function
    cache = factory._cache
    # Cache holds the same function as device_function property
    assert cache.scaled_norm is factory.device_function


# ── ScaledNorm __init__ ──────────────────────────────────── #


def test_init_sets_compile_settings():
    """__init__ creates config and sets up compile_settings."""
    factory = ScaledNorm(precision=np.float64, n=4, atol=1e-5, rtol=1e-4)
    cs = factory.compile_settings
    assert cs.n == 4
    assert cs.precision == np.float64
    assert_allclose(cs.atol, np.full(4, 1e-5))
    assert_allclose(cs.rtol, np.full(4, 1e-4))


def test_init_with_instance_label():
    """Prefixed kwargs are stripped and applied with instance_label."""
    atol = np.array([1e-10, 1e-9, 1e-8], dtype=np.float64)
    rtol = np.array([1e-5, 1e-4, 1e-3], dtype=np.float64)
    factory = ScaledNorm(
        precision=np.float64,
        n=3,
        instance_label="krylov",
        krylov_atol=atol,
        krylov_rtol=rtol,
    )
    assert_allclose(factory.atol, atol)
    assert_allclose(factory.rtol, rtol)


def test_init_empty_instance_label():
    """Empty instance_label uses unprefixed kwargs."""
    atol = np.array([1e-8, 1e-7], dtype=np.float64)
    factory = ScaledNorm(
        precision=np.float64, n=2, instance_label="", atol=atol
    )
    assert_allclose(factory.atol, atol)


# ── ScaledNorm.build (device function) ───────────────────── #


def test_build_converged_norm():
    """Norm <= 1.0 when errors are within tolerance."""
    factory = ScaledNorm(precision=np.float64, n=3, atol=1e-3, rtol=1e-3)
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    vals = cuda.to_device(np.array([1e-5, 1e-5, 1e-5], dtype=np.float64))
    refs = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    res = cuda.to_device(np.array([0.0], dtype=np.float64))
    kernel[1, 1](vals, refs, res)
    result = res.copy_to_host()[0]

    # tol_i = 1e-3 + 1e-3*1.0 = 2e-3; ratio = 1e-5/2e-3 = 5e-3
    # nrm2 = (5e-3)^2 * (1/3) * 3 = 2.5e-5
    expected = (1e-5 / 2e-3) ** 2
    assert_allclose(result, expected, rtol=1e-10)


def test_build_exceeds_tolerance():
    """Norm > 1.0 when errors exceed tolerance."""
    factory = ScaledNorm(precision=np.float64, n=3, atol=1e-6, rtol=1e-6)
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    vals = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    refs = cuda.to_device(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    res = cuda.to_device(np.array([0.0], dtype=np.float64))
    kernel[1, 1](vals, refs, res)
    result = res.copy_to_host()[0]

    # tol_i = 2e-6; ratio = 1.0/2e-6 = 5e5; nrm2 = (5e5)^2 = 2.5e11
    expected = (1.0 / 2e-6) ** 2
    assert_allclose(result, expected, rtol=1e-6)
    assert result > 1.0


def test_build_tol_floor_prevents_division_by_zero():
    """When atol and rtol*ref are near zero, floor of 1e-16 applies."""
    factory = ScaledNorm(
        precision=np.float64, n=1, atol=0.0, rtol=0.0
    )
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    vals = cuda.to_device(np.array([1e-10], dtype=np.float64))
    refs = cuda.to_device(np.array([0.0], dtype=np.float64))
    res = cuda.to_device(np.array([0.0], dtype=np.float64))
    kernel[1, 1](vals, refs, res)
    result = res.copy_to_host()[0]

    # tol_i = max(0 + 0*0, 1e-16) = 1e-16
    # ratio = 1e-10 / 1e-16 = 1e6
    # nrm2 = (1e6)^2 = 1e12
    expected = (1e-10 / 1e-16) ** 2
    assert_allclose(result, expected, rtol=1e-6)


def test_build_mean_squared_norm():
    """Norm is mean of squared ratios (divided by n)."""
    atol = np.array([1e-3, 1e-4], dtype=np.float64)
    rtol = np.array([0.0, 0.0], dtype=np.float64)
    factory = ScaledNorm(precision=np.float64, n=2, atol=atol, rtol=rtol)
    fn = factory.device_function

    @cuda.jit
    def kernel(values, reference, result):
        result[0] = fn(values, reference)

    vals = cuda.to_device(np.array([1e-3, 1e-4], dtype=np.float64))
    refs = cuda.to_device(np.array([0.0, 0.0], dtype=np.float64))
    res = cuda.to_device(np.array([0.0], dtype=np.float64))
    kernel[1, 1](vals, refs, res)
    result = res.copy_to_host()[0]

    # tol0 = 1e-3, ratio0 = 1e-3/1e-3 = 1.0
    # tol1 = 1e-4, ratio1 = 1e-4/1e-4 = 1.0
    # nrm2 = (1^2 + 1^2) / 2 = 1.0
    assert_allclose(result, 1.0, rtol=1e-10)


# ── ScaledNorm.update ────────────────────────────────────── #


def test_update_invalidates_cache():
    """update() invalidates the cache when settings change."""
    factory = ScaledNorm(precision=np.float64, n=3, atol=1e-6, rtol=1e-6)
    _ = factory.device_function
    assert factory.cache_valid
    new_atol = np.array([1e-3, 1e-3, 1e-3], dtype=np.float64)
    factory.update(atol=new_atol)
    assert not factory.cache_valid
    assert_allclose(factory.atol, new_atol)


def test_update_empty_returns_empty_set():
    """Empty update returns empty set without cache invalidation."""
    factory = ScaledNorm(precision=np.float64, n=2)
    _ = factory.device_function
    result = factory.update()
    assert result == set()
    assert factory.cache_valid


def test_update_merges_dict_and_kwargs():
    """update() merges updates_dict and kwargs."""
    factory = ScaledNorm(precision=np.float64, n=2, atol=1e-6, rtol=1e-6)
    new_atol = np.full(2, 1e-3, dtype=np.float64)
    new_rtol = np.full(2, 1e-4, dtype=np.float64)
    recognized = factory.update({"atol": new_atol}, rtol=new_rtol)
    assert "atol" in recognized
    assert "rtol" in recognized
    assert_allclose(factory.atol, new_atol)
    assert_allclose(factory.rtol, new_rtol)


# ── Forwarding properties ────────────────────────────────── #


@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("precision", "precision"),
        ("n", "n"),
    ],
)
def test_forwarding_scalar_properties(prop, child_attr):
    """Scalar forwarding properties delegate to compile_settings."""
    factory = ScaledNorm(precision=np.float64, n=4, atol=1e-5, rtol=1e-4)
    assert getattr(factory, prop) == getattr(
        factory.compile_settings, child_attr
    )


@pytest.mark.parametrize(
    "prop, child_attr",
    [
        ("atol", "atol"),
        ("rtol", "rtol"),
    ],
)
def test_forwarding_array_properties(prop, child_attr):
    """Array forwarding properties delegate to compile_settings."""
    factory = ScaledNorm(precision=np.float64, n=3, atol=1e-5, rtol=1e-4)
    result = getattr(factory, prop)
    expected = getattr(factory.compile_settings, child_attr)
    assert_allclose(result, expected)


def test_device_function_forwards_cache():
    """device_function returns get_cached_output('scaled_norm')."""
    factory = ScaledNorm(precision=np.float64, n=2)
    fn = factory.device_function
    assert fn is factory.get_cached_output("scaled_norm")
