"""Tests for cubie.integrators.loops.ode_loop_config.ODELoopConfig."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cubie.integrators.loops.ode_loop_config import ODELoopConfig
from cubie.outputhandling.output_config import OutputCompileFlags


# ── Construction / defaults ─────────────────────────────── #


def test_system_size_defaults():
    """All system size fields default to 0."""
    cfg = ODELoopConfig(precision=np.float32)
    for attr in (
        "n_states", "n_parameters", "n_drivers",
        "n_observables", "n_error", "n_counters",
    ):
        assert getattr(cfg, attr) == 0


def test_system_size_custom_values():
    """System size fields store provided values."""
    cfg = ODELoopConfig(
        precision=np.float32,
        n_states=4, n_parameters=3, n_drivers=2,
        n_observables=5, n_error=4, n_counters=8,
    )
    assert cfg.n_states == 4
    assert cfg.n_parameters == 3
    assert cfg.n_drivers == 2
    assert cfg.n_observables == 5
    assert cfg.n_error == 4
    assert cfg.n_counters == 8


def test_buffer_height_defaults():
    """Buffer height fields default to 0."""
    cfg = ODELoopConfig(precision=np.float32)
    assert cfg.state_summaries_buffer_height == 0
    assert cfg.observable_summaries_buffer_height == 0


def test_buffer_height_custom():
    """Buffer height fields store provided values."""
    cfg = ODELoopConfig(
        precision=np.float32,
        state_summaries_buffer_height=10,
        observable_summaries_buffer_height=7,
    )
    assert cfg.state_summaries_buffer_height == 10
    assert cfg.observable_summaries_buffer_height == 7


BUFFER_LOCATION_FIELDS = [
    "state_location",
    "proposed_state_location",
    "parameters_location",
    "drivers_location",
    "proposed_drivers_location",
    "observables_location",
    "proposed_observables_location",
    "error_location",
    "counters_location",
    "state_summary_location",
    "observable_summary_location",
    "dt_location",
    "accept_step_location",
    "proposed_counters_location",
]


@pytest.mark.parametrize("field_name", BUFFER_LOCATION_FIELDS)
def test_buffer_location_defaults_to_local(field_name):
    """Each buffer location field defaults to 'local'."""
    cfg = ODELoopConfig(precision=np.float32)
    assert getattr(cfg, field_name) == "local"


@pytest.mark.parametrize("field_name", BUFFER_LOCATION_FIELDS)
def test_buffer_location_accepts_shared(field_name):
    """Each buffer location field accepts 'shared'."""
    cfg = ODELoopConfig(precision=np.float32, **{field_name: "shared"})
    assert getattr(cfg, field_name) == "shared"


@pytest.mark.parametrize("field_name", BUFFER_LOCATION_FIELDS)
def test_buffer_location_rejects_invalid(field_name):
    """Buffer location fields reject values outside ['shared', 'local']."""
    with pytest.raises(ValueError):
        ODELoopConfig(precision=np.float32, **{field_name: "global"})


def test_compile_flags_default_is_output_compile_flags():
    """compile_flags defaults to an OutputCompileFlags instance."""
    cfg = ODELoopConfig(precision=np.float32)
    # isinstance justified: verifying factory produces correct type,
    # combined with value check below
    assert isinstance(cfg.compile_flags, OutputCompileFlags)
    # Default OutputCompileFlags has all flags False
    assert cfg.compile_flags.save_state is False


def test_timing_defaults_none():
    """Timing fields default to None."""
    cfg = ODELoopConfig(precision=np.float32)
    assert cfg._save_every is None
    assert cfg._summarise_every is None
    assert cfg._sample_summaries_every is None


def test_flag_defaults():
    """Boolean flag fields default to False."""
    cfg = ODELoopConfig(precision=np.float32)
    assert cfg.save_last is False
    assert cfg.save_regularly is False
    assert cfg.summarise_regularly is False


def test_device_function_defaults_none():
    """All 7 device function fields default to None."""
    cfg = ODELoopConfig(precision=np.float32)
    for attr in (
        "save_state_fn", "update_summaries_fn", "save_summaries_fn",
        "step_controller_fn", "step_function",
        "evaluate_driver_at_t", "evaluate_observables",
    ):
        assert getattr(cfg, attr) is None


def test_dt_default():
    """_dt defaults to 0.01."""
    cfg = ODELoopConfig(precision=np.float32)
    assert cfg._dt == pytest.approx(0.01)


def test_is_adaptive_default():
    """is_adaptive defaults to False."""
    cfg = ODELoopConfig(precision=np.float32)
    assert cfg.is_adaptive is False


# ── samples_per_summary property ────────────────────────── #


@pytest.mark.parametrize(
    "summarise, sample, expected_zero",
    [
        pytest.param(None, 0.02, True, id="summarise_none"),
        pytest.param(0.04, None, True, id="sample_none"),
        pytest.param(None, None, True, id="both_none"),
    ],
)
def test_samples_per_summary_returns_zero_when_none(
    summarise, sample, expected_zero,
):
    """Returns 0 when either timing parameter is None."""
    cfg = ODELoopConfig(
        precision=np.float32,
        summarise_every=summarise,
        sample_summaries_every=sample,
    )
    assert cfg.samples_per_summary == 0


def test_samples_per_summary_computes_ratio():
    """Computes integer ratio of summarise_every / sample_summaries_every."""
    cfg = ODELoopConfig(
        precision=np.float64,
        summarise_every=0.1,
        sample_summaries_every=0.02,
    )
    expected = int(round(np.float64(0.1) / np.float64(0.02)))
    assert cfg.samples_per_summary == expected


def test_samples_per_summary_warns_on_adjustment():
    """Warning emitted when adjusted value differs from raw _summarise_every."""
    # 0.03 / 0.02 = 1.5 -> rounds to 2 -> adjusted = 0.04 != 0.03
    # Actually that deviation is 0.5 which is > 0.01, so it raises.
    # We need deviation <= 0.01. Use values where rounding adjusts slightly.
    # With float32: summarise_every=0.039, sample=0.02 -> ratio ~1.95 -> round=2
    # deviation = |2 - 1.95| = 0.05 > 0.01, raises.
    # Actually the code does round() first, then int(), then checks
    # deviation = abs(raw_ratio - samples_per_summary). But raw_ratio
    # is already round(...) which is an int, so deviation is always 0.
    # Let me re-read the code.
    #
    # raw_ratio = round(summarise_every / sample_summaries_every)
    # samples_per_summary = int(raw_ratio)
    # deviation = abs(raw_ratio - samples_per_summary)
    #
    # round() returns int in Python 3, so deviation is always 0.
    # The warning branch checks: adjusted != self._summarise_every
    # So it warns when summarise_every * n != original _summarise_every.
    cfg = ODELoopConfig(
        precision=np.float64,
        summarise_every=0.099,
        sample_summaries_every=0.02,
    )
    # ratio = 0.099/0.02 = 4.95 -> round = 5 -> adjusted = 5*0.02 = 0.1
    # deviation = |5 - 5| = 0  <= 0.01 -> enters first branch
    # adjusted (0.1) != _summarise_every (0.099) -> warns
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = cfg.samples_per_summary
    assert result == 5
    assert len(w) >= 1
    assert "summarise_every adjusted from" in str(w[0].message)



# ── Precision properties ────────────────────────────────── #


@pytest.mark.parametrize(
    "prec",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
    ],
)
def test_save_every_returns_precision_cast(prec):
    """save_every returns precision-cast value or None."""
    cfg_with = ODELoopConfig(precision=prec, save_every=0.05)
    assert cfg_with.save_every == prec(0.05)
    assert type(cfg_with.save_every) == prec

    cfg_none = ODELoopConfig(precision=prec)
    assert cfg_none.save_every is None


@pytest.mark.parametrize(
    "prec",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
    ],
)
def test_summarise_every_returns_precision_cast(prec):
    """summarise_every returns precision-cast value or None."""
    cfg_with = ODELoopConfig(precision=prec, summarise_every=0.1)
    assert cfg_with.summarise_every == prec(0.1)
    assert type(cfg_with.summarise_every) == prec

    cfg_none = ODELoopConfig(precision=prec)
    assert cfg_none.summarise_every is None


@pytest.mark.parametrize(
    "prec",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
    ],
)
def test_sample_summaries_every_returns_precision_cast(prec):
    """sample_summaries_every returns precision-cast value or None."""
    cfg_with = ODELoopConfig(precision=prec, sample_summaries_every=0.02)
    assert cfg_with.sample_summaries_every == prec(0.02)
    assert type(cfg_with.sample_summaries_every) == prec

    cfg_none = ODELoopConfig(precision=prec)
    assert cfg_none.sample_summaries_every is None


@pytest.mark.parametrize(
    "prec",
    [
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
    ],
)
def test_dt_returns_precision_cast(prec):
    """dt returns precision-cast value."""
    cfg = ODELoopConfig(precision=prec, dt=0.005)
    assert cfg.dt == prec(0.005)
    assert type(cfg.dt) == prec
