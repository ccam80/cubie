"""Tests for cubie.result_codes."""

import numpy as np
import pytest

from cubie import CUBIE_RESULT_CODES
from cubie.result_codes import decode_status_codes


@pytest.mark.parametrize(
    "member, value",
    [
        ("SUCCESS", 0),
        ("NEWTON_BACKTRACKING_NO_SUITABLE_STEP", 1),
        ("MAX_NEWTON_ITERATIONS_EXCEEDED", 2),
        ("MAX_LINEAR_ITERATIONS_EXCEEDED", 4),
        ("STEP_TOO_SMALL", 8),
        ("DT_EFF_EFFECTIVELY_ZERO", 16),
        ("MAX_LOOP_ITERS_EXCEEDED", 32),
        ("STAGNATION", 64),
    ],
)
def test_member_values(member, value):
    """Each result-code member holds its documented bit value."""
    assert int(CUBIE_RESULT_CODES[member]) == value


def test_decode_none_returns_empty():
    """Decoding ``None`` yields an empty mapping."""
    assert decode_status_codes(None) == {}


def test_decode_empty_returns_empty():
    """Decoding an empty array yields an empty mapping."""
    assert decode_status_codes(np.array([], dtype=np.int32)) == {}


def test_decode_omits_successful_runs():
    """Runs with status 0 are omitted from the decoded mapping."""
    codes = np.array([0, 0, 0], dtype=np.int32)
    assert decode_status_codes(codes) == {}


def test_decode_single_bits():
    """Single-bit status words decode to one named flag each."""
    codes = np.array([0, 8, 2, 64], dtype=np.int32)
    assert decode_status_codes(codes) == {
        1: ["STEP_TOO_SMALL"],
        2: ["MAX_NEWTON_ITERATIONS_EXCEEDED"],
        3: ["STAGNATION"],
    }


def test_decode_combined_bits():
    """A multi-bit status word decodes to every set flag, in bit order."""
    # 12 == MAX_LINEAR_ITERATIONS_EXCEEDED (4) | STEP_TOO_SMALL (8)
    codes = np.array([12], dtype=np.int32)
    assert decode_status_codes(codes) == {
        0: ["MAX_LINEAR_ITERATIONS_EXCEEDED", "STEP_TOO_SMALL"],
    }
