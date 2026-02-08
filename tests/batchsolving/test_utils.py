"""Tests for cubie.batchsolving._utils."""

from __future__ import annotations

import cubie.batchsolving._utils as utils_mod


# ── Empty module ────────────────────────────────────────── #


def test_utils_module_has_no_public_exports():
    """Module has no public names (all validators removed)."""
    public = [n for n in dir(utils_mod) if not n.startswith("_")]
    assert public == []
