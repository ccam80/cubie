"""Structural tests for :mod:`cubie.integrators.loops.ode_loop`."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest


# Build, update, getter tests combined
def test_getters(
    loop_mutable,
    precision,
    solver_settings,
):
    loop = loop_mutable
    assert isinstance(loop.device_function, Callable), "Loop builds"

    #Test getters get
    assert loop.precision == precision, "precision getter"
    assert loop.save_every == precision(solver_settings['save_every']), \
        "save_every getter"
    assert loop.summarise_every == precision(solver_settings[
                                              'summarise_every']),\
        "summarise_every getter"
    # test update
    loop.update({"save_every": 2 * solver_settings["save_every"]})
    assert loop.save_every == pytest.approx(
        2 * solver_settings["save_every"], rel=1e-6, abs=1e-6
    )


@pytest.mark.parametrize(
    "solver_settings_override",
    [
        {
            "precision": np.float32,
            "duration": 0.1,
            "output_types": ["state", "time"],
            "algorithm": "euler",
            "dt": 0.01,
            "save_every": None,
            "summarise_every": None,
            "sample_summaries_every": None,
        }
    ],
    indirect=True,
)
def test_save_last_flag_from_config(loop_mutable):
    """Verify IVPLoop reads save_last flag from ODELoopConfig.

    When all timing parameters are None, ODELoopConfig sets save_last=True.
    IVPLoop.build() should read this from config.save_last.
    """
    config = loop_mutable.compile_settings
    assert config.save_last is True
