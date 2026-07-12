"""Central registry for ``CUBIE_*`` environment variable overrides.

Environment variables provide process-wide defaults for behaviour that can
also be controlled per-solver through arguments. Explicit arguments always
take precedence over environment values; environment values take precedence
over built-in defaults.

Values are read lazily at the point of use (factory construction or module
import), so setting a variable after the relevant object has been created
has no effect. This is deliberate: compiled device functions are cached
against their compile settings, and re-reading the environment mid-session
would bypass that bookkeeping.

Published Functions
-------------------
:func:`env_bool`
    Parse a boolean-valued environment variable.
:func:`lineinfo_default`
    Default for the ``lineinfo`` compile setting (``CUBIE_LINEINFO``).

Recognised Variables
--------------------
``CUBIE_LINEINFO``
    Compile all device functions and kernels with source-line correlation
    data (``-lineinfo``) for profilers such as Nsight Compute. Truthy
    values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive). Default
    off.
"""

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable.

    Parameters
    ----------
    name
        Environment variable name.
    default
        Value returned when the variable is unset.

    Returns
    -------
    bool
        Parsed value.

    Raises
    ------
    ValueError
        If the variable is set to an unrecognised value.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"Environment variable {name}={raw!r} is not a recognised boolean; "
        f"use one of {sorted(_TRUTHY | _FALSY - {''})}."
    )


def lineinfo_default() -> bool:
    """Return the default ``lineinfo`` compile setting.

    Reads ``CUBIE_LINEINFO`` from the environment; explicit
    ``Solver(lineinfo=...)`` arguments override this value.
    """
    return env_bool("CUBIE_LINEINFO", False)
