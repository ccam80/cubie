"""Session save/load utilities for CuBIE solver configurations.

Provides functions to persist and restore solver compile settings
across Python sessions.
"""

import pickle
import warnings
from pathlib import Path
from typing import Any, TYPE_CHECKING

from cubie.odesystems.symbolic.odefile import GENERATED_DIR

if TYPE_CHECKING:
    from cubie.batchsolving.solver import Solver

SESSIONS_DIR = GENERATED_DIR / "sessions"


def save_session(solver: "Solver", name: str) -> Path:
    """Save solver compile settings to a named session file.

    Parameters
    ----------
    solver
        Solver instance whose settings will be saved.
    name
        Session name used for the file (without extension).

    Returns
    -------
    Path
        Path to the saved session file.

    Notes
    -----
    Overwrites existing sessions with the same name, emitting a
    warning.
    """
    if not name:
        raise ValueError("Session name cannot be empty")

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_path = SESSIONS_DIR / f"{name}.pkl"

    if session_path.exists():
        warnings.warn(
            f"Overwriting existing session '{name}'",
            UserWarning,
            stacklevel=2,
        )

    compile_settings = solver.kernel.compile_settings
    with open(session_path, 'wb') as f:
        pickle.dump(compile_settings, f, protocol=pickle.HIGHEST_PROTOCOL)

    return session_path


def load_from_session(name: str) -> Any:
    """Load compile settings from a named session file.

    Parameters
    ----------
    name
        Session name to load (without extension).

    Returns
    -------
    Any
        The saved compile_settings object.

    Raises
    ------
    FileNotFoundError
        If the session file does not exist.
    """
    if not name:
        raise ValueError("Session name cannot be empty")

    session_path = SESSIONS_DIR / f"{name}.pkl"

    if not session_path.exists():
        raise FileNotFoundError(
            f"Session '{name}' not found at {session_path}"
        )

    with open(session_path, 'rb') as f:
        return pickle.load(f)
