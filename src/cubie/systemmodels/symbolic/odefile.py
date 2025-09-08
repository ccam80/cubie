"""Utilities for generating Python functions from SymPy code."""

from importlib import util
from os import getcwd
from pathlib import Path

cwd = getcwd()
GENERATED_DIR = Path(cwd) / "generated"

HEADER = ("\n# This file was generated automatically by Cubie. Don't make "
          "changes in here - they'll just be overwritten! Instead, modify "
          "the sympy input which you used to define the file.\n"
          "from numba import cuda\n"
          "import math\n"
          "\n")

class ODEFile:
    """Class for managing generated files."""

    def __init__(self, system_name, fn_hash):
        GENERATED_DIR.mkdir(exist_ok=True)
        self.file_path = GENERATED_DIR / f"{system_name}.py"
        self.fn_hash = fn_hash
        self._init_file(fn_hash)

    def _init_file(self, fn_hash):
        if not self.cached_file_valid(fn_hash):
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(f"#{fn_hash}")
                f.write("\n")
                f.write(HEADER)
            return True
        else:
            return False

    def cached_file_valid(self, fn_hash):
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                existing_hash = f.readline().strip().lstrip("#")
                if existing_hash == fn_hash:
                    return True
        return False

    def _import_function(self, func_name):
        """Import *func_name* from the generated file."""
        spec = util.spec_from_file_location(func_name, self.file_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    def import_function(self, func_name: str, code_lines: str | None = None):
        """Import a factory function, generating it if needed.

        If ``func_name`` is missing from the cached file, ``code_lines`` are
        appended and the function is imported. If the function is missing and
        ``code_lines`` is ``None``, a :class:`ValueError` is raised.
        """
        if not self.cached_file_valid(self.fn_hash):
            self._init_file(self.fn_hash)
        text = self.file_path.read_text() if self.file_path.exists() else ""
        base_name = func_name.replace("_factory", "")
        if func_name not in text or f"return {base_name}" not in text:
            if code_lines is None:
                raise ValueError(
                    f"{func_name} not found in cache and no code provided."
                )
            self.add_function(code_lines, func_name)
        return self._import_function(func_name)

    def add_function(self, printed_code: str, func_name: str) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(printed_code)

