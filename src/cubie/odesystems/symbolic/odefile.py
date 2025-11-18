"""Manage cached SymPy-generated Python functions.

The module writes generated functions to a cache directory and loads them on
subsequent runs, avoiding recompilation when the source equations are
unchanged.
"""

from importlib import util
from os import getcwd
from pathlib import Path
from typing import Callable, Optional

cwd = getcwd()
GENERATED_DIR = Path(cwd) / "generated"

HEADER = ("\n# This file was generated automatically by Cubie. Don't make "
          "changes in here - they'll just be overwritten! Instead, modify "
          "the sympy input which you used to define the file.\n"
          "from numba import cuda, int32\n"
          "import math\n"
          "from cubie.cuda_simsafe import *\n"
          "\n")

class ODEFile:
    """Cache generated ODE functions on disk and reload them when possible."""

    def __init__(self, system_name: str, fn_hash: int) -> None:
        """Initialise a cache file for a system definition.

        Parameters
        ----------
        system_name
            Name used when constructing the generated module filename.
        fn_hash
            Hash representing the symbolic system definition.
        """
        GENERATED_DIR.mkdir(exist_ok=True)
        self.file_path = GENERATED_DIR / f"{system_name}.py"
        self.fn_hash = fn_hash
        self._init_file(fn_hash)

    def _init_file(self, fn_hash: int) -> bool:
        """Create a new generated file when the stored hash is stale.

        Parameters
        ----------
        fn_hash
            Hash representing the symbolic system definition.

        Returns
        -------
        bool
            ``True`` when the file was (re)created, ``False`` otherwise.
        """
        if not self.cached_file_valid(fn_hash):
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(f"#{fn_hash}")
                f.write("\n")
                f.write(HEADER)
            return True
        return False

    def cached_file_valid(self, fn_hash: int) -> bool:
        """Check that the cache file exists and stores the expected hash.

        Parameters
        ----------
        fn_hash
            Hash representing the symbolic system definition.

        Returns
        -------
        bool
            ``True`` when the stored hash matches ``fn_hash``.
        """
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                existing_hash = f.readline().strip().lstrip("#")
                if existing_hash == str(fn_hash):
                    return True
        return False

    def _import_function(self, func_name: str) -> Callable:
        """Import ``func_name`` from the generated module.

        Parameters
        ----------
        func_name
            Name of the generated function to import.

        Returns
        -------
        Callable
            The imported factory function.
        """
        spec = util.spec_from_file_location(func_name, self.file_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    def import_function(
        self,
        func_name: str,
        code_lines: Optional[str] = None,
    ) -> Callable:
        """Import a generated function, generating it when absent.

        Parameters
        ----------
        func_name
            Name of the factory function to import.
        code_lines
            Source code used to generate the function when it is not cached.

        Returns
        -------
        Callable
            Imported factory function.

        Raises
        ------
        ValueError
            Raised when the function is absent from the cache and
            ``code_lines`` is ``None``.
        """
        if not self.cached_file_valid(self.fn_hash):
            self._init_file(self.fn_hash)
        text = self.file_path.read_text() if self.file_path.exists() else ""
        # Check if function is properly defined (has both definition and return)
        # Look for the function definition followed by a return at the same indentation level
        has_function = f"def {func_name}(" in text
        if has_function:
            lines = text.split('\n')
            in_function = False
            func_indent = None
            has_return = False
            for line in lines:
                if f"def {func_name}(" in line:
                    in_function = True
                    # Determine indentation level of the function
                    func_indent = len(line) - len(line.lstrip())
                elif in_function:
                    stripped = line.lstrip()
                    if not stripped:  # Empty line
                        continue
                    current_indent = len(line) - len(stripped)
                    # Check if we've left the function (dedented to same or less indentation)
                    if current_indent <= func_indent and stripped.startswith("def "):
                        break
                    # Look for return statement at the function's indentation level + one indent
                    if current_indent == func_indent + 4 and stripped.startswith("return "):
                        has_return = True
                        break
            if not has_return:
                has_function = False
        
        if not has_function:
            if code_lines is None:
                raise ValueError(
                    f"{func_name} not found in cache and no code provided."
                )
            self.add_function(code_lines, func_name)
        return self._import_function(func_name)

    def add_function(self, printed_code: str, func_name: str) -> None:
        """Append generated code to the cache file.

        Parameters
        ----------
        printed_code
            Generated source code for the function.
        func_name
            Name of the function being stored. Included for parity with the
            import pathway but unused by this method.
        """
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(printed_code)

