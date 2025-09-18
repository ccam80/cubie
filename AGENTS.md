## Style
Follow PEP8; max line length 79 characters, comment length 71 characters. Do not add commits that explain what you are doing 
to the user; write comments that explain unconventional or complex operations to future developers. Write numpydocs-style
docstrings for all functions and classes. Write type hints for all functions and methods.
Use descriptive variable names rather than minimal ones.
Use descriptive function names rather than minimal ones.
Type hints are compulsory, in PEP484 format in function definitions, rather than in docstrings.

## Tests
To run tests, use "pytest" from the command line. The dev environment is in Windows, so format terminal commands for powershell.
Create tests using pytests. Always use pytest fixtures, parameterised by settings dictionaries that can be indirectly overriden by "override" fixtures. Observe this pattern in tests/conftest.py.
Do not use mock or patch in tests.
A test which fails is a good test. Do not design tests to work around bugs or quirks in the code. Design tests to test 
that the code works as intended.
Never shortcut "is_device" or implement patches to get around other cuda-related checks that fail - this defeats the purpose.

## Environment
Install from workspace/cubie with pip install -e .[dev]
To run tests from an environment without CUDA drivers, set the environment variable NUMBA_ENABLE_CUDASIM="1".
If running tests without CUDA drivers, then omit pytests marked nocudasim and cupy.