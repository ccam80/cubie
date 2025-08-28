Create tests using pytests. Always use pytest fixtures, parameterised by settings dictionaries that can be indirectly overriden by "override" fixtures. Observe this pattern in tests/conftest.py.
Do not use mock or patch in tests.

Respect a comment line length of 75 characters. Respect a code line length of 79 characters. Follow PEP8 format.

Use descriptive variable names rather than minimal ones.

Use descriptive function names rather than minimal ones.

Do not include comments explaining your actions as part of a conversation. Only add comments if critical to the understanding of a new reader to the code.

To run tests, use "pytest" from the command line. The dev environment is in Windows, so format terminal commands for powershell.

To run tests from an environment without CUDA drivers, set the environment variable NUMBA_ENABLE_CUDASIM="1"