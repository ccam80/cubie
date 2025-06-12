# Project Guidelines
    
When running tests, navigate to the project root directory and execute the command `pytest`. Do not run individual test files directly or use complex pytest commands with multiple arguments.
Add tests for a given module to test_<ModuleName>.py, where <ModuleName> is the name of the module being tested. For example, tests for the `SystemParameters` module should be in `test_SystemParameters.py`.

Create new tests using pytest formats, do not set up unittests classes.

