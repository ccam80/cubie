Do not use && to join terminal commands, the default terminal is powershell and these don't work.
When creating tests, use pytest fixtures. Strongly prefer fixtures instantiating cubie objects over creating mocks or 
patches unless absolutely unavoidable.
Only run tests in current test file, as total test run is slow.
Expect any tests requiring cuda to fail. Prompt user to run tests if they use any cuda functionality.