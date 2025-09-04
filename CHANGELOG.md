# Changelog

## [0.0.3](https://github.com/ccam80/cubie/compare/v0.0.2...v0.0.3) (2025-09-04)


### Features

* Parser accepts and translates sympy and user-provided functions ([#108](https://github.com/ccam80/cubie/issues/108)) ([25f3c5d](https://github.com/ccam80/cubie/commit/25f3c5d89edca3d85040b1438995d45f010b99a4))
* Symbolic input parsing added ([14577c1](https://github.com/ccam80/cubie/commit/14577c1e237791f2997ab53ffc842b5325763766))
* symbolic interface and analytical jacobian generation added ([14577c1](https://github.com/ccam80/cubie/commit/14577c1e237791f2997ab53ffc842b5325763766))


### Bug Fixes

* BaseArrayManager.py once again contains all of it's methods, after half of the class was sausage-fingered clean off. ([96ba1cb](https://github.com/ccam80/cubie/commit/96ba1cb3d1fb6bb2bb0e6786ddb0e8dabd8512ae))
* buggy regex removed from pyproject ([bc5606a](https://github.com/ccam80/cubie/commit/bc5606aa67ce84994a5009f0b26dbd5d8e45e603))
* ignore generated python files ([14577c1](https://github.com/ccam80/cubie/commit/14577c1e237791f2997ab53ffc842b5325763766))
* implement four-byte padding to reduce shared memory conflicts ([944066e](https://github.com/ccam80/cubie/commit/944066ee2ced3561b9607cbfcbfb48988f8256b9))
* jacobian product codegen now does a simple dead-code removal sweep at the expression level ([5a46559](https://github.com/ccam80/cubie/commit/5a46559f273a8e4408fbc777ef25c55dbfc06be8))
* metric function compilation now deferred until fn requested. ([fc42de7](https://github.com/ccam80/cubie/commit/fc42de7d438efb7968b4f4663016f9aae7cd82aa))
* sympy piecewise printing patched in subclass ([ca20f84](https://github.com/ccam80/cubie/commit/ca20f84061dda44c44f0ccac451d1aaa128ca4bc))
* SystemValues now interpreting sympy symbols correctly ([0549bea](https://github.com/ccam80/cubie/commit/0549beac411c21dd8a54167f517d8846a8500458))
* SystemValues, BaseODE now have comprehensible repros ([0549bea](https://github.com/ccam80/cubie/commit/0549beac411c21dd8a54167f517d8846a8500458))


### Documentation

* batchsolving module now has all docstrings in numpydocs-friendly format ([8e6e8cf](https://github.com/ccam80/cubie/commit/8e6e8cf415b86017137cd00421e14de58168e019))
* conf.py path reverted for Sphinx build ([5383de6](https://github.com/ccam80/cubie/commit/5383de6e551efde3ecf965efc50210608f136f12))
* docs updated and thinned to match structure ([9df6978](https://github.com/ccam80/cubie/commit/9df69786697ccf35f29f528a828a057a67d302ab))
* first-pass narrative docs added ([14577c1](https://github.com/ccam80/cubie/commit/14577c1e237791f2997ab53ffc842b5325763766))
* get sphinx-build working again and ReadTheDocs themed ([6df9a82](https://github.com/ccam80/cubie/commit/6df9a82fefa6307a9ef981ebe70ee2bc0c72161f))
* insert google verification tag, cross-link repo and docs ([0d61cd2](https://github.com/ccam80/cubie/commit/0d61cd2f93d1e79d2dc823a9830387d863cf8abb))
* integrators section docstrings brought in line with numpydocs format ([8b22790](https://github.com/ccam80/cubie/commit/8b22790c912b50f0681b7fb6d7eda925e87be64b))
* memory section docstrings brought into numpy format ([d495551](https://github.com/ccam80/cubie/commit/d495551cce6f0a150da76b6000f87ce512e4345a))
* output_functions section docstrings brought into numpy format ([4e7b2c0](https://github.com/ccam80/cubie/commit/4e7b2c06d1ec8b0ee873d97e0e30e0f4b42a849c))
* pypi version badge added ([8b22790](https://github.com/ccam80/cubie/commit/8b22790c912b50f0681b7fb6d7eda925e87be64b))
* readme now has code coverage badge ([8e6e8cf](https://github.com/ccam80/cubie/commit/8e6e8cf415b86017137cd00421e14de58168e019))
* systemmodels section docstrings brought into numpy format ([9df6978](https://github.com/ccam80/cubie/commit/9df69786697ccf35f29f528a828a057a67d302ab))


### Miscellaneous Chores

* re-release 0.0.3 ([359abe3](https://github.com/ccam80/cubie/commit/359abe33cb49f47f6b4dcaacf3c630d82d95c69c))
* release 0.0.3 ([684ef91](https://github.com/ccam80/cubie/commit/684ef91d144178828b2ec5fe7bf3addd58b625a9))

## [0.0.2](https://github.com/ccam80/cubie/compare/v0.0.1...v0.0.2) (2025-08-18)


### Features

* BaseArrayManager class added to unify approach to allocating/deallocating device arrays through the memory manager ([416e363](https://github.com/ccam80/cubie/commit/416e3632085eaac507e596c22bafb34c22f107a2))
* BatchConfigurator now accepts extra user input types to match usage ([a345679](https://github.com/ccam80/cubie/commit/a3456797cbd37eb47a3d75b10b9d870fe6b22203))
* BatchInputArrays and BatchOutputArrays now subclass BaseArrayManager ([808695e](https://github.com/ccam80/cubie/commit/808695ed21eee0a777178babb2f181f6fc85afef))
* Memory Manager extended to queue and process requests from multiple objects ([67e66bf](https://github.com/ccam80/cubie/commit/67e66bfe82fc02258afcbb94d342621505632074))
* MemoryManager implemented ([3387142](https://github.com/ccam80/cubie/commit/3387142c7925c3ee76e60175bb1cddd0a1b87ce7))
* Solver interface now set up for real people to use (I think) ([2b05c1e](https://github.com/ccam80/cubie/commit/2b05c1eb1a590f7fcd99f4aa0d33171ea4f5de37))
* UserArrays now handles delivery from device arrays to end user for inspection ([9d93a7a](https://github.com/ccam80/cubie/commit/9d93a7a560500519b9cd500b70d082610fc9efc4))


### Bug Fixes

* BatchSolverKernel now uses blocksize to calculate dynamic shared memory correctly, and reduces blocksize if it's &gt; limit ([5d8263f](https://github.com/ccam80/cubie/commit/5d8263fd9aae11f6006f53bd349c44d169341405))
* bug in previosu commit: BatchSolverKernel now uses blocksize to calculate dynamic shared memory correctly, and reduces blocksize if it's &gt; limit ([afda7cb](https://github.com/ccam80/cubie/commit/afda7cbe7d72e2d3a0e0134dca1be49ca81dbe08))
* fix circular imports introduced in 636b5e3 ([2865160](https://github.com/ccam80/cubie/commit/2865160cc31fb7589cc75148dfbeaad88d78af37))
* force odd shared memory size per run to minimise clashes. ([fc12c76](https://github.com/ccam80/cubie/commit/fc12c768dbc7bd27bbd90b469dae8258c09149d9))
* forward declarations no longer cause circular imports ([e3841ff](https://github.com/ccam80/cubie/commit/e3841ffabdd745e1a5892417b19b7524a8d65f0a)), closes [#73](https://github.com/ccam80/cubie/issues/73)
* Newly-initialised memory manager no longer breaks in CUDA sim ([db99862](https://github.com/ccam80/cubie/commit/db99862da766224b1f1588fd15f41cb524dc40bc))
* output config flags now treated as derived quantities instead of attributes ([9d93a7a](https://github.com/ccam80/cubie/commit/9d93a7a560500519b9cd500b70d082610fc9efc4))
* pyproject.toml now points at correct license and readme files for building. ([f7bd7f0](https://github.com/ccam80/cubie/commit/f7bd7f00079c4e804cd92e9bfdcf18bd2089c42f))
* pyproject.toml now points at correct license file (but for real this time) ([93591ed](https://github.com/ccam80/cubie/commit/93591ed9c3c4465ff812b0402ec58c993d7cd63c))
* SolverKernel now solves and summarises accurately. ([879da10](https://github.com/ccam80/cubie/commit/879da1033c7e5318ea2633213d55501f9cb186f5))
* SolverKernel tests made CUDA-simulator-friendly ([ad702bb](https://github.com/ccam80/cubie/commit/ad702bb3a37f95e831723fc270978a5876316829))
* UserArrays now SolveResult, and works with array managers to produce a sensible output ([994cd7c](https://github.com/ccam80/cubie/commit/994cd7ccc6bcd53c1e02b73f7397d5fe0c4e1d65))
* UserArrays.as_numpy now returns copies rather than mappedarrays ([01af17d](https://github.com/ccam80/cubie/commit/01af17d493ceb74ac9b05468b929a7f6290cdd19))


### Documentation

* Docs don't mention CuMC anymore, and we shall never speak of it again ([fff2e00](https://github.com/ccam80/cubie/commit/fff2e0088168ac8106c50ff3d479cc49ccad4864))
* Docs updated to reflect cubie refactor ([205e748](https://github.com/ccam80/cubie/commit/205e7489360c458818904c4e2bd0860639d54acf))
* Properties which expose lower-level attributes are now docstringed as such ([a97d368](https://github.com/ccam80/cubie/commit/a97d368500ae618ae35ca0cce40ce54ebc98380b))


### Miscellaneous Chores

* release 0.0.2 ([60b3cd1](https://github.com/ccam80/cubie/commit/60b3cd1887078b2bbcb92ff0160bfc1222fddbd6))

## 0.0.1 (2025-08-01)


### Features

* BatchConfigurator implemented and passing tests. ([0bc00f2](https://github.com/ccam80/smc/commit/0bc00f22422238ac8fcfe68fc07691de07757945))
* BatchConfigurator implemented and passing tests. ([f4859e4](https://github.com/ccam80/smc/commit/f4859e448893f6a5738297cb9bca2a1987248fe9))
* BatchConfigurator implemented and passing tests. ([edf60fa](https://github.com/ccam80/smc/commit/edf60faf40febc0b652a7f18d8c79f7c72a8fe0a))
* first attempt at a batch configurator ([4d723ed](https://github.com/ccam80/smc/commit/4d723edb192159b97d999bd2e011d1c65d108b05))
* Initial dev version release. Not all features documented in changelog; some commit messages poorly named. Expect better changelogs in subsequent releases ([1a7691e](https://github.com/ccam80/smc/commit/1a7691e31352cd1e6ff66538c96caa223e2f364d))


### Bug Fixes

* Add a nozeros toggle to array sizes ([1300f3f](https://github.com/ccam80/smc/commit/1300f3f7285062bcda5ab7278ed0ab3cd2daa9de))
* **ci:** Fix release-please label permissions ([fa5d77f](https://github.com/ccam80/smc/commit/fa5d77f2292555475ec6e638cf11de0ccd90e45e))
* complete GPU tests on release tag ([f622a03](https://github.com/ccam80/smc/commit/f622a03e0445c6ae9088153c7a0811190eb81089))
* complete GPU tests on release tag ([f622a03](https://github.com/ccam80/smc/commit/f622a03e0445c6ae9088153c7a0811190eb81089))
* correct import error in ODEData.py ([cdd3144](https://github.com/ccam80/smc/commit/cdd3144e6a715284d6fc76f6670351f3f726607b))
* Corrected ancient typo in SystemValues that made all parameter setting invalid, confirmed test coverage ([c05f532](https://github.com/ccam80/smc/commit/c05f5326f4187c95c8c64cd0de68afa9ff193d96))
* fix circular dependency, improve arraysizes interface ([98ecf54](https://github.com/ccam80/smc/commit/98ecf54d0751491634d3fad39aa3a5331cdcf28f))
* **git:** close issue [#41](https://github.com/ccam80/smc/issues/41) ([da83993](https://github.com/ccam80/smc/commit/da83993e0997194fa3c569ab451868c53fe5a3c4))
* **git:** remove local junk from repo [#41](https://github.com/ccam80/smc/issues/41) ([a3cb122](https://github.com/ccam80/smc/commit/a3cb1224cb61a2da412b544512dcf5a5f9acf116))
* Implemented adapters for array size and allocation classes ([7a4da6a](https://github.com/ccam80/smc/commit/7a4da6a4f6a3c586e2024c29c5a924c4328c006a))
* Improve adapters and access to output_sizes objects through higher objects ([649e8cf](https://github.com/ccam80/smc/commit/649e8cf8e94e10dce28e67bcad303e46cb98efc0))
* Output array indices now gated by boolean flags to avoid memory access errors ([4f5628c](https://github.com/ccam80/smc/commit/4f5628ce3fad9e57c6a03bcde65a032bc7b27428))
* **OutputHandling:** switched indexing order in 2d arrays to match intended striding. ([86a1d2f](https://github.com/ccam80/smc/commit/86a1d2f5f7622b51f302c56e330b51a7293c542f))
* Plumbing now works between lower-level modules ([d79cfdb](https://github.com/ccam80/smc/commit/d79cfdb0817f6c1e8aa9977c2a6a5af6854e4b7a))
* remove bug introduced in time-saving ([ce96d23](https://github.com/ccam80/smc/commit/ce96d236b6cf5b8c7d87f3a1a4cc24a50cfa2ad4))
* remove doto [#19](https://github.com/ccam80/smc/issues/19) ([d3457bd](https://github.com/ccam80/smc/commit/d3457bd23d17f030840ee690601151e289ccf152))
* Remove todos to close issues ([f2e7638](https://github.com/ccam80/smc/commit/f2e76384e407c1d7ac8c2a36e80916b9c79bd3b3))
* Set buffer height methods in output_functions to properties to follow the convention for the rest of the module ([3a5c387](https://github.com/ccam80/smc/commit/3a5c387e1414b56b2cfb0d4e1fd0e19ba82769bd))
* SingleIntegratorRun and children now re-validate timing after an update ([0baad64](https://github.com/ccam80/smc/commit/0baad6433925af427a9f3b60cee4df71121c7e4d))
* Swat rename bug in summary_metrics testing ([cffd94d](https://github.com/ccam80/smc/commit/cffd94df0da154d8d85b11026b9fb53a59826bb8))


### Miscellaneous Chores

* release 0.0.1 ([73c85c5](https://github.com/ccam80/smc/commit/73c85c50c3c2991b2ae1e9237caf1aa2fd15316b))
