# Changelog

## [Unreleased]

### Performance

* Last-step caching optimization for Runge-Kutta tableaus where final stage weights match a row in the coupling matrix
* Tableau properties `b_matches_a_row` and `b_hat_matches_a_row` for automatic detection of optimization opportunities
* Eliminates redundant accumulation in RODAS4P, RODAS5P, and RadauIIA5 methods through compile-time branch optimization
* Transparent to users (no API changes required)

## [0.0.5](https://github.com/ccam80/cubie/compare/v0.0.4...v0.0.5) (2025-11-04)


### Features

* "Instrumented" device steps added for diagnostics ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* additional ERK, DIRK, Rosenbrock tableaus added ([ed866d7](https://github.com/ccam80/cubie/commit/ed866d7153e4255136a8155f55cb92826f319d6b))
* Algorithms now set their own step control defaults ([#139](https://github.com/ccam80/cubie/issues/139)) ([2a0efed](https://github.com/ccam80/cubie/commit/2a0efedd7a02a10be5179f0ef84fe72ebfddba84))
* Array managers now support heterogeneous arrays within the same container ([0905422](https://github.com/ccam80/cubie/commit/0905422b8fbaf6c1b05f127d162a2f44bc40c53a))
* Compile-settings updates now won't force a rebuild if the value hasn't changed. ([5a9e281](https://github.com/ccam80/cubie/commit/5a9e281bb16455c37ca70c85d56831d50f809fc7))
* DIRK and DIRKTableaus added ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* Explicit and Diagonally-Implicit Runge Kutta algorithms added ([#151](https://github.com/ccam80/cubie/issues/151)) ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* Explicit RK and Tableau added, closes [#83](https://github.com/ccam80/cubie/issues/83) ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* Fully-Implicit Runge-Kutta (FIRK) Methods implemented ([#162](https://github.com/ccam80/cubie/issues/162)) ([ed866d7](https://github.com/ccam80/cubie/commit/ed866d7153e4255136a8155f55cb92826f319d6b))
* Generic Butcher Tableau implemented ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* Generic Rosenbrock-W methods added ([#148](https://github.com/ccam80/cubie/issues/148)) ([6abfed2](https://github.com/ccam80/cubie/commit/6abfed2c0d051b597b11ea3601a4806eecfc7aac))
* minimal FSAL caching added to DIRK, ERK, Rosenbrock ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* N-stage flattened linear operators, preconditioners, nonlinear residual codegen added ([ed866d7](https://github.com/ccam80/cubie/commit/ed866d7153e4255136a8155f55cb92826f319d6b))
* Parser now processes indexed arrays as variables ([#152](https://github.com/ccam80/cubie/issues/152)) ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))
* rodasnp methods, dop853, tsit5, vern7 ([87b53bc](https://github.com/ccam80/cubie/commit/87b53bc6b87227db223989a5576e17846448b685))
* Rosenbrock methods now 100% more rosenbrock ([#157](https://github.com/ccam80/cubie/issues/157)) ([35641e6](https://github.com/ccam80/cubie/commit/35641e6ab24f76193a03e29c01c64822298dba63))
* status codes now aggregated by batchSolverKernel ([0905422](https://github.com/ccam80/cubie/commit/0905422b8fbaf6c1b05f127d162a2f44bc40c53a))
* Steps and step controllers now have a unified argument-filtering factory ([2a0efed](https://github.com/ccam80/cubie/commit/2a0efedd7a02a10be5179f0ef84fe72ebfddba84))
* Tableau libraries and tableau resolvers/getters added ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* There are now auxiliary-cached jacobian functions for reusing some computational work. ([6abfed2](https://github.com/ccam80/cubie/commit/6abfed2c0d051b597b11ea3601a4806eecfc7aac))
* Third and fourth order SDIRK tableaus added ([91ee1e9](https://github.com/ccam80/cubie/commit/91ee1e967d343b38ed521f52012ca4a414ecdbd2))
* Time-derivative helpers added for symbolic functions and interpolated arrays ([35641e6](https://github.com/ccam80/cubie/commit/35641e6ab24f76193a03e29c01c64822298dba63))
* Very rough caching of jvp nodes implemented for rosenbrock solvers. ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))
* working arrays and quantities in algorithms and solvers now draw from a memory "pool", allowing easier reuse ([b0f5b6f](https://github.com/ccam80/cubie/commit/b0f5b6f879d82caa700c853042a860736da02714))


### Bug Fixes

* add error to sdirk 4, correct controllers for loop tests ([47a6cb1](https://github.com/ccam80/cubie/commit/47a6cb1fb20a53b55c38711668abd3b98fa33ba5))
* Added (non-CI) testing for DIRK loops added. ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))
* batchgridbuilder class now has static wrappers for batchgridbuilder helper functions. ([74c08fa](https://github.com/ccam80/cubie/commit/74c08fa0a9846c988399e9893a1acee3193a62ca))
* BatchsolverKernel's compile settings now features the compile-critical settings it always should have had. ([f383157](https://github.com/ccam80/cubie/commit/f383157a06180841071bf9df90d23c1081c907f2))
* Buffer footprints reduced by aliasing vectors with disjoint lifetimes ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* correct off-by-one error and datatype discrepancy in cpu driver evaluator ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* DIRK now accumulates rhs's and scales after stages, reducing round-off ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))
* faulty implementation of ode23s removed, dop853 tableau amended ([d5a784c](https://github.com/ccam80/cubie/commit/d5a784c366a22d81b4c12651b0536c231263098d))
* matplotlib now spelled correctly in pyproject.toml ([3773ff8](https://github.com/ccam80/cubie/commit/3773ff8c0a73fca8986794d82289f2d0e3a3c169))
* Meaty loop tests confined to test_ode_loop.py ([7fe5677](https://github.com/ccam80/cubie/commit/7fe5677be087bff94a75e1c3e9843938f05c139a))
* numerous numerical errors amended after instrumenting steps ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* precision type hints now correct; no more pesky yellow lines. ([2a0efed](https://github.com/ccam80/cubie/commit/2a0efedd7a02a10be5179f0ef84fe72ebfddba84))
* Rosenbrock buffer footprint reduced by 2n ([ed866d7](https://github.com/ccam80/cubie/commit/ed866d7153e4255136a8155f55cb92826f319d6b))
* settings passing from solver now less cramped and hopefully more robust ([2a0efed](https://github.com/ccam80/cubie/commit/2a0efedd7a02a10be5179f0ef84fe72ebfddba84))
* Step controllers no longer mutate error vector ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))


### Documentation

* Add autodocs subpages for manual project structure docs ([c2c0d7c](https://github.com/ccam80/cubie/commit/c2c0d7cd0c9173cccd38d84d3969d6e66406e4b4))
* add copilot instructions ([#161](https://github.com/ccam80/cubie/issues/161)) ([2bfe6f2](https://github.com/ccam80/cubie/commit/2bfe6f2a28e9e20ed48e855435901ea8145c5a78))
* added "buffer map" comments to generic algorithms to demystify aliasing ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* autodocs param lists now format one line per param ([cb3d74b](https://github.com/ccam80/cubie/commit/cb3d74bf4a578705abf31c797e0662aea3202879))
* Batchsolving module source files now better documented in numpydocs format ([97a0ab8](https://github.com/ccam80/cubie/commit/97a0ab8a9510ea1e46f1aa6cc3a16f0a51ee9ae8))
* de-computer some language in api reference, rejig indexes ([b558994](https://github.com/ccam80/cubie/commit/b5589943dbec791ee41b71c012b148e856352e83))
* increase index depth, force one-param-per-line printing ([b822f4c](https://github.com/ccam80/cubie/commit/b822f4c402f3321c6a399163e5e0497433493a8e))
* more docs organisation ([5631989](https://github.com/ccam80/cubie/commit/563198977c99f688549152feb798335a5eb3bf36))
* more docs organisation ([cd98043](https://github.com/ccam80/cubie/commit/cd98043d02c663ddfe2f3df981cd218ddbc99c95))
* refactor api structure; remove autosummary, implement manual docs ([881ce4a](https://github.com/ccam80/cubie/commit/881ce4ac81c8405f0a8cffe00788de7c987a3dce))
* Refs in "getting started" are now actually refs not highlighted garbage. ([6abfed2](https://github.com/ccam80/cubie/commit/6abfed2c0d051b597b11ea3601a4806eecfc7aac))
* top-level batchsolving docs added ([6cfddfa](https://github.com/ccam80/cubie/commit/6cfddfa8d86d656be4b79b34d068290680e60046))


### Miscellaneous Chores

* release 0.0.5 ([93a7255](https://github.com/ccam80/cubie/commit/93a72554cf461daf58da86b5bbc46bebb197d4d9))

## [0.0.4](https://github.com/ccam80/cubie/compare/v0.0.3...v0.0.4) (2025-10-05)


### Features

* Adaptive step-size controllers added : i (traditional), pi, pid, gustafsson acceleration ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Adaptive time-step controllers now have a programmable dead-band. ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* AGENTS.md extended and partially updated to summarise entire project for ai agents ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* arbitrary drivers can now be looped or clamped to zero (smoothly). ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* Backwards Euler implicit fixed-step method added (with and without predictor-corrector mechanism), closes [#114](https://github.com/ccam80/cubie/issues/114). ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Codegen for residual functions, jvps, and various solver helper functions created ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Crank-Nicolson trapezoidal adaptive-step algorithm implemented. ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* cuda simulation patches consolidated for cuda-free environment tests. ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Forcing (driver) terms now adaptive-step friendly ([#132](https://github.com/ccam80/cubie/issues/132)) ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* matrix free solvers added ([1dffd94](https://github.com/ccam80/cubie/commit/1dffd94ae965a72c3f42ea2be09761cedcd10582))
* Nonlinear Newton-Krylov iterative solver with preconditiong added for implicit methods, closes [#101](https://github.com/ccam80/cubie/issues/101), [#102](https://github.com/ccam80/cubie/issues/102), [#111](https://github.com/ccam80/cubie/issues/111) ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* plotting added to driver interpolator - keep an eye on what the machine is doing. ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* shared memory padding now closer to optimal (nothing to be done about 64-bit values), closes [#86](https://github.com/ccam80/cubie/issues/86). ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))


### Bug Fixes

* Array "chunking" logic now respects "unchunkable" arrays in allocation ([53141df](https://github.com/ccam80/cubie/commit/53141dfa72a7568813f85c86ef5d9b6f682db856))
* CPU test step controllers now raise dt_too_small errors ([19af8ff](https://github.com/ccam80/cubie/commit/19af8ff620419155e4c6e8d3512aa1224421d3ad))
* Crank-Nicolson and adaptive controllers now 50% more idiot-error free. ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* CUDAFactory now updates underscored config variables as intended ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Many edits to precision settings and flow throughout system (making it work) ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Observables calculation now occurs in sync with state for adaptive-step loops ([#130](https://github.com/ccam80/cubie/issues/130)) ([19af8ff](https://github.com/ccam80/cubie/commit/19af8ff620419155e4c6e8d3512aa1224421d3ad))
* pypi version tag trigger now using correct syntax ([3b3caa1](https://github.com/ccam80/cubie/commit/3b3caa1d1e434edc61dc86174d211ac83444383f))
* some sign confusion and bogus gains corrected in step controllers ([19af8ff](https://github.com/ccam80/cubie/commit/19af8ff620419155e4c6e8d3512aa1224421d3ad))


### Documentation

* Batch arrays re-docstringed in keeping with rest of library. ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* fix circular import for docs building ([64d8933](https://github.com/ccam80/cubie/commit/64d8933e965b52f80664f2f5ddefc5ab03d96077))
* manual docs added for integrators, memory modules. submodules of systems, outputhandling  documented. ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* step controller comparison added to docs/examples ([19af8ff](https://github.com/ccam80/cubie/commit/19af8ff620419155e4c6e8d3512aa1224421d3ad))
* top-level summaries of odesystems and outputhandling ([a9cb4c0](https://github.com/ccam80/cubie/commit/a9cb4c08383aa452d64a5cf63bec9b3cb86d5b75))
* update docstrings in outputhandling and odesystems root directorys ([87bb133](https://github.com/ccam80/cubie/commit/87bb133643a8b8b14ff6731469e2ef5e7eded3ee))


### Miscellaneous Chores

* release 0.04 ([ed5045a](https://github.com/ccam80/cubie/commit/ed5045ad6e5677cc1f4543d0c1a8673c76028fe2))

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
* odesystems section docstrings brought into numpy format ([9df6978](https://github.com/ccam80/cubie/commit/9df69786697ccf35f29f528a828a057a67d302ab))


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
