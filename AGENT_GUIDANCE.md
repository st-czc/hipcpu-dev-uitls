# HIP CPU Backend

This file is the entry point for HIP CPU backend work. All paths are relative
to the super-project root.

## Layout

- `compiler/llvm-project` — the main LLVM and Clang compiler. It is based on an
  upstream main commit.
- `compiler/amd-llvm` — a pruned AMD fork. It provides comgr and the libraries
  the HIP runtime uses.
- `compiler/hipcpu-passes` — the pass plugin for the SYCL Native CPU backend.
  It builds against `compiler/llvm-project`.
- `compiler/intel-llvm` — a full intel/llvm checkout. It is the reference
  source for `compiler/hipcpu-passes`.
- `rocm-systems/projects/clr` — the HIP runtime. The CPU backend code is in
  `clr/rocclr/device/cpu`.
- `hipcpu-dev-utils/hip-tests` — the HIP unit-test harness.

## Build

For a full build, run the scripts in this order.

```sh
./update_submodules.sh
./hipcpu_build.sh
```

For an incremental rebuild of one subproject, run both commands in order.

```sh
cmake --build build/<subproject>/build -- install
cmake --build build -- install
```

Example for the HIP runtime:

```sh
cmake --build build/core/clr/build -- install
cmake --build build -- install
```

## Test

```sh
cd hipcpu-dev-utils/hip-tests
./run.sh
```

Read `hipcpu-dev-utils/hip-tests/readme.md` for the flags.

## Bounds

- Use only the build entry points above.
- Do not run `build_tools/fetch_sources.py`. It resets all submodules and
  reapplies patches.
- The remote configuration (URLs, branches, pins) is user-maintained. Do not
  change it.
