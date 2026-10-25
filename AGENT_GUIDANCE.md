# HIP CPU Backend

This file is the entry point for HIP CPU backend work. All paths are relative
to the super-project root.

## Layout

- `compiler/llvm-project` — the main LLVM and Clang compiler. It is based on an
  upstream main commit.
- `compiler/amd-llvm` — a pruned checkout of the same repo (the AMD-side
  pieces: comgr, device-libs, hipcc, …). It builds against `compiler/llvm-project`.
- `compiler/hipcpu-passes` — the pass plugin for the SYCL Native CPU backend.
  It builds against `compiler/llvm-project`.
- `compiler/intel-llvm` — a full intel/llvm checkout. It is the reference
  source for `compiler/hipcpu-passes`.
- `rocm-systems/projects/clr` — the HIP runtime. The CPU backend code is in
  `clr/rocclr/device/cpu`.
- `hipcpu-dev-utils/hip-tests` — the HIP unit-test harness.

## Compilation flow

Device code becomes a device image in four steps:

1. Clang compiles device code for the `spvcpu` target to LLVM bitcode.
2. `llvm-link` merges that bitcode with the CPU support library
   `libsycl-nativecpu_utils.bc`.
3. `opt` runs the `sycl-native-cpu` pipeline from the `hipcpu-passes` plugin.
   The pipeline lowers the bitcode to host code.
4. Clang compiles the host code to a shared library (the device image).

## Build

For a full build, run the scripts in this order.

```sh
./update_submodules.sh
./hipcpu_build.sh
```

For an incremental rebuild, run:

```sh
cmake --build build -- install
```

The compiler is the one exception: after editing `compiler/llvm-project`,
`compiler/hipcpu-passes`, or `compiler/amd-llvm`, rebuild the `compiler/amd-llvm`
subproject first — it compiles clang/LLVM from `compiler/llvm-project`:

```sh
cmake --build build/compiler/amd-llvm/build -- install
cmake --build build -- install
```

## Test

```sh
hipcpu-dev-utils/hip-tests/run.sh --help
```

Read `hipcpu-dev-utils/hip-tests/readme.md` before you use the test harness.

## Bounds

- Use only the build entry points above.
- Do not run `build_tools/fetch_sources.py`. It resets all submodules and
  reapplies patches.
- The remote configuration (URLs, branches, pins) is user-maintained. Do not
  change it.
