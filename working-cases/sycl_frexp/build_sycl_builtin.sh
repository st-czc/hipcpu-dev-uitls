#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
SYCLCXX="$ROOT/compiler/hipcpu-artifacts/intel-llvm/build/install/bin/clang++"
mkdir -p build
cd build
"$SYCLCXX" -O2 -fsycl -fsycl-targets=native_cpu ../sycl_builtin_frexp.cpp -o sycl_builtin_frexp
