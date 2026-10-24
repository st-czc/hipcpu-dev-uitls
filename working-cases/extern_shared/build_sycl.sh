#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
SYCLCXX="$ROOT/compiler/intel-llvm/build/bin/clang++"
mkdir -p build/sycl
cd build/sycl
"$SYCLCXX" -O0 -fsycl -fsycl-targets=native_cpu -fsycl-device-only --save-temps ../../extern_shared_two_sycl.cpp -o extern_shared_two_sycl

"$ROOT/install/lib/llvm/bin/llvm-dis" extern_shared_two_sycl -o extern_shared_two_sycl.ll
