#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Compile hip_gemm.cpp against the hipblas-cpu build (libhipblas + librocblas)
# and the HIP-CPU toolchain (HIP headers + libamdhip64).

ROOT="$(cd ../.. && pwd)"
TOOLCHAIN="$ROOT/install"
PREFIX="$PWD/install" # hipblas-cpu install tree
CLANGXX="$TOOLCHAIN/lib/llvm/bin/clang++"

mkdir -p out

"$CLANGXX" -x hip --offload-arch=spvcpu \
  -I "$PREFIX/include" \
  -I "$TOOLCHAIN/include" \
  hip_gemm.cpp \
  -L "$PREFIX/lib" -lhipblas -lrocblas \
  -L "$TOOLCHAIN/lib" \
  -o out/hip_gemm

echo "built out/hip_gemm"
echo "run with: LD_LIBRARY_PATH=$PREFIX/lib:$TOOLCHAIN/lib out/hip_gemm"
