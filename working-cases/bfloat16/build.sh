#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
CLANGXX="$ROOT/install/lib/llvm/bin/clang++"
mkdir -p build

for name in bfloat16_axpy hip_bf16_axpy; do
  echo "=== $name ==="
  if "$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value "$name.hip" -o "build/$name" 2>"build/$name.log"; then
    "./build/$name"
  else
    echo "COMPILE-FAIL:"
    tail -8 "build/$name.log"
  fi
done
