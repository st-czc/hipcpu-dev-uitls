#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
CLANGXX="$ROOT/install/lib/llvm/bin/clang++"
mkdir -p build
"$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value math_builtins.hip -o build/math_builtins
