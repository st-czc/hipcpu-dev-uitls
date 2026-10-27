#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
CLANGXX="$ROOT/install/lib/llvm/bin/clang++"
mkdir -p build
"$CLANGXX" -O3 -x hip --offload-arch=spvcpu -Wno-unused-value hypot.hip -o build/hypot
