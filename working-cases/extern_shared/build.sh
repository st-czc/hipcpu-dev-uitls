#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
CLANGXX="$ROOT/install/lib/llvm/bin/clang++"
mkdir -p build
"$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value extern_shared_one.hip -o build/extern_shared_one
"$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value extern_shared_two.hip -o build/extern_shared_two
