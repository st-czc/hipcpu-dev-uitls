#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
CLANGXX="$ROOT/install/lib/llvm/bin/clang++"

mkdir -p out/A out/B

(cd out/B && "$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value --save-temps ../../reqd_work_group_size.hip -o reqd_work_group_size)
(cd out/A && "$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value -DUSE_ATTR --save-temps ../../reqd_work_group_size.hip -o reqd_work_group_size)
