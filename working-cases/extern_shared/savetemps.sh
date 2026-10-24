#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
CLANGXX="$ROOT/install/lib/llvm/bin/clang++"
mkdir -p savetemps
cd savetemps
"$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value --save-temps ../extern_shared_one.hip -o extern_shared_one
"$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value --save-temps ../extern_shared_two.hip -o extern_shared_two

"$ROOT/install/lib/llvm/bin/llvm-dis" extern_shared_one.hip-hip-spvcpu-unknown-hipcpu.hipfb.spvcpu.spvcpu-merged.bc
"$ROOT/install/lib/llvm/bin/llvm-dis" extern_shared_two.hip-hip-spvcpu-unknown-hipcpu.hipfb.spvcpu.spvcpu-merged.bc
"$ROOT/install/lib/llvm/bin/llvm-dis" extern_shared_one.hip-hip-spvcpu-unknown-hipcpu.hipfb.spvcpu.spvcpu-lowered.bc
"$ROOT/install/lib/llvm/bin/llvm-dis" extern_shared_two.hip-hip-spvcpu-unknown-hipcpu.hipfb.spvcpu.spvcpu-lowered.bc
