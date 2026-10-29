#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DIR="$PWD"
ROOT="$(cd ../../.. && pwd)"
CLANGXX="$ROOT/install/lib/llvm/bin/clang++"
LLVM_DIS="$ROOT/install/lib/llvm/bin/llvm-dis"
mkdir -p build ir

# Build one case.  Compile with --save-temps inside a throwaway dir that is
# removed when the function returns, so clang's intermediates never touch this
# directory.  The pre-link device bitcode (*spvcpu.tmp.bc) is disassembled to
# ir/<name>.ll for every case, including the ones that fail to link.
compile() {
  local name="$1"
  local tmp
  tmp="$(mktemp -d "$DIR/build/.tmp.XXXXXX")"
  trap 'rm -rf "$tmp"' RETURN
  (
    cd "$tmp"
    "$CLANGXX" -x hip --offload-arch=spvcpu -Wno-unused-value --save-temps \
      "$DIR/$name.hip" -o "$name" 2>"$tmp/err.log" || true
  )
  local bc
  bc="$(ls "$tmp"/"$name"*spvcpu.tmp.bc 2>/dev/null | head -1)"
  [ -n "$bc" ] && "$LLVM_DIS" "$bc" -o "ir/$name.ll"
  if [ -f "$tmp/$name" ]; then
    cp "$tmp/$name" "build/$name"
    printf '%-24s PASS      ir/%s.ll\n' "$name" "$name"
  else
    printf '%-24s LINK-FAIL ir/%s.ll\n' "$name" "$name"
  fi
}

# HIP __half2 (struct) — all these work:
compile hip_half2_hfma2      # __hfma2 -> canonical <2 x half> intrinsic
compile hip_half2_mul_add    # a*b+c (operators) -> no raw llvm.fma.v2f16
compile fma_builtin          # __builtin_elementwise_fma -> canonical intrinsic

# Raw llvm.fma.* via __asm label -> host ABI coerces half types:
compile fp16_fma             # bare <2 x half> -> i32 -> LINK-FAIL
compile hip_half2_raw_fma    # HIP __half2 -> i32 -> LINK-FAIL
compile fp16_fma_half4       # bare <4 x half> -> double -> links but WRONG
