#!/bin/bash
# Build and run HIP unit tests against the spvcpu toolchain, recording results.
set -euo pipefail

trap 'exit 130' INT
trap 'exit 143' TERM

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$THIS_DIR/../.." && pwd)"
INSTALL="$ROOT/install"
HIP_TESTS="$ROOT/rocm-systems/projects/hip-tests"
BUILD="$HIP_TESTS/build"
RESULTS="$THIS_DIR/results"
STAMP="$(date +%Y%m%d-%H%M%S)"
RECORDS_DIR="$RESULTS/000_records/$STAMP"

CLEAN=0
CASE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
  --clean) CLEAN=1 ;;
  -c | --case)
    CASE="${2:?missing argument for $1}"
    shift
    ;;
  -h | --help)
    echo "usage: $0 [--clean] [-c|--case <grep-filter>]"
    exit 0
    ;;
  *)
    echo "unknown option: $1" >&2
    echo "usage: $0 [--clean] [-c|--case <grep-filter>]" >&2
    exit 2
    ;;
  esac
  shift
done

export PATH="$INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL/lib:$INSTALL/lib/llvm/lib:$INSTALL/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"

if [ "$CLEAN" -eq 1 ]; then
  rm -rf "$BUILD"
fi

if [ ! -d "$BUILD" ]; then
  cmake -S "$HIP_TESTS/catch" -B "$BUILD" -GNinja \
    -DHIP_PLATFORM=amd \
    -DCMAKE_HIP_ARCHITECTURES=spvcpu \
    -DOFFLOAD_ARCH_STR="--offload-arch=spvcpu" \
    -DSTANDALONE_TESTS=ON \
    -DBUILD_CONTRACT_TESTS=OFF \
    -DCMAKE_PREFIX_PATH="$INSTALL" \
    -DCMAKE_INSTALL_PREFIX="$BUILD/install" \
    -Dhiprtc_DIR="$THIS_DIR/.cmake-stubs" \
    -Dhsa-runtime64_DIR="$THIS_DIR/.cmake-stubs" \
    -DCMAKE_C_COMPILER="$INSTALL/lib/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="$INSTALL/lib/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER="$INSTALL/lib/llvm/bin/clang++"
fi

TARGETS="$(ninja -C "$BUILD" -t targets all | grep '^catch_tests/unit/' | grep HIP_EXECUTABLE_LINKER)"
if [ -n "$CASE" ]; then
  TARGETS="$(printf '%s\n' "$TARGETS" | grep "$CASE" || true)"
fi

if [ -z "$TARGETS" ]; then
  echo "no tests match" >&2
  exit 1
fi

mkdir -p "$RESULTS"
SUCCESS="$RESULTS/success.txt"
TODO="$RESULTS/todo.csv"
: >"$SUCCESS"
: >"$TODO"

PASS_COUNT=0
FAIL_COUNT=0
while IFS= read -r LINE; do
  [ -n "$LINE" ] || continue
  REL="${LINE%%: *}"
  REL="${REL#catch_tests/unit/}"
  [ -f "$HIP_TESTS/catch/unit/$REL.cc" ] || continue

  CASE_DIR="$RESULTS/$REL"
  mkdir -p "$CASE_DIR"

  if ninja -C "$BUILD" "catch_tests/unit/$REL" &>"$CASE_DIR/build.log"; then
    if (
      timeout 120 "$BUILD/catch_tests/unit/$REL"
      exit $?
    ) &>"$CASE_DIR/run.log"; then
      echo "$REL" >>"$SUCCESS"
      echo "$REL --- passed"
      PASS_COUNT=$((PASS_COUNT + 1))
    else
      echo "$REL,run" >>"$TODO"
      echo "$REL --- failed run"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
  else
    echo "$REL,build" >>"$TODO"
    echo "$REL --- failed build"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done <<<"$TARGETS"

mkdir -p "$RECORDS_DIR"
cp "$SUCCESS" "$RECORDS_DIR/success.txt"
cp "$TODO" "$RECORDS_DIR/todo.csv"

echo "passed=$PASS_COUNT failed=$FAIL_COUNT"
echo "todo:    $TODO"
echo "success: $SUCCESS"
if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi
