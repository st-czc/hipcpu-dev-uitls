# sycl_frexp

Two SYCL Native CPU cases, same kernel shape, different `frexp` spelling.

The HIP failure comes from `llvm.frexp.f32.i32 -> {float, i32}`, a literal
struct return. Only the spelling that reaches that intrinsic reproduces it.

| file | device spelling | lowered to | result |
| --- | --- | --- | --- |
| `sycl_builtin_frexp.cpp` | `sycl::frexp` | `__spirv_ocl_frexp` bit manipulation, no intrinsic | builds and passes |
| `sycl_clang_builtin_frexp.cpp` | `__builtin_frexpf` | `llvm.frexp.f32.i32` | vecz aborts during compilation |

## Run

```sh
./build_sycl_builtin.sh          # sycl::frexp
./build_sycl_clang_builtin.sh    # __builtin_frexpf

LD_LIBRARY_PATH=../../../compiler/hipcpu-artifacts/intel-llvm/build/install/lib \
  ./build/sycl_builtin_frexp
```

## Expected

`sycl_builtin_frexp`: prints `sycl_builtin_frexp: PASS (0/256 mismatch)`.

`sycl_clang_builtin_frexp`: the build aborts in `vecz::RunVeczPass` /
`vecz::Packetizer::Impl::packetizeBinaryOp` with

```
Assertion `S1->getType() == S2->getType() && "Cannot create binary operator
with two operands of differing type!"' failed.
```

That is the same defect as the HIP `spvcpu` failure on `math/root_funcs`
(`hypot`/`norm` kernels, which reach `llvm.frexp` through
`BUILTIN_FREXP_EXP_F32/F64`). With the vecz fix in
`packetization_helpers.cpp` this case should build and print
`sycl_clang_builtin_frexp: PASS (0/256 mismatch)`.
