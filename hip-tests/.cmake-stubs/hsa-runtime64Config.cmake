# Local stub for CPU-only HIP test builds.
# hip-config-amd.cmake calls find_dependency(hsa-runtime64). hip::host /
# hip::device link hip::amdhip64 directly, so an empty config satisfies the
# find.
