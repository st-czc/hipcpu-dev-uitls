# Local stub for CPU-only HIP test builds.
# unit/rtc, contract/* and multiproc link hiprtc::hiprtc. This workflow builds
# only kernel/ tests, so an empty INTERFACE target satisfies the generate step.
if(NOT TARGET hiprtc::hiprtc)
  add_library(hiprtc::hiprtc INTERFACE IMPORTED)
endif()
