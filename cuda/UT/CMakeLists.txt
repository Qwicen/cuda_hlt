file(GLOB UTDecoding "UTDecoding/src/*cu")
include_directories(UTDecoding/include)

cuda_add_library(UT STATIC
  ${UTDecoding}
)
