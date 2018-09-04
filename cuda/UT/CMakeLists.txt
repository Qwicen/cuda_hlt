file(GLOB UTDecoding "UTDecoding/src/*cu")
include_directories(UTDecoding/include)
include_directories(../veloUT/common/include)
include_directories(../../main/include)
include_directories(../velo/common/include)

cuda_add_library(UT STATIC
  ${UTDecoding}
)
