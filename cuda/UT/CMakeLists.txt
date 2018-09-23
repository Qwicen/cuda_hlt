file(GLOB UT_decoding "UTDecoding/src/*cu")
file(GLOB UT_sorting "sorting/src/*cu")
file(GLOB veloUT_tracking "PrVeloUT/src/*cu")
file(GLOB compassUT_tracking "compassUT/src/*cu")

include_directories(../../main/include)
include_directories(../velo/common/include)
include_directories(../utils/sorting/include)
include_directories(../event_model/velo/include)
include_directories(sorting/include)
include_directories(common/include)
include_directories(UTDecoding/include)
include_directories(PrVeloUT/include)
include_directories(compassUT/include)

cuda_add_library(UT STATIC
  ${UT_decoding}
  ${UT_sorting}
  ${veloUT_tracking}
  ${compassUT_tracking}
)
