file(GLOB UT_common "common/src/*cu")
file(GLOB UT_decoding "UTDecoding/src/*cu")
file(GLOB UT_tracking "PrVeloUT/src/*cu")

include_directories(../../main/include)
include_directories(../../stream/handlers/include)
include_directories(../velo/common/include)
include_directories(../event_model/velo/include)
include_directories(../event_model/common/include)
include_directories(../utils/sorting/include)
include_directories(sorting/include)
include_directories(common/include)
include_directories(UTDecoding/include)
include_directories(PrVeloUT/include)

cuda_add_library(UT STATIC
  ${UT_common}
  ${UT_decoding}
  ${UT_tracking}
)
