file(GLOB scifi_tracking "PrForward/src/*cu")

include_directories(../../checker/tracking/include)
include_directories(../../main/include)
include_directories(PrForward/include)
include_directories(../UT/common/include)
include_directories(../UT/PrVeloUT/include)
include_directories(../UT/UTDecoding/include)
include_directories(common/include)
include_directories(../velo/common/include)
include_directories(../event_model/common/include)
include_directories(../event_model/velo/include)

cuda_add_library(SciFi STATIC
  ${scifi_tracking}
)
