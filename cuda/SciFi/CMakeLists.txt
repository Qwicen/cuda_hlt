file(GLOB scifi_tracking "PrForward/src/*cu")
file(GLOB scifi_common "common/src/*cu")
file(GLOB scifi_preprocessing "preprocessing/src/*cu")

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
include_directories(preprocessing/include)
include_directories(../utils/sorting/include)
include_directories(../../main/include)
include_directories(../../stream/handlers/include)

cuda_add_library(SciFi STATIC
  ${scifi_common}
  ${scifi_preprocessing}
  ${scifi_tracking}
)
