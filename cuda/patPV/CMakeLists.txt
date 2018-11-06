file(GLOB patPV "src/patPV.cu")
include_directories(include)
include_directories(../../main/include)
include_directories(../velo/common/include)
include_directories(../veloUT/common/include)
include_directories(../event_model/velo/include)
include_directories(../../stream/gear/include)
include_directories(../event_model/velo/include)
include_directories(../event_model/common/include)
include_directories(../utils/sorting/include)
include_directories(../utils/prefix_sum/include)

cuda_add_library(PatPV STATIC
  ${patPV} src/getSeeds.cu src/fitSeeds.cu
)
