file(GLOB prefix_sum "prefix_sum/src/*cu")

include_directories(prefix_sum/include/)
include_directories(../velo/common/include)
include_directories(../event_model/velo/include)
include_directories(../event_model/common/include)
include_directories(../UT/PrVeloUT/include)
include_directories(../main/include)
include_directories(../../stream/handlers/include)

cuda_add_library(Utils STATIC
  ${prefix_sum}
)
