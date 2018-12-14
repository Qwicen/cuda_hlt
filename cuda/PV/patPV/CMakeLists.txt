include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(../common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/veloUT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/prefix_sum/include)
include_directories(../beamlinePV/include)

cuda_add_library(PatPV STATIC
  ${patPV} src/GetSeeds.cu src/FitSeeds.cu
)
