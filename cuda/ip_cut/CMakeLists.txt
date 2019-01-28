file(GLOB ip_cut "src/*cu")

include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/associate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/beamlinePV/include)

add_library(IPCut STATIC
  ${ip_cut}
)
set_property(TARGET IPCut PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)
