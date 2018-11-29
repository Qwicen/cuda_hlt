file(GLOB global_event_cuts "src/*cu")

include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(include)

cuda_add_library(GlobalEventCuts STATIC
  ${global_event_cuts}
)