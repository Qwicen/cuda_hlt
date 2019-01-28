file(GLOB associate_srcs "src/*.cu")
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/associate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/float_operations/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/beamlinePV/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)

add_library(Associate STATIC
  ${associate_srcs}
)
set_property(TARGET Associate PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)
