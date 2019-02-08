file(GLOB scifi_tracking "PrForward/src/*cu")
file(GLOB scifi_common "common/src/*cu")
file(GLOB scifi_preprocessing "preprocessing/src/*cu")
file(GLOB scifi_consolidate "consolidate/src/*cu")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/UT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/tracking/include)
include_directories(PrForward/include)
include_directories(common/include)
include_directories(preprocessing/include)
include_directories(consolidate/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/UTDecoding/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/setup/include)

add_library(SciFi STATIC
  ${scifi_common}
  ${scifi_preprocessing}
  ${scifi_tracking}
  ${scifi_consolidate}
)
set_property(TARGET SciFi PROPERTY CUDA_SEPARABLE_COMPILATION ON)
