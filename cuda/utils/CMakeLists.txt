#file(GLOB prefix_sum "prefix_sum/src/*cu" "../UT/common/src/*cu" "../SciFi/common/src/*cu")
file(GLOB prefix_sum "prefix_sum/src/*cu")

include_directories(prefix_sum/include/)

include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/UT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include)

add_library(Utils STATIC
  ${prefix_sum}
)

target_link_libraries(Utils PRIVATE
  UT
  SciFi
)

set_property(TARGET Utils PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)
