include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/UTDecoding/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/sorting/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/checker/tracking/include)

file(GLOB x86VeloUT_cpp "src/*cpp")

add_library(x86VeloUT STATIC
  ${x86VeloUT_cpp}
  )

set_property(TARGET x86VeloUT PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)

if ( ROOT_FOUND )
  target_compile_definitions(x86VeloUT PUBLIC WITH_ROOT)
  target_include_directories(x86VeloUT BEFORE PRIVATE
    ${ROOT_INCLUDE_DIRS}
  )
  target_link_libraries(x86VeloUT PRIVATE
    ${ROOT_LIBRARIES}
  )
endif()
