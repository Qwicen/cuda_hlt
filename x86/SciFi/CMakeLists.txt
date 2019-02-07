include_directories(include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/checker/tracking/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/UTDecoding/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/UT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/sorting)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/setup/include)

file(GLOB x86Forward_src "src/*cpp" "${CMAKE_SOURCE_DIR}/cuda/SciFi/common/src/*cu")
file(GLOB PrForwardTools "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/FindStereoHits.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/FindXHits.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/HitUtils.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/LinearFitting.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/ParabolaFitting.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/PrForwardTools.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/ReferencePlaneProjection.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/TrackUtils.cu")

add_library(x86Forward STATIC
  ${x86Forward_src}
  ${PrForwardTools}
)

set_property(TARGET x86Forward PROPERTY
             CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET x86Forward PROPERTY
             CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if ( ROOT_FOUND )
  target_compile_definitions(x86Forward PUBLIC WITH_ROOT)
  target_link_libraries(x86Forward PRIVATE
    ${ROOT_LIBRARIES}
  )
  target_include_directories(x86Forward BEFORE PRIVATE
    ${ROOT_INCLUDE_DIRS}
  )
endif()
