include_directories(include)
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
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/sorting)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)

file(GLOB x86Forward "src/*cpp" "${CMAKE_SOURCE_DIR}/cuda/SciFi/common/src/*cu")
file(GLOB PrForwardTools "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/FindStereoHits.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/FindXHits.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/HitUtils.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/LinearFitting.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/ParabolaFitting.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/PrForwardTools.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/ReferencePlaneProjection.cu"
  "${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/src/TrackUtils.cu")

cuda_add_library(x86Forward STATIC
  ${x86Forward}
  ${PrForwardTools}
)

if ( ROOT_FOUND )
  target_compile_definitions(x86Forward PUBLIC WITH_ROOT)
  target_link_libraries(x86Forward
    ${ROOT_LIBRARIES}
  )
  target_include_directories(x86Forward PUBLIC ${ROOT_INCLUDE_DIRS})
endif()
