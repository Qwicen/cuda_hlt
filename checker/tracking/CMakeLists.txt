include_directories(include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/UT/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/SciFi/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/SciFi/PrForward/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/kalman/ParKalman/include)
include_directories(${ROOT_INCLUDE_DIRS})

file(GLOB tracking_checker_sources "src/*cpp")

add_library(TrackChecking STATIC
  ${tracking_checker_sources}
)
set_target_properties(TrackChecking PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ROOT_FOUND)
  target_compile_definitions(TrackChecking PRIVATE WITH_ROOT)
  target_include_directories(TrackChecking BEFORE PRIVATE
    ${ROOT_INCLUDE_DIRS}
  )
  target_link_libraries(TrackChecking PRIVATE
    ${ROOT_LIBRARIES}
  )
endif()

if(USE_KALMAN_SINGLE_PRECISION)
  target_compile_definitions(TrackChecking PRIVATE KALMAN_SINGLE_PRECISION)
endif()
