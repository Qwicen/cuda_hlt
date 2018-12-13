include_directories(include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${ROOT_INCLUDE_DIRS})

file(GLOB pv_checker_sources "src/*cpp")

add_library(PVChecking SHARED
  ${pv_checker_sources}
)
set_target_properties(PVChecking PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ROOT_FOUND)
  target_compile_definitions(PVChecking PRIVATE WITH_ROOT)
  target_include_directories(PVChecking BEFORE PRIVATE
    ${ROOT_INCLUDE_DIRS}
  )
  target_link_libraries(PVChecking PRIVATE
    ${ROOT_LIBRARIES}
  )
endif()
