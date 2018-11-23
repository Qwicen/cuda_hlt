include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/UT/PrVeloUT/include)
include_directories(${ROOT_INCLUDE_DIRS})

file(GLOB pv_checker_sources "src/*cpp")

cuda_add_library(PVChecking SHARED
  ${pv_checker_sources}
)

if(ROOT_FOUND)
  target_compile_definitions(PVChecking PUBLIC WITH_ROOT)
endif()
