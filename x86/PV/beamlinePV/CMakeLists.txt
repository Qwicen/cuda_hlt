include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/velo/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/utils/float_operations/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/velo/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/beamlinePV/include)
include_directories(${CMAKE_SOURCE_DIR}/main/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)
include_directories(${CMAKE_SOURCE_DIR}/stream/setup/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB x86beamlinePV "src/*cpp")

add_library(x86beamlinePV STATIC
  ${x86beamlinePV}
  )
set_property(TARGET x86beamlinePV PROPERTY
             CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if ( ROOT_FOUND )
  target_compile_definitions(x86beamlinePV PUBLIC WITH_ROOT)
  target_include_directories(x86beamlinePV PUBLIC ${ROOT_INCLUDE_DIRS})
  target_link_libraries(x86beamlinePV
    ${ROOT_LIBRARIES}
    )
endif()
