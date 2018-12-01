include_directories(include)
include_directories(../../../cuda/velo/common/include)
include_directories(../../../cuda/PV/common/include)
include_directories(../../../cuda/event_model/velo/include)
include_directories(../../../cuda/event_model/common/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda/PV/beamlinePV/include)
include_directories(../../../main/include)
include_directories(../../../stream/gear/include)


file(GLOB x86beamlinePV "src/*cpp")

cuda_add_library(x86beamlinePV STATIC
  ${x86beamlinePV}
  )

if ( ROOT_FOUND )
  target_compile_definitions(x86beamlinePV PUBLIC WITH_ROOT)
  target_include_directories(x86beamlinePV PUBLIC ${ROOT_INCLUDE_DIRS}) 
  target_link_libraries(x86beamlinePV
    ${ROOT_LIBRARIES}
  )
endif()
