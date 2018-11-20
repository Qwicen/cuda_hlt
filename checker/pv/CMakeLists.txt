include_directories(include)
include_directories(../../main/include)
include_directories(../../cuda/velo/common/include)
include_directories(../../cuda/PV/common/include)
include_directories(../../cuda/event_model/velo/include)
include_directories(../../cuda/event_model/common/include)
include_directories(../../cuda/UT/PrVeloUT/include)
include_directories(../../cuda/PV/beamlinePV/include) 



file(GLOB pv_checker_sources "src/*cpp")

cuda_add_library(PVChecking SHARED
  ${pv_checker_sources}
)

if ( ROOT_FOUND )
  target_include_directories(PVChecking PUBLIC ${ROOT_INCLUDE_DIRS}) 
  target_compile_definitions(PVChecking PUBLIC WITH_ROOT)
endif()
