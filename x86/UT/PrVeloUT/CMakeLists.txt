include_directories(include)
include_directories(../../../cuda/velo/common/include)
include_directories(../../../cuda/UT/UTDecoding/include)
include_directories(../../../cuda/UT/PrVeloUT/include)
include_directories(../../../cuda/UT/common/include)
include_directories(../../../cuda/utils/sorting/include)
include_directories(../../../cuda/event_model/velo/include)
include_directories(../../../cuda/event_model/common/include)
include_directories(../../../main/include)
include_directories(../../../cuda/SciFi/common/include)
include_directories(../../../cuda/muon/common/include)
include_directories(../../../checker/tracking/include)

file(GLOB x86VeloUT_cpp "src/*cpp")

cuda_add_library(x86VeloUT STATIC
  ${x86VeloUT_cpp}
  )

if ( ROOT_FOUND )
  target_compile_definitions(x86VeloUT PUBLIC WITH_ROOT)
  target_link_libraries(x86VeloUT
    ${ROOT_LIBRARIES}
  )
endif()
