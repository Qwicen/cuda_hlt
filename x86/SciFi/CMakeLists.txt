include_directories(include)
include_directories(../../checker/tracking/include)
include_directories(../../main/include)
include_directories(../../cuda/SciFi/PrForward/include)
include_directories(../../cuda/UT/common/include)
include_directories(../../cuda/UT/PrVeloUT/include)
include_directories(../../cuda/UT/UTDecoding/include)
include_directories(../../cuda/SciFi/common/include)
include_directories(../../cuda/Muon/common/include)
include_directories(../../cuda/velo/common/include)
include_directories(../../cuda/event_model/common/include)
include_directories(../../cuda/event_model/velo/include)

file(GLOB x86Forward "src/*cpp" "../../cuda/SciFi/PrForward/src/*cu")

cuda_add_library(x86Forward STATIC
  ${x86Forward}
)
if ( ROOT_FOUND )
  target_compile_definitions(x86Forward PUBLIC WITH_ROOT)
  target_link_libraries(x86Forward
    ${ROOT_LIBRARIES}
  )
endif()
