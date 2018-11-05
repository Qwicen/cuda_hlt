include_directories(include)
include_directories(../../main/include)
include_directories(../../cuda/UT/PrVeloUT/include)
include_directories(../../cuda/UT/common/include)
include_directories(../../cuda/UT/common/include)
include_directories(../../cuda/velo/common/include)
include_directories(../../cuda/event_model/velo/include)
include_directories(../../cuda/event_model/common/include)
include_directories(../../cuda/SciFi/common/include)
include_directories(../../cuda/SciFi/PrForward/include)



file(GLOB tracking_checker_sources "src/*cpp")

cuda_add_library(TrackChecking SHARED
  ${tracking_checker_sources}
)

if ( ROOT_FOUND )
  target_compile_definitions(TrackChecking PUBLIC WITH_ROOT)
endif()
