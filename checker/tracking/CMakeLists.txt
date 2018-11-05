include_directories(include)
include_directories(../../main/include)
include_directories(../../cuda/UT/PrVeloUT/include)



file(GLOB tracking_checker_sources "src/*cpp")

add_library(TrackChecking SHARED
  ${tracking_checker_sources}
)

if ( ROOT_FOUND )
  target_compile_definitions(TrackChecking PUBLIC WITH_ROOT)
endif()
