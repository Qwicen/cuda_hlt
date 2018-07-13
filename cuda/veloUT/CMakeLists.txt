file(GLOB veloUT_common "common/src/*cu")
file(GLOB veloUT_tracking "PrVeloUT/src/*cu")

include_directories(common/include)
include_directories(PrVeloUT/include)

cuda_add_library(VeloUT STATIC
  ${veloUT_common}
  ${veloUT_tracking}
)

if ( MC_CHECK )
  target_compile_definitions(VeloUT PUBLIC MC_CHECK)
endif()
