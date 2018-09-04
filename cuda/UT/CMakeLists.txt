
file(GLOB UT_decoding "UTDecoding/src/*cu")
file(GLOB veloUT_tracking "PrVeloUT/src/*cu")
file(GLOB veloUT_sorting "sorting/src/*cu")

include_directories(../../main/include)
include_directories(../velo/common/include)
include_directories(../utils/include)
include_directories(common/include)
include_directories(PrVeloUT/include)
include_directories(sorting/include)
include_directories(UTDecoding/include)

cuda_add_library(VeloUT STATIC
  ${UT_decoding}
  ${veloUT_tracking}
  ${veloUT_sorting}
)

if ( MC_CHECK )
  target_compile_definitions(VeloUT PUBLIC MC_CHECK)
endif()
