
file(GLOB UT_decoding "UTDecoding/src/*cu")
file(GLOB UT_sorting "sorting/src/*cu")
file(GLOB veloUT_tracking "PrVeloUT/src/*cu")

include_directories(../../main/include)
include_directories(../velo/common/include)
include_directories(../SciFi/common/include)
include_directories(../utils/include)
include_directories(common/include)
include_directories(UTDecoding/include)
include_directories(PrVeloUT/include)
include_directories(sorting/include)

cuda_add_library(UT STATIC
  ${UT_decoding}
  ${UT_sorting}
  ${veloUT_tracking}
)


if ( MC_CHECK )
  target_compile_definitions(UT PUBLIC MC_CHECK)
endif()
