#add_compile_options(-x CUDA)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -x c++)

file(GLOB veloUT_common "common/src/*cu")
file(GLOB veloUT_tracking "PrVeloUT/src/*cu")
#file(GLOB veloUT_tracking_cpp "PrVeloUT/src/*cpp")

include_directories(common/include)
include_directories(PrVeloUT/include)
include_directories(../../main/include)
include_directories(../velo/common/include)

cuda_add_library(VeloUT STATIC
  ${veloUT_common}
  ${veloUT_tracking}
)

if ( MC_CHECK )
  target_compile_definitions(VeloUT PUBLIC MC_CHECK)
endif()
