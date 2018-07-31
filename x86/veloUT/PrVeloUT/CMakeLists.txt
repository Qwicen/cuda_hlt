include_directories(include)
include_directories(../../../cuda/veloUT/PrVeloUT/include)
include_directories(../../../cuda/velo/common/include)
include_directories(../../../cuda/veloUT/common/include)
include_directories(../../../main/include)

cuda_add_library(x86VeloUT STATIC
  src/PrVeloUTWrapper.cpp
  ../../../cuda/veloUT/PrVeloUT/src/PrVeloUT.cu
  )

