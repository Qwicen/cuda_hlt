include_directories(include)
#include_directories(../cuda/veloUT/common/include)
#include_directories(../cuda/velo/common/include)

cuda_add_library(x86VeloUT STATIC
  src/PrVeloUT.cpp
  ../../../cuda/veloUT/common/src/VeloUTDefinitions.cu
  )

