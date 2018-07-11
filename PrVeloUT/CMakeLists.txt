include_directories(include)
include_directories(../cuda/veloUT/common/include)

cuda_add_library(VeloUT STATIC
  src/PrVeloUT.h
  src/PrVeloUT.cpp
  ../cuda/veloUT/common/src/VeloUTDefinitions.cu
  )

