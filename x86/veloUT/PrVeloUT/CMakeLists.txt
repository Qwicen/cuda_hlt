include_directories(include)
include_directories(../../../cuda/veloUT/PrVeloUT/include)

cuda_add_library(x86VeloUT STATIC
  src/PrVeloUTWrapper.cpp
  ../../../cuda/veloUT/PrVeloUT/src/PrVeloUT.cpp
  ../../../cuda/veloUT/common/src/VeloUTDefinitions.cpp
  )

