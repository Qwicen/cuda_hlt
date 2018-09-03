include_directories(include)
include_directories(../main/include)
include_directories(../cuda/veloUT/common/include)
include_directories(../cuda/forward/common/include)
include_directories(../cuda/velo/common/include)

cuda_add_library(Forward SHARED
  src/PrForward.cpp
  )

