include_directories(include)
include_directories(../cuda/velo/common/include)
include_directories(../cuda/patPV/include)

cuda_add_library(PatPV SHARED
  src/PVSeedTool.cpp
  src/AdaptivePV3DFitter.cpp
  
  )

